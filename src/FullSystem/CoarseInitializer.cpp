/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */
#include <stdlib.h>
#include <opencv2/highgui.hpp>
#include "FullSystem/CoarseInitializer.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "util/nanoflann.h"


#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso {

    CoarseInitializer::CoarseInitializer(int ww, int hh) : thisToNext_aff(0, 0), thisToNext(SE3()) {
        for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
            points[lvl] = 0;
            numPoints[lvl] = 0;
        }

        JbBuffer = new Vec10f[ww * hh];
        JbBuffer_new = new Vec10f[ww * hh];


        frameID = -1;
        fixAffine = false;
        printDebug = false;

//	wM.diagonal()[0] = wM.diagonal()[1] = wM.diagonal()[2] = SCALE_XI_ROT;
//	wM.diagonal()[3] = wM.diagonal()[4] = wM.diagonal()[5] = SCALE_XI_TRANS;
        wM.diagonal()[0] = wM.diagonal()[1] = wM.diagonal()[2] = SCALE_XI_TRANS;
        wM.diagonal()[3] = wM.diagonal()[4] = wM.diagonal()[5] = SCALE_XI_ROT;
        wM.diagonal()[6] = SCALE_A;
        wM.diagonal()[7] = SCALE_B;
    }

    CoarseInitializer::~CoarseInitializer() {
        for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
            if (points[lvl] != 0) delete[] points[lvl];
        }

        delete[] JbBuffer;
        delete[] JbBuffer_new;
    }


    bool CoarseInitializer::trackFrame(FrameHessian *newFrameHessian, std::vector<IOWrap::Output3DWrapper *> &wraps) {
        newFrame = newFrameHessian;

        for (IOWrap::Output3DWrapper *ow : wraps)
            ow->pushLiveFrame(newFrameHessian);

        int maxIterations[] = {5, 5, 10, 30, 50};


        alphaK = 2.5 * 2.5;//*freeDebugParam1*freeDebugParam1;
        alphaW = 150 * 150;//*freeDebugParam2*freeDebugParam2;
        regWeight = 0.8;//*freeDebugParam4;
        couplingWeight = 1;//*freeDebugParam5;

        if (!snapped) {
//		thisToNext.translation().setZero();
            thisToNext = SE3();
            for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
                int npts = numPoints[lvl];
                Pnt *ptsl = points[lvl];
                for (int i = 0; i < npts; i++) {
                    ptsl[i].iR = 1;
                    ptsl[i].idepth_new = 1;
                    ptsl[i].lastHessian = 0;
                }
            }
        }


        SE3 refToNew_current = thisToNext;
        AffLight refToNew_aff_current = thisToNext_aff;

        if (firstFrame->ab_exposure > 0 && newFrame->ab_exposure > 0) {
//		printf("firstFrame->ab_exposure:%.4f, newFrame->ab_exposure:%.4f \n",
//			   firstFrame->ab_exposure, newFrame->ab_exposure);
            refToNew_aff_current = AffLight(logf(newFrame->ab_exposure / firstFrame->ab_exposure),
                                            0); // coarse approximation.
        }


        Vec3f latestRes = Vec3f::Zero();
        for (int lvl = pyrLevelsUsed - 1; lvl >= 0; lvl--) {


            if (lvl < pyrLevelsUsed - 1)
                propagateDown(lvl + 1);

            Mat88f H, Hsc;
            Vec8f b, bsc;

            resetPoints(lvl);
            Vec3f resOld = calcResAndGS(lvl, H, b, Hsc, bsc, refToNew_current, refToNew_aff_current, false);
            applyStep(lvl);

            float lambda = 0.1;
            float eps = 1e-4;
            int fails = 0;

            if (printDebug) {
                printf("lvl %d, it %d (l=%f) %s: %.3f+%.5f -> %.3f+%.5f (%.3f->%.3f) (|inc| = %f)! \t",
                       lvl, 0, lambda,
                       "INITIA",
                       sqrtf((float) (resOld[0] / resOld[2])),
                       sqrtf((float) (resOld[1] / resOld[2])),
                       sqrtf((float) (resOld[0] / resOld[2])),
                       sqrtf((float) (resOld[1] / resOld[2])),
                       (resOld[0] + resOld[1]) / resOld[2],
                       (resOld[0] + resOld[1]) / resOld[2],
                       0.0f);
                std::cout << refToNew_current.log().transpose() << " AFF " << refToNew_aff_current.vec().transpose()
                          << "\n";
            }

            int iteration = 0;
            while (true) {
                Mat88f Hl = H;
                for (int i = 0; i < 8; i++) Hl(i, i) *= (1 + lambda);
                Hl -= Hsc * (1 / (1 + lambda));
                Vec8f bl = b - bsc * (1 / (1 + lambda));

                Hl = wM * Hl * wM * (0.01f / (w[lvl] * h[lvl]));
                bl = wM * bl * (0.01f / (w[lvl] * h[lvl]));


                Vec8f inc;
                if (fixAffine) {
                    inc.head<6>() = -(wM.toDenseMatrix().topLeftCorner<6, 6>() *
                                      (Hl.topLeftCorner<6, 6>().ldlt().solve(bl.head<6>())));
                    inc.tail<2>().setZero();
                } else
                    inc = -(wM * (Hl.ldlt().solve(bl)));    //=-H^-1 * b.


                SE3 refToNew_new = SE3::exp(inc.head<6>().cast<double>()) * refToNew_current;
                AffLight refToNew_aff_new = refToNew_aff_current;
                refToNew_aff_new.a += inc[6];
                refToNew_aff_new.b += inc[7];
                doStep(lvl, lambda, inc);


                Mat88f H_new, Hsc_new;
                Vec8f b_new, bsc_new;
                Vec3f resNew = calcResAndGS(lvl, H_new, b_new, Hsc_new, bsc_new, refToNew_new, refToNew_aff_new, false);
                Vec3f regEnergy = calcEC(lvl);

                float eTotalNew = (resNew[0] + resNew[1] + regEnergy[1]);
                float eTotalOld = (resOld[0] + resOld[1] + regEnergy[0]);


                bool accept = eTotalOld > eTotalNew;

                if (printDebug) {
                    printf("lvl %d, it %d (l=%f) %s: %.5f + %.5f + %.5f -> %.5f + %.5f + %.5f (%.2f->%.2f) (|inc| = %f)! \t",
                           lvl, iteration, lambda,
                           (accept ? "ACCEPT" : "REJECT"),
                           sqrtf((float) (resOld[0] / resOld[2])),
                           sqrtf((float) (regEnergy[0] / regEnergy[2])),
                           sqrtf((float) (resOld[1] / resOld[2])),
                           sqrtf((float) (resNew[0] / resNew[2])),
                           sqrtf((float) (regEnergy[1] / regEnergy[2])),
                           sqrtf((float) (resNew[1] / resNew[2])),
                           eTotalOld / resNew[2],
                           eTotalNew / resNew[2],
                           inc.norm());
                    std::cout << refToNew_new.log().transpose() << " AFF " << refToNew_aff_new.vec().transpose()
                              << "\n";
                }

                if (accept) {

                    if (resNew[1] == alphaK * numPoints[lvl]) {
                        if (!snapped)
                            printf("snapped -> true At level_%d iteration_%d!\n", lvl, iteration);
                        snapped = true;
                    }
                    H = H_new;
                    b = b_new;
                    Hsc = Hsc_new;
                    bsc = bsc_new;
                    resOld = resNew;
                    refToNew_aff_current = refToNew_aff_new;
                    refToNew_current = refToNew_new;
                    applyStep(lvl);
                    optReg(lvl);
                    lambda *= 0.5;
                    fails = 0;
                    if (lambda < 0.0001) lambda = 0.0001;
                } else {
                    fails++;
                    lambda *= 4;
                    if (lambda > 10000) lambda = 10000;
                }

                bool quitOpt = false;

                if (!(inc.norm() > eps) || iteration >= maxIterations[lvl] || fails >= 2) {
                    Mat88f H, Hsc;
                    Vec8f b, bsc;

                    quitOpt = true;
                    if (!(inc.norm() > eps)) {
                        printf("quit level_%d iteration_%d because of !(inc.norm():%.6f > eps:%.6f)\n", lvl, iteration,
                               inc.norm(), eps);
                    } else if (iteration >= maxIterations[lvl]) {
                        printf("quit level_%d iteration_%d because of iteration(%d) >= maxIterations(%d), inc.norm:%.6f\n",
                               lvl, iteration, iteration, maxIterations[lvl], inc.norm());
                    } else if (fails >= 2) {
                        printf("quit level_%d iteration_%d because of fails >= 2, inc.norm:%.6f\n", lvl, iteration,
                               inc.norm());
                    }
                }


                if (quitOpt) break;
                iteration++;
            }
            latestRes = resOld;

        }


        thisToNext = refToNew_current;
        thisToNext_aff = refToNew_aff_current;

        for (int i = 0; i < pyrLevelsUsed - 1; i++)
            propagateUp(i);


        frameID++;
        // snapped变为true之前, snappedAt永远为0
        if (!snapped) snappedAt = 0;
        // snapped第一次为true时, snappedAt被设置为当时的frameID
        if (snapped && snappedAt == 0)
            snappedAt = frameID;


        printf("snapped(%s) && frameID(%d) > snappedAt(%d)+5 = %s\n",
               snapped ? "true" : "false", frameID, snappedAt,
               (snapped && frameID > snappedAt + 5) ? "true" : "false");
        debugPlot(0, wraps);


        return snapped && frameID > snappedAt + 5;
    }

    void CoarseInitializer::debugPlot(int lvl, std::vector<IOWrap::Output3DWrapper *> &wraps) {
        bool needCall = true;
        for (IOWrap::Output3DWrapper *ow : wraps)
            needCall = needCall || ow->needPushDepthImage();
        if (!needCall) return;


        int wl = w[lvl], hl = h[lvl];
        Eigen::Vector3f *colorRef = firstFrame->dIp[lvl];

        MinimalImageB3 iRImg(wl, hl);

        for (int i = 0; i < wl * hl; i++)
            iRImg.at(i) = Vec3b(colorRef[i][0], colorRef[i][0], colorRef[i][0]);


        int npts = numPoints[lvl];

        float nid = 0, sid = 0;
        for (int i = 0; i < npts; i++) {
            Pnt *point = points[lvl] + i;
            if (point->isGood) {
                nid++;
                sid += point->iR;
            }
        }
        printf("lvl_%d: good:%.2f\n", lvl, nid);
        float fac = nid / sid;


        for (int i = 0; i < npts; i++) {
            Pnt *point = points[lvl] + i;

            if (!point->isGood)
                iRImg.setPixel9(point->u + 0.5f, point->v + 0.5f, Vec3b(0, 0, 0));

            else
                iRImg.setPixel9(point->u + 0.5f, point->v + 0.5f, makeRainbow3B(point->iR * fac));
        }


//	IOWrap::displayImage("idepth-R", &iRImg, false);
        for (IOWrap::Output3DWrapper *ow : wraps)
            ow->pushDepthImage(&iRImg);
    }

// calculates residual, Hessian and Hessian-block neede for re-substituting depth.
    Vec3f CoarseInitializer::calcResAndGS(
            int lvl, Mat88f &H_out, Vec8f &b_out,
            Mat88f &H_out_sc, Vec8f &b_out_sc,
            const SE3 &refToNew, AffLight refToNew_aff,
            bool plot) {
        int wl = w[lvl], hl = h[lvl];
        Eigen::Vector3f *colorRef = firstFrame->dIp[lvl];
        Eigen::Vector3f *colorNew = newFrame->dIp[lvl];

        Mat33f RKi = (refToNew.rotationMatrix() * Ki[lvl]).cast<float>();
        Vec3f t = refToNew.translation().cast<float>();
        Eigen::Vector2f r2new_aff = Eigen::Vector2f(exp(refToNew_aff.a), refToNew_aff.b);

        float fxl = fx[lvl];
        float fyl = fy[lvl];
        float cxl = cx[lvl];
        float cyl = cy[lvl];


        Accumulator11 E;
        acc9.initialize();
        E.initialize();


        int npts = numPoints[lvl];
        Pnt *ptsl = points[lvl];
        for (int i = 0; i < npts; i++) {

            Pnt *point = ptsl + i;

            point->maxstep = 1e10;
            if (!point->isGood) {
                E.updateSingle((float) (point->energy[0]));
                point->energy_new = point->energy;
                point->isGood_new = false;
                continue;
            }

            EIGEN_ALIGN32 VecNRf dp0;
            EIGEN_ALIGN32 VecNRf dp1;
            EIGEN_ALIGN32 VecNRf dp2;
            EIGEN_ALIGN32 VecNRf dp3;
            EIGEN_ALIGN32 VecNRf dp4;
            EIGEN_ALIGN32 VecNRf dp5;
            EIGEN_ALIGN32 VecNRf dp6;
            EIGEN_ALIGN32 VecNRf dp7;
            EIGEN_ALIGN32 VecNRf dd;
            EIGEN_ALIGN32 VecNRf r;
            JbBuffer_new[i].setZero();

            // sum over all residuals.
            bool isGood = true;
            float energy = 0;
            // Reference: https://zhuanlan.zhihu.com/p/29177540
            for (int idx = 0; idx < patternNum; idx++) {
                int dx = patternP[idx][0];
                int dy = patternP[idx][1];


                // 逆深度 ρT = 1/dT, ρH = 1/dH, 像素坐标 xH = (uH, vH, 1.0), xT = (uT, vT, 1.0)t
                // 令 PT = (XT, YT, ZT)t 为投影点在target相机坐标系中的三维坐标，
                // i.e. PT = (XT, YT, ZT)t = [ RTH * Ki * (1/ρH) * xH + tTH ]
                //                         = (1/ρH) * [ RTH * Ki * xH + tTH * ρH ]
                // and let's make pt = (pt[0], pt[1], pt[2])t = [ RTH * Ki * xH + tTH * ρH ]   ---- ①
                // pt可以理解为像素xH在host帧中归一化的三维坐标投影到target帧上的三维点
                // then we get PT = (1/ρH) * pt,
                // 归一化, PT = (1/ρH) * pt[2] * (pt[0]/pt[2], pt[1]/pt[2], 1.0)t
                //           = (1/ρH) * pt[2] * (u, v, 1.0)t     ---- ② ③
                // so (u, v, 1.0) 其实是投影点在target相机坐标系中归一化后的三维坐标
                //
                // target帧中投影点的像素齐次坐标:
                // 		(1/ρT) * (uT, vT, 1.0)t = (1/ρT) * xT = K * [ RTH * Ki * (1/ρH) * xH + tTH ]	 ★★★★★
                // i.e. (uT, vT, 1.0)t = xT = ρT * K * PT
                //                          = ρT * K * [ (1/ρH) * pt[2] * (u, v, 1.0)t ]
                //                          = (ρT/ρH) * pt[2] * K * (u, v, 1.0)t
                //                          = (ρT/ρH) * pt[2] * (fx * u + cx, fy * v + cy, 1.0)t
                //                          = (ρT/ρH) * pt[2] * (Ku, Kv, 1.0)t   ---- ④ ⑤
                // easily derive that 1.0 = (ρT/ρH) * pt[2], hence ρT = ρH / pt[2]		---- ⑥
                // and uT = Ku, vT = Kv, i.e. Ku, Kv是target帧中投影点的像素坐标
                Vec3f pt = RKi * Vec3f(point->u + dx, point->v + dy, 1) + t * point->idepth_new;    // ①
                float u = pt[0] / pt[2];                                                        // ②
                float v = pt[1] / pt[2];                                                        // ③
                float Ku = fxl * u + cxl;                                                        // ④
                float Kv = fyl * v + cyl;                                                        // ⑤
                float new_idepth = point->idepth_new / pt[2];                                    // ⑥

                if (!(Ku > 1 && Kv > 1 && Ku < wl - 2 && Kv < hl - 2 && new_idepth > 0)) {
                    isGood = false;
                    break;
                }

                Vec3f hitColor = getInterpolatedElement33(colorNew, Ku, Kv, wl);
                //Vec3f hitColor = getInterpolatedElement33BiCub(colorNew, Ku, Kv, wl);

//			float rlR = colorRef[int(point->u+dx + (point->v+dy) * wl)][0];
                float rlR = getInterpolatedElement31(colorRef, point->u + dx, point->v + dy, wl);

                if (!std::isfinite(rlR) || !std::isfinite((float) hitColor[0])) {
                    isGood = false;
                    break;
                }


                float residual = hitColor[0] - r2new_aff[0] * rlR - r2new_aff[1];
                // Huber norm:
                // http://sepwww.stanford.edu/public/docs/sep121/paper_html/node12.html#SECTION00220000000000000000
                // The Huber norm is an hybrid l1^2 / l2^2 error measure that is robust to outliers.
                //
                //				/ (r^2)/2				, 0 ≤ |r| ≤ α
                //		Hα(r) =|
                //				\ α * (|r| - (α/2))		, |r| > α
                //
                // Huber loss: https://en.wikipedia.org/wiki/Huber_loss
                // This function is quadratic for small values of r, and linear for large values,
                // Here:
                //
                //				/ r^2					, 0 ≤ |r| ≤ α  (since hw = 1)
                //		Hα(r) =|
                //				\ 2 * α * (|r| - (α/2))	, |r| > α (since hw = α/|r|)
                //
                float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
                energy += hw * residual * residual * (2 - hw);


                // 残值表达式: r = I(xT) - a * I(xH) - b
                // Jacobian求解:
                //   1. 图像雅可比: ∂I(xT)/∂xT
                //   2. 几何雅可比, 投影像素坐标相对于相机位姿及点的逆深度变化率: ∂xT/∂(δξ), ∂xT/∂ρH
                //   3. 光度雅可比: ∂r/∂a, ∂r/∂b
                //
                // (一) 几何部分
                // [1] 对逆深度求导 ∂xT/∂ρH
                //	   上面已经得到 [u, v, 1.0]t 是投影点在target帧中相机坐标系下归一化后的三维坐标
                //       -  uT -			  -  u  -
                //      |   vT  | = xT = K * |   v   | = K * (ρT/ρH) * [ RTH * Ki * xH + tTH * ρH ]
                //		 - 1.0 -			  - 1.0 -   观察到RTH * Ki * xH 项与ρH无关, 令其为常数项C(3x1的列向量)
                //									   = K * (ρT/ρH) * [ C + tTH * ρH ]
                //						 	  			   	  - (ρT/ρH) * (C[0] + tTH[0] * ρH) -
                //									   = K * |  (ρT/ρH) * (C[1] + tTH[1] * ρH)  |
                //							  	 		      - (ρT/ρH) * (C[2] + tTH[2] * ρH) -
                //
                //						 	  			   	  - (ρT/ρH) * (C[0] + tTH[0] * ρH) -
                //									   = K * |  (ρT/ρH) * (C[1] + tTH[1] * ρH)  |
                //							  	 		      - 			 1.0		       -
                //
                // 有一个未知量ρT需要替换
                // 由于 第三行 1.0 = (ρT/ρH) * (C[2] + tTH[2] * ρH)
                // 可得 (ρT/ρH) = (C[2] + tTH[2] * ρH)^(-1)
                // 因此
                // 		u = [ (C[2] + tTH[2] * ρH)^(-1) ] * [ C[0] + tTH[0] * ρH ]
                //      v = [ (C[2] + tTH[2] * ρH)^(-1) ] * [ C[1] + tTH[1] * ρH ]
                //
                //  ∂u/∂ρH = (C[2] + tTH[2] * ρH)^(-1) * tTH[0] + (C[0] + tTH[0] * ρH) * [ -tTH[2] * (C[2] + tTH[2] * ρH)^(-2) ]
                //         = (C[2] + tTH[2] * ρH)^(-1) * [ tTH[0] -  tTH[2] * (C[0] + tTH[0] * ρH) / (C[2] + tTH[2] * ρH) ]
                //         = (1/pt[2]) * [ tTH[0] - tTH[2] * (pt[0]/pt[2])]
                //         = (tTH[0] - tTH[2] * u) / pt[2]   ----  ①
                //  ∂v/∂ρH = (tTH[1] - tTH[2] * v) / pt[2]   ----  ②
                //
                //						 -	(∂uT/∂u) * (∂u/∂ρH)	-     - fx*((tTH[0] - tTH[2]*u)/pt[2]) -
                // 根据链式法则 ∂xT/∂ρH = |  	  					 | = |	 								|	--- ⑭
                //  					 -	(∂vT/∂v) * (∂v/∂ρH) -     - fy*((tTH[1] - tTH[2]*v)/pt[2]) -
                //
                //
                // [2] 对位姿求导 ∂xT/∂ξ
                // 根据链式法则 ∂xT/∂(δξ) = (∂xT/∂PT) * (∂PT/∂(δξ))
                // a. 像素对空间点 ∂xT/∂PT
                //     -  uT -			    -  u  -     - fx * u + cx -	    - fx * XT/ZT + cx -
                //    |   vT  | = xT = K * |   v   | = |  fy * v + cy  | = |  fy * YT/ZT + cy  |
                //	   - 1.0 -			  	- 1.0 - 	-       1.0   -     _ 		1.0 	  -
                //
                //              | ∂uT/∂XT   ∂uT/∂YT  ∂uT/∂ZT  |   | fx/ZT     0     -fx*XT/ZT^2 |
                //    ∂xT/∂PT = |                             | = |                             |
                //              | ∂vT/∂XT   ∂vT/∂YT  ∂vT/∂ZT  |   |   0     fy/ZT   -fy*YT/ZT^2 |
                //
                //
                // b. 投影点对李代数 ∂PT/∂ρξ                      -   0   -w3   w2  -
                //    向量 w = [w1, w2, w3]t 的反对称矩阵 [w]x =  |   w3   0    -w1  |
                //                                              -  -w2   w1    0  -
                //
                //                 -                   -        | 1   0   0    0   ZT   -YT |
                //   ∂PT/∂(δξ) =  |  I3x3      	-[PT]x  |	= 	| 0   1   0  -ZT   0     XT |
                //                 -                   -3x6     | 0   0   1   YT  -XT    0 |
                //
                //   综上, ∂xT/∂(δξ) = (∂xT/∂PT) * (∂PT/∂(δξ))
                //                    | fx/ZT     0     -fx*XT/ZT^2 |     | 1   0   0   0   ZT   -YT |
                //                  = |                             |  *  | 0   1   0  -ZT   0    XT |
                //                    |   0     fy/ZT   -fy*YT/ZT^2 |     | 0   0   1   YT  -XT   0  |
                //
                //                    | fx/ZT     0     -fx*XT/ZT^2     -fx*XT*YT/ZT^2     fx*(1+XT^2/ZT^2)   -fx*YT/ZT  |
                //                  = |                                                                                  |
                //                    |   0     fy/ZT   -fy*YT/ZT^2    -fy*(1+YT^2/ZT^2)    fy*XT*YT/ZT^2      fy*XT/ZT  |2x6
                //
                //                    | fx*ρT     0     -fx*u*ρT     -fx*u*v     	fx*(1+u^2)    -fx*v  |
                //                  = |                                                                  |
                //                    |   0     fy*ρT   -fy*v*ρT    -fy*(1+v^2)       fy*u*v      fy*u   |2x6
                //
                //
                // (二) 图像部分
                // 灰度对像素
                // ∂I(xT)/∂xT = [ (I(uT+1, vT) - I(uT-1, vT))/2   (I(uT, vT+1) - I(uT, vT-1))/2 ]
                //            = [ dx, dy ]1x2
                //
                // (三) 光度部分
                // ∂r/∂a = -a * I(xH)    --- ⑪
                // ∂r/∂b = -1            --- ⑫
                //
                //
                // (四) 总结
                //      -                                                        -
                // J = | (∂I(xT)/∂xT) * [ ∂xT/∂(δξ)   ∂xT/∂ρH ]   (∂r/∂a   ∂r/∂b) |
                //      -                                                        -1x8
                // 把逆深度项放后面，因为其它项与帧相关，逆深度与点相关
                //      -                                                                -
                // J = | ∂I(xT)/∂xT * ∂xT/∂(δξ)     ∂r/∂a   ∂r/∂b   ∂I(xT)/∂xT * ∂xT/∂ρH  |
                //      - (图像部分) (几何位姿部分)     （光度部分）    (图像部分)(几何深度部分) -1x8
                //
                //  						  |  uT  |				    	-  XT / ZT  -			  -  u  -
                // 其中,  (1/ρT)xT = (1/ρT) * |   vT  | = K * PT = K * ZT * |   YT / ZT  | = K * ZT * |   v  |
                //                            |  1.0 |                  	-    1.0    -			  - 1.0 -
                // 可得 1/ZT = ρT, u = XT / ZT, v = YT / ZT
                // 令中间项 dxInterp = dx*fx , dyInterp = dy*fy  --- ③④
                //
                //
                // 前6项 i.e. ∂I(xT)/∂xT * ∂xT/∂(δξ) =  -          dx*fx*ρT             -      --- ⑤
                //                                     |          dy*fy*ρT              |     --- ⑥
                //                                     |    -(dx*fx*u + dy*fy*v)*ρT     |     --- ⑦
                //                                     |   -dx*fx*u*v - dy*fy*(1+v^2)   |     --- ⑧
                //                                     |   dx*fx*(1+u^2) + dy*fy*u*v	|     --- ⑨
                //                                     |   	  -dx*fx*v + dy*fy*u		|     --- ⑩
                //                                      -                       	   -1x6
                //
                //												   | (∂uT/∂v) * (∂u/∂ρH) |
                // 最后1项 i.e. ∂I(xT)/∂xT * ∂xT/∂ρH = ∂I(xT)/∂xT * | (∂vT/∂v) * (∂v/∂ρH) |
                //                                               | fx * ((tTH[0] - tTH[2]*u) / pt[2]) |
                //                                   = [dx dy] * |  			   			 	      |		--- ⑬
                //                                               | fy * ((tTH[1] - tTH[2]*v) / pt[2]) |
                //
                //
                float dxdd = (t[0] - t[2] * u) / pt[2];                // ①
                float dydd = (t[1] - t[2] * v) / pt[2];                // ②

                // Robust Estimation using M-estimation via Iteratively reweighted least squares(IRLS)
                // https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares
                // http://users.stat.umn.edu/~sandy/courses/8053/handouts/robust.pdf
                // https://www.sebastiansylvan.com/post/robustestimation/
                if (hw < 1) hw = sqrtf(setting_hw_multiplier * hw);                            //

                float dxInterp = hw * hitColor[1] * fxl;                // ③
                float dyInterp = hw * hitColor[2] * fyl;                // ④
                dp0[idx] = new_idepth * dxInterp;                        // ⑤
                dp1[idx] = new_idepth * dyInterp;                        // ⑥
                dp2[idx] = -new_idepth * (u * dxInterp + v * dyInterp);    // ⑦
                dp3[idx] = -u * v * dxInterp - (1 + v * v) * dyInterp;        // ⑧
                dp4[idx] = (1 + u * u) * dxInterp + u * v * dyInterp;            // ⑨
                dp5[idx] = -v * dxInterp + u * dyInterp;                // ⑩
                dp6[idx] = -hw * r2new_aff[0] * rlR;                 // ⑪
                dp7[idx] = -hw * 1;                                  // ⑫
                dd[idx] = dxInterp * dxdd + dyInterp * dydd;        // ⑬
                r[idx] = hw * residual;

                //  - dxdd*fxl -     - (∂uT/∂u) * (∂u/∂ρH) -                    令 - a -
                // |  			| = |					    | = ∂xT/∂PT = Vec2f = |     |
                //  - dydd*fxy -	 - (∂vT/∂v) * (∂v/∂ρH) -                       - b -
                //
                // 这里Vec2f代表的物理意义是: 投影的像素点位置如何随逆深度值的变化而变化
                // i.e. 若逆深度变化ΔρH, 则target帧上投影点像素坐标横、纵坐标分别变化: (a*ΔρH, b*ΔρH)
                // 由于前面计算像素梯度时, dx dy都跨越了2个像素范围
                // 因此 [(a*ΔρH)^2 + (b*ΔρH)^2]max = 2^2 + 2^2 = 8
                // i.e. (ΔρH)max = sqrt[8/(a^2 + b^2)]    	--- ⑭
//			float maxstep = 1.0f / Vec2f(dxdd*fxl, dydd*fyl).norm();   // ⑭
                float maxstep = sqrtf(8.0) / Vec2f(dxdd * fxl, dydd * fyl).norm();   // ⑭
                if (maxstep < point->maxstep) point->maxstep = maxstep;

                // immediately compute dp*dd' and dd*dd' in JbBuffer1.
                JbBuffer_new[i][0] += dp0[idx] * dd[idx];
                JbBuffer_new[i][1] += dp1[idx] * dd[idx];
                JbBuffer_new[i][2] += dp2[idx] * dd[idx];
                JbBuffer_new[i][3] += dp3[idx] * dd[idx];
                JbBuffer_new[i][4] += dp4[idx] * dd[idx];
                JbBuffer_new[i][5] += dp5[idx] * dd[idx];
                JbBuffer_new[i][6] += dp6[idx] * dd[idx];
                JbBuffer_new[i][7] += dp7[idx] * dd[idx];
                JbBuffer_new[i][8] += r[idx] * dd[idx];
                JbBuffer_new[i][9] += dd[idx] * dd[idx];
            }

            if (!isGood || energy > point->outlierTH * 20) {
                E.updateSingle((float) (point->energy[0]));
                point->isGood_new = false;
                point->energy_new = point->energy;
                continue;
            }


            // add into energy.
            E.updateSingle(energy);
            point->isGood_new = true;
            point->energy_new[0] = energy;

            // update Hessian matrix.
            for (int i = 0; i + 3 < patternNum; i += 4)
                acc9.updateSSE(
                        _mm_load_ps(((float *) (&dp0)) + i),
                        _mm_load_ps(((float *) (&dp1)) + i),
                        _mm_load_ps(((float *) (&dp2)) + i),
                        _mm_load_ps(((float *) (&dp3)) + i),
                        _mm_load_ps(((float *) (&dp4)) + i),
                        _mm_load_ps(((float *) (&dp5)) + i),
                        _mm_load_ps(((float *) (&dp6)) + i),
                        _mm_load_ps(((float *) (&dp7)) + i),
                        _mm_load_ps(((float *) (&r)) + i));

            // no use at all
//		for(int i=((patternNum>>2)<<2); i < patternNum; i++)
//            acc9.updateSingle(
//                    (float) dp0[i], (float) dp1[i], (float) dp2[i], (float) dp3[i],
//                    (float) dp4[i], (float) dp5[i], (float) dp6[i], (float) dp7[i],
//                    (float) r[i]);

        }

        E.finish();
        acc9.finish();


        // Regularized Energy:(^T表示转置）
        // if !snapped
        //      EAlpha = alphaW*[ Σ(X0ρi+ΔXρi - 1)^2 + n*((X0tx+ΔXtx)^2 + (X0ty+ΔXty)^2 + (X0tz+ΔXtz)^2) ]
        //             = alphaW*[ (X0ρ+ΔXρ - 1)^T(X0ρ+ΔXρ - 1) + n*(X0t+ΔXt)^T(X0t+ΔXt) ]
        // else
        //      EAlpha = alphaK*n + Σ(X0ρi+ΔXρi - iR)^2
        //             = alphaK*n + (X0ρ+ΔXρ - iR)^T(X0ρ+ΔXρ - iR)
        //
        // hence, Total energy to minimize:
        //      E(ΔX) = Σ|ri(X0+ΔX)|h + EAlpha
        //
        // 线性化：
        // E(ΔX) ≈ Φ(ΔX) = Σ hwi*|ri(X0) +ri'(x0)*ΔX|2 + EAlpha
        //               = [J*ΔX + r(X0)]^T*HW*[J*ΔX + r(X0)] + EAlpha
        //
        // if !snapped
        //      ∂Φ(ΔX)/∂(ΔX) = 2*[(J*ΔX + r(X0))^T*HW*J] + alphaW*[2(X0ρ+ΔXρ - 1)^T + 2n(X0t+ΔXt)^T]
        // else
        //      ∂Φ(ΔX)/∂(ΔX) = 2*[(J*ΔX + r(X0))^T*HW*J] + 2(X0ρ+ΔXρ - iR)^T
        //
        // 令 ∂Φ(ΔX)/∂(ΔX) = 0
        // if !snapped                _ n*ΔXt _                            _ n*X0t _
        //      J^T*HW*J*ΔX + alphaW*|         | = -J^T*HW*r(X0) - alphaW*|         |
        //                            -  ΔXρ  -                            - X0ρ-1 -
        // else                _   _                            _       _
        //      J^T*HW*J*ΔX + |     | = -J^T*HW*r(X0) - alphaW*|         |
        //                     -ΔXρ-                            - X0ρ-iR -
        //
        //                                 - A^T -
        // 令 J = [ Anx8  Dnxn ], 则 J^T = |      |
        //                                 - D^T -
        // 上面的式子变成
        //
        //  - A^T*HW*A + 平移相关      A^T*HW*D -     - ΔXA -     -  -(A^T*HW*r + 平移相关)  -
        // |                                    | * |       | = |                           |
        //  - D^T*HW*A      D^T*HW*D + 逆深度相关-    - ΔXD  -    -  -(D^T*HW*r + 逆深度相关) -
        //
        // Schur补：
        // 将上式写成
        //  - U    C -     - ΔXA -     -  -εA -
        // |          | * |       | = |        |
        //  - C^T  V -     - ΔXD -     -  -εD -
        //
        //  - I    -C*V^-1 -     - U    C -     - ΔXA -     - I    -C*V^-1 -     -  -εA  -
        // |                | * |          | * |       | = |                | * |         |
        //  - 0          I -     - C^T  V -     - ΔXD -     - 0          I -     -  -εD  -
        //
        // 得 (U  -  C*V^-1*C) * ΔXA = - (εA - C*V^-1*εD)
        //      C^T*ΔXA + V*ΔXD = -εD  ==> ΔXD = -V^-1*(εD + C^T*ΔXA)
        //
        // 对应到代码中， H=U, JbBuffer_new[9]=V, Hsc=C*V^-1*C, b=εA, bsc=C*V^-1*εD
        //



        // calculate alpha energy, and decide if we cap it.
        Accumulator11 EAlpha;
        EAlpha.initialize();
        for (int i = 0; i < npts; i++) {
            Pnt *point = ptsl + i;
            if (!point->isGood_new) {
                EAlpha.updateSingle((float) (point->energy[1]));
            } else {
                point->energy_new[1] = (point->idepth_new - 1) * (point->idepth_new - 1);
                EAlpha.updateSingle((float) (point->energy_new[1]));
            }
        }
        EAlpha.finish();
        float alphaEnergy = alphaW * (EAlpha.A + refToNew.translation().squaredNorm() * npts);

//	printf("AE = %f * %f + %f\n", alphaW, EAlpha.A, refToNew.translation().squaredNorm() * npts);


        // compute alpha opt.
        float alphaOpt;
        if (alphaEnergy > alphaK * npts) {
            alphaOpt = 0;
            alphaEnergy = alphaK * npts;
        } else {
            alphaOpt = alphaW;
        }
//    printf("alphaOpt:%.2f, alphaEnergy:%.2f\n", alphaOpt, alphaEnergy);


        acc9SC.initialize();
        for (int i = 0; i < npts; i++) {
            Pnt *point = ptsl + i;
            if (!point->isGood_new)
                continue;

            point->lastHessian_new = JbBuffer_new[i][9];

            JbBuffer_new[i][8] += alphaOpt * (point->idepth_new - 1);
            JbBuffer_new[i][9] += alphaOpt;


            if (alphaOpt == 0) {
                JbBuffer_new[i][8] += couplingWeight * (point->idepth_new - point->iR);
                JbBuffer_new[i][9] += couplingWeight;
            }

            JbBuffer_new[i][9] = 1 / (1 + JbBuffer_new[i][9]);
//		JbBuffer_new[i][9] = 1/(JbBuffer_new[i][9]);
            acc9SC.updateSingleWeighted(
                    (float) JbBuffer_new[i][0], (float) JbBuffer_new[i][1], (float) JbBuffer_new[i][2],
                    (float) JbBuffer_new[i][3],
                    (float) JbBuffer_new[i][4], (float) JbBuffer_new[i][5], (float) JbBuffer_new[i][6],
                    (float) JbBuffer_new[i][7],
                    (float) JbBuffer_new[i][8], (float) JbBuffer_new[i][9]);
        }
        acc9SC.finish();


        //printf("nelements in H: %d, in E: %d, in Hsc: %d / 9!\n", (int)acc9.num, (int)E.num, (int)acc9SC.num*9);
        H_out = acc9.H.topLeftCorner<8, 8>();// / acc9.num;
        b_out = acc9.H.topRightCorner<8, 1>();// / acc9.num;
        H_out_sc = acc9SC.H.topLeftCorner<8, 8>();// / acc9.num;
        b_out_sc = acc9SC.H.topRightCorner<8, 1>();// / acc9.num;



        H_out(0, 0) += alphaOpt * npts;
        H_out(1, 1) += alphaOpt * npts;
        H_out(2, 2) += alphaOpt * npts;

        //Sophus: 平移在前，旋转在后
        Vec3f tlog = refToNew.log().head<3>().cast<float>();
        b_out[0] += tlog[0] * alphaOpt * npts;
        b_out[1] += tlog[1] * alphaOpt * npts;
        b_out[2] += tlog[2] * alphaOpt * npts;


        return Vec3f(E.A, alphaEnergy, E.num);
    }

    float CoarseInitializer::rescale() {
        float factor = 20 * thisToNext.translation().norm();
//	float factori = 1.0f/factor;
//	float factori2 = factori*factori;
//
//	for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
//	{
//		int npts = numPoints[lvl];
//		Pnt* ptsl = points[lvl];
//		for(int i=0;i<npts;i++)
//		{
//			ptsl[i].iR *= factor;
//			ptsl[i].idepth_new *= factor;
//			ptsl[i].lastHessian *= factori2;
//		}
//	}
//	thisToNext.translation() *= factori;

        return factor;
    }


    Vec3f CoarseInitializer::calcEC(int lvl) {
        if (!snapped) return Vec3f(0, 0, numPoints[lvl]);

        AccumulatorX<2> E;
        E.initialize();
        int npts = numPoints[lvl];
        for (int i = 0; i < npts; i++) {
            Pnt *point = points[lvl] + i;
            if (!point->isGood_new) continue;
            float rOld = (point->idepth - point->iR);
            float rNew = (point->idepth_new - point->iR);
            E.updateNoWeight(Vec2f(rOld * rOld, rNew * rNew));

            //printf("%f %f %f!\n", point->idepth, point->idepth_new, point->iR);
        }
        E.finish();

        //printf("ER: %f %f %f!\n", couplingWeight*E.A1m[0], couplingWeight*E.A1m[1], (float)E.num.numIn1m);
        return Vec3f(couplingWeight * E.A1m[0], couplingWeight * E.A1m[1], E.num);
    }

    void CoarseInitializer::optReg(int lvl) {
        int npts = numPoints[lvl];
        Pnt *ptsl = points[lvl];
        if (!snapped) {
            for (int i = 0; i < npts; i++)
                ptsl[i].iR = 1;
            return;
        }


        for (int i = 0; i < npts; i++) {
            Pnt *point = ptsl + i;
            if (!point->isGood) continue;

            float idnn[10];
            int nnn = 0;
            for (int j = 0; j < 10; j++) {
                if (point->neighbours[j] == -1) continue;
                Pnt *other = ptsl + point->neighbours[j];
                if (!other->isGood) continue;
                idnn[nnn] = other->iR;
                nnn++;
            }

            if (nnn > 2) {
                std::nth_element(idnn, idnn + nnn / 2, idnn + nnn);
                point->iR = (1 - regWeight) * point->idepth + regWeight * idnn[nnn / 2];
            }
        }

    }


    void CoarseInitializer::propagateUp(int srcLvl) {
        assert(srcLvl + 1 < pyrLevelsUsed);
        // set idepth of target

        int nptss = numPoints[srcLvl];
        int nptst = numPoints[srcLvl + 1];
        Pnt *ptss = points[srcLvl];
        Pnt *ptst = points[srcLvl + 1];

        // set to zero.
        for (int i = 0; i < nptst; i++) {
            Pnt *parent = ptst + i;
            parent->iR = 0;
            parent->iRSumNum = 0;
        }

        for (int i = 0; i < nptss; i++) {
            Pnt *point = ptss + i;
            if (!point->isGood) continue;

            Pnt *parent = ptst + point->parent;
            parent->iR += point->iR * point->lastHessian;
            parent->iRSumNum += point->lastHessian;
        }

        for (int i = 0; i < nptst; i++) {
            Pnt *parent = ptst + i;
            if (parent->iRSumNum > 0) {
                parent->idepth = parent->iR = (parent->iR / parent->iRSumNum);
                parent->isGood = true;
            }
        }

        optReg(srcLvl + 1);
    }

    void CoarseInitializer::propagateDown(int srcLvl) {
        assert(srcLvl > 0);
        // set idepth of target

        int nptst = numPoints[srcLvl - 1];
        Pnt *ptss = points[srcLvl];
        Pnt *ptst = points[srcLvl - 1];

        for (int i = 0; i < nptst; i++) {
            Pnt *point = ptst + i;
            Pnt *parent = ptss + point->parent;

            if (!parent->isGood || parent->lastHessian < 0.1) continue;
            if (!point->isGood) {
                point->iR = point->idepth = point->idepth_new = parent->iR;
                point->isGood = true;
                point->lastHessian = 0;
            } else {
                float newiR = (point->iR * point->lastHessian * 2 + parent->iR * parent->lastHessian) /
                              (point->lastHessian * 2 + parent->lastHessian);
                point->iR = point->idepth = point->idepth_new = newiR;
            }
        }
        optReg(srcLvl - 1);
    }


    void CoarseInitializer::makeGradients(Eigen::Vector3f **data) {
        for (int lvl = 1; lvl < pyrLevelsUsed; lvl++) {
            int lvlm1 = lvl - 1;
            int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

            Eigen::Vector3f *dINew_l = data[lvl];
            Eigen::Vector3f *dINew_lm = data[lvlm1];

            for (int y = 0; y < hl; y++)
                for (int x = 0; x < wl; x++)
                    dINew_l[x + y * wl][0] = 0.25f * (dINew_lm[2 * x + 2 * y * wlm1][0] +
                                                      dINew_lm[2 * x + 1 + 2 * y * wlm1][0] +
                                                      dINew_lm[2 * x + 2 * y * wlm1 + wlm1][0] +
                                                      dINew_lm[2 * x + 1 + 2 * y * wlm1 + wlm1][0]);

            for (int idx = wl; idx < wl * (hl - 1); idx++) {
                dINew_l[idx][1] = 0.5f * (dINew_l[idx + 1][0] - dINew_l[idx - 1][0]);
                dINew_l[idx][2] = 0.5f * (dINew_l[idx + wl][0] - dINew_l[idx - wl][0]);
            }
        }
    }

    void CoarseInitializer::setFirst(CalibHessian *HCalib, FrameHessian *newFrameHessian) {

        makeK(HCalib);
        firstFrame = newFrameHessian;

        PixelSelector sel(w[0], h[0]);

        float *statusMap = new float[w[0] * h[0]];
        bool *statusMapB = new bool[w[0] * h[0]];

        float densities[] = {0.03, 0.05, 0.15, 0.5, 1};
        for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
            sel.currentPotential = 3;
            int npts;
            if (lvl == 0)
                npts = sel.makeMaps(firstFrame, statusMap, densities[lvl] * w[0] * h[0], 1, false, 2);
            else
                npts = makePixelStatus(firstFrame->dIp[lvl], statusMapB, w[lvl], h[lvl], densities[lvl] * w[0] * h[0]);


            if (points[lvl] != 0) delete[] points[lvl];
            points[lvl] = new Pnt[npts];

            // set idepth map to initially 1 everywhere.
            int wl = w[lvl], hl = h[lvl];
            Pnt *pl = points[lvl];
            int nl = 0;
            for (int y = patternPadding + 1; y < hl - patternPadding - 2; y++)
                for (int x = patternPadding + 1; x < wl - patternPadding - 2; x++) {
                    //if(x==2) printf("y=%d!\n",y);
                    if ((lvl != 0 && statusMapB[x + y * wl]) || (lvl == 0 && statusMap[x + y * wl] != 0)) {
                        //assert(patternNum==9);
                        pl[nl].u = x + 0.1;
                        pl[nl].v = y + 0.1;
                        pl[nl].idepth = 1;
                        pl[nl].iR = 1;
                        pl[nl].isGood = true;
                        pl[nl].energy.setZero();
                        pl[nl].lastHessian = 0;
                        pl[nl].lastHessian_new = 0;
                        pl[nl].my_type = (lvl != 0) ? 1 : statusMap[x + y * wl];

                        Eigen::Vector3f *cpt = firstFrame->dIp[lvl] + x + y * w[lvl];
                        float sumGrad2 = 0;
                        for (int idx = 0; idx < patternNum; idx++) {
                            int dx = patternP[idx][0];
                            int dy = patternP[idx][1];
                            float absgrad = cpt[dx + dy * w[lvl]].tail<2>().squaredNorm();
                            sumGrad2 += absgrad;
                        }

//				float gth = setting_outlierTH * (sqrtf(sumGrad2)+setting_outlierTHSumComponent);
//				pl[nl].outlierTH = patternNum*gth*gth;


                        pl[nl].outlierTH = patternNum * setting_outlierTH;


                        nl++;
                        assert(nl <= npts);
                    }
                }


            numPoints[lvl] = nl;
            printf("level[%d] %d\n", lvl, nl);
        }
        delete[] statusMap;
        delete[] statusMapB;

        makeNN();

        thisToNext = SE3();
        snapped = false;
        frameID = snappedAt = 0;

//	for(int i=0;i<pyrLevelsUsed;i++)
//		dGrads[i].setZero();

    }

    void CoarseInitializer::resetPoints(int lvl) {
        Pnt *pts = points[lvl];
        int npts = numPoints[lvl];
        for (int i = 0; i < npts; i++) {
            pts[i].energy.setZero();
            pts[i].idepth_new = pts[i].idepth;


            if (lvl == pyrLevelsUsed - 1 && !pts[i].isGood) {
                float snd = 0, sn = 0;
                for (int n = 0; n < 10; n++) {
                    if (pts[i].neighbours[n] == -1 || !pts[pts[i].neighbours[n]].isGood) continue;
                    snd += pts[pts[i].neighbours[n]].iR;
                    sn += 1;
                }

                if (sn > 0) {
                    pts[i].isGood = true;
                    pts[i].iR = pts[i].idepth = pts[i].idepth_new = snd / sn;
                }
            }
        }
    }

    void CoarseInitializer::doStep(int lvl, float lambda, Vec8f inc) {

        const float maxPixelStep = 0.25;
        const float idMaxStep = 1e10;
        Pnt *pts = points[lvl];
        int npts = numPoints[lvl];
        for (int i = 0; i < npts; i++) {
            if (!pts[i].isGood) continue;


            float b = JbBuffer[i][8] + JbBuffer[i].head<8>().dot(inc);
            float step = -b * JbBuffer[i][9] / (1 + lambda);


            float maxstep = maxPixelStep * pts[i].maxstep;
            if (maxstep > idMaxStep) maxstep = idMaxStep;

            if (step > maxstep) step = maxstep;
            if (step < -maxstep) step = -maxstep;

            float newIdepth = pts[i].idepth + step;
            if (newIdepth < 1e-3) newIdepth = 1e-3;
            if (newIdepth > 50) newIdepth = 50;
            pts[i].idepth_new = newIdepth;
        }

    }

    void CoarseInitializer::applyStep(int lvl) {
        Pnt *pts = points[lvl];
        int npts = numPoints[lvl];
        for (int i = 0; i < npts; i++) {
            if (!pts[i].isGood) {
                pts[i].idepth = pts[i].idepth_new = pts[i].iR;
                continue;
            }
            pts[i].energy = pts[i].energy_new;
            pts[i].isGood = pts[i].isGood_new;
            pts[i].idepth = pts[i].idepth_new;
            pts[i].lastHessian = pts[i].lastHessian_new;
        }
        std::swap<Vec10f *>(JbBuffer, JbBuffer_new);
    }

    void CoarseInitializer::makeK(CalibHessian *HCalib) {
        w[0] = wG[0];
        h[0] = hG[0];

        fx[0] = HCalib->fxl();
        fy[0] = HCalib->fyl();
        cx[0] = HCalib->cxl();
        cy[0] = HCalib->cyl();

        for (int level = 1; level < pyrLevelsUsed; ++level) {
            w[level] = w[0] >> level;
            h[level] = h[0] >> level;
            fx[level] = fx[level - 1] * 0.5;
            fy[level] = fy[level - 1] * 0.5;
            if (plus_dot_five) {
                cx[level] = (cx[0] + 0.5) / ((int) 1 << level) - 0.5;
                cy[level] = (cy[0] + 0.5) / ((int) 1 << level) - 0.5;
            } else {
                cx[level] = cx[0] / ((int) 1 << level);
                cy[level] = cy[0] / ((int) 1 << level);
            }
        }

        for (int level = 0; level < pyrLevelsUsed; ++level) {
            K[level] << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
            Ki[level] = K[level].inverse();
            fxi[level] = Ki[level](0, 0);
            fyi[level] = Ki[level](1, 1);
            cxi[level] = Ki[level](0, 2);
            cyi[level] = Ki[level](1, 2);
        }
    }


    void CoarseInitializer::makeNN() {
        const float NNDistFactor = 0.05;

        typedef nanoflann::KDTreeSingleIndexAdaptor<
                nanoflann::L2_Simple_Adaptor<float, FLANNPointcloud>,
                FLANNPointcloud, 2> KDTree;

        // build indices
        FLANNPointcloud pcs[PYR_LEVELS];
        KDTree *indexes[PYR_LEVELS];
        for (int i = 0; i < pyrLevelsUsed; i++) {
            pcs[i] = FLANNPointcloud(numPoints[i], points[i]);
            indexes[i] = new KDTree(2, pcs[i], nanoflann::KDTreeSingleIndexAdaptorParams(5));
            indexes[i]->buildIndex();
        }

        const int nn = 10;

        // find NN & parents
        for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
            Pnt *pts = points[lvl];
            int npts = numPoints[lvl];

            int ret_index[nn];
            float ret_dist[nn];
            nanoflann::KNNResultSet<float, int, int> resultSet(nn);
            nanoflann::KNNResultSet<float, int, int> resultSet1(1);

            for (int i = 0; i < npts; i++) {
                //resultSet.init(pts[i].neighbours, pts[i].neighboursDist );
                resultSet.init(ret_index, ret_dist);
                Vec2f pt = Vec2f(pts[i].u, pts[i].v);
                indexes[lvl]->findNeighbors(resultSet, (float *) &pt, nanoflann::SearchParams());
                int myidx = 0;
                float sumDF = 0;
                for (int k = 0; k < nn; k++) {
                    pts[i].neighbours[myidx] = ret_index[k];
                    float df = expf(-ret_dist[k] * NNDistFactor);
                    sumDF += df;
                    pts[i].neighboursDist[myidx] = df;
                    assert(ret_index[k] >= 0 && ret_index[k] < npts);
                    myidx++;
                }
                for (int k = 0; k < nn; k++)
                    pts[i].neighboursDist[k] *= 10 / sumDF;


                if (lvl < pyrLevelsUsed - 1) {
                    resultSet1.init(ret_index, ret_dist);
                    pt = pt * 0.5f - Vec2f(0.25f, 0.25f);
                    indexes[lvl + 1]->findNeighbors(resultSet1, (float *) &pt, nanoflann::SearchParams());

                    pts[i].parent = ret_index[0];
                    pts[i].parentDist = expf(-ret_dist[0] * NNDistFactor);

                    assert(ret_index[0] >= 0 && ret_index[0] < numPoints[lvl + 1]);
                } else {
                    pts[i].parent = -1;
                    pts[i].parentDist = -1;
                }
            }
        }



        // done.

        for (int i = 0; i < pyrLevelsUsed; i++)
            delete indexes[i];
    }

    void CoarseInitializer::DislayPoints(int lvl) {
        int wl = w[lvl], hl = h[lvl];
        Eigen::Vector3f *colorRef = firstFrame->dIp[lvl];

        MinimalImageB3 iRImg(wl, hl);

        for (int i = 0; i < wl * hl; i++)
            iRImg.at(i) = Vec3b(colorRef[i][0], colorRef[i][0], colorRef[i][0]);


        int npts = numPoints[lvl];

        float nid = 0, sid = 0;
        for (int i = 0; i < npts; i++) {
            Pnt *point = points[lvl] + i;
            if (point->isGood) {
                nid++;
                sid += point->iR;
            }
        }
        float fac = nid / sid;


        for (int i = 0; i < npts; i++) {
            Pnt *point = points[lvl] + i;

            if (point->isGood)
//        if(!point->isGood)
//            iRImg.setPixel9(point->u+0.5f,point->v+0.5f,Vec3b(0,0,0));
//
//        else
                iRImg.setPixel9(point->u + 0.5f, point->v + 0.5f, makeRainbow3B(point->iR * fac));

        }

        IOWrap::displayImage("idepth-R", &iRImg, false);
        cv::waitKey(0);
    }

    void CoarseInitializer::DislayChosenPoints(int lvl, std::vector<int> &indexes) {
        int wl = w[lvl], hl = h[lvl];
        Eigen::Vector3f *colorRef = firstFrame->dIp[lvl];

        MinimalImageB3 iRImg(wl, hl);

        for (int i = 0; i < wl * hl; i++)
            iRImg.at(i) = Vec3b(colorRef[i][0], colorRef[i][0], colorRef[i][0]);


        int npts = numPoints[lvl];

        float max_iR = -1;
        float min_iR = std::numeric_limits<float>::max();
        float nid = 0, sid = 0;
        for (int i = 0; i < npts; i++) {
            Pnt *point = points[lvl] + i;
            if (point->isGood) {
                nid++;
                sid += point->iR;

                if (std::find(indexes.begin(), indexes.end(), i) != indexes.end()) {
                    if (point->idepth > max_iR) {
                        max_iR = point->idepth;
                    }
                    if (point->idepth < min_iR) {
                        min_iR = point->idepth;
                    }
                }
            }
        }
        float fac = nid / sid;


//	float GAP = max_iR - min_iR;
        float GAP = 1 / min_iR - 1 / max_iR;
        for (int i = 0; i < indexes.size(); i++) {
            int idx = indexes[i];
            Pnt *point = points[lvl] + idx;

//		if (point->isGood)
//        if(!point->isGood)
//            iRImg.setPixel9(point->u+0.5f,point->v+0.5f,Vec3b(0,0,0));
//
//        else
            iRImg.setPixel9(point->u + 0.5f, point->v + 0.5f, makeRainbow3B(point->iR * fac));

//		unsigned char value = uchar(point->iR / GAP * 255.0);
//		unsigned char value = uchar(1/point->idepth / GAP * 255.0);
//		unsigned char value = uchar( i / indexes.size() * 255.0);
//		iRImg.setPixel9(point->u+0.5f,point->v+0.5f, Vec3b(value, value, value));
        }

        IOWrap::displayImage("idepth-R", &iRImg, false);
        cv::waitKey(0);
    }

}

