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

#include "FullSystem/FullSystem.h"
 
#include "stdio.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <algorithm>
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>

#include "FullSystem/ResidualProjections.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "FullSystem/HessianBlocks.h"

namespace dso
{
int PointFrameResidual::instanceCounter = 0;


long runningResID=0;


PointFrameResidual::PointFrameResidual(){assert(false); instanceCounter++;}

PointFrameResidual::~PointFrameResidual(){assert(efResidual==0); instanceCounter--; delete J;}

PointFrameResidual::PointFrameResidual(PointHessian* point_, FrameHessian* host_, FrameHessian* target_) :
	point(point_),
	host(host_),
	target(target_)
{
	efResidual=0;
	instanceCounter++;
	resetOOB();
	J = new RawResidualJacobian();
	assert(((long)J)%16==0);

	isNew=true;
}




double PointFrameResidual::linearize(CalibHessian* HCalib)
{
	state_NewEnergyWithOutlier=-1;

	if(state_state == ResState::OOB)
		{ state_NewState = ResState::OOB; return state_energy; }

	FrameFramePrecalc* precalc = &(host->targetPrecalc[target->idx]);
	float energyLeft=0;
	const Eigen::Vector3f* dIl = target->dI;
	//const float* const Il = target->I;
	const Mat33f &PRE_KRKiTll = precalc->PRE_KRKiTll;
	const Vec3f &PRE_KtTll = precalc->PRE_KtTll;
	const Mat33f &PRE_RTll_0 = precalc->PRE_RTll_0;
	const Vec3f &PRE_tTll_0 = precalc->PRE_tTll_0;
	const float * const color = point->color;
	const float * const weights = point->weights;

	Vec2f affLL = precalc->PRE_aff_mode;
	float b0 = precalc->PRE_b0_mode;


	Vec6f d_xi_x, d_xi_y;
	Vec4f d_C_x, d_C_y;
	float d_d_x, d_d_y;
	{
		float drescale, u, v, new_idepth;
		float Ku, Kv;
		Vec3f KliP;

		// 注意此处的投影是用x=0时刻的逆深度值及位姿，便于下面计算几何部分的FEJ
		if(!projectPoint(point->u, point->v, point->idepth_zero_scaled, 0, 0,HCalib,
				PRE_RTll_0,PRE_tTll_0, drescale, u, v, Ku, Kv, KliP, new_idepth))
			{ state_NewState = ResState::OOB; return state_energy; }

		centerProjectedTo = Vec3f(Ku, Kv, new_idepth);


		// diff d_idepth
		// (∂uT/∂u) * (∂u/∂ρH) 投影到target帧上的像素点横坐标对逆深度求导
		d_d_x = drescale * (PRE_tTll_0[0]-PRE_tTll_0[2]*u)*SCALE_IDEPTH*HCalib->fxl();
		// (∂vT/∂v) * (∂v/∂ρH) 投影到target帧上的像素点纵坐标对逆深度求导
		d_d_y = drescale * (PRE_tTll_0[1]-PRE_tTll_0[2]*v)*SCALE_IDEPTH*HCalib->fyl();




		// diff calib
		// 令 PH = Ki * (1/ρH) * xH, 即像素点xH在host帧中的相机坐标, (u, v, 1.0) 是投影到target帧中归一化的三维坐标
		//
		//       -  uT -              -  u  -
		//      |   vT  | = xT = K * |   v   |    = K * ρT * [ RTH * PH + tTH ]
		//       - 1.0 -              - 1.0 -
		//                                        = K * ρT * [ RTH * Ki * (1/ρH) * xH + tTH ]
		//
		//                       - fx * u + cx -     - fx * ρT * (r1 * PH + t1) + cx -
		//                    = |  fy * v + cy  | = |  fy * ρT * (r2 * PH + t2) + cy  |
		//                       -     1.0     -     -      ρT * (r3 * PH + t2)      -
		//
		//                                   1
		//  u = ρT * (r1 * PH + t1) = ---------------- * (r1 * PH + t1)
		//                             (r3 * PH + t3)
		//
		//                                   1
		//  v = ρT * (r2 * PH + t2) = ---------------- * (r2 * PH + t2)
		//                             (r3 * PH + t3)
		//
		//                      r3                                   1
		//  ∂u/∂PH = - ------------------ * (r1 * PH + t1) + ---------------- * r1
		//              (r3 * PH + t3)^2                      (r3 * PH + t3)
		//
		//         = r1 * ρT - r3 * ρT^2 * (r1 * PH + t1)
		//
		//         = ρT * [ r1 - r3 * u ]1x3
		//
		//                      r3                                   1
		//  ∂v/∂PH = - ------------------ * (r2 * PH + t2) + ---------------- * r2
		//              (r3 * PH + t3)^2                      (r3 * PH + t3)
		//
		//         = r2 * ρT - r3 * ρT^2 * (r2 * PH + t2)
		//
		//         = ρT * [ r2 - r3 * v ]1x3
		//
		//
		//         1     -   uH/fx - cx/fx  -      1     -  KliP[0]  -
		//  PH = ---- * |    vH/fy - cy/fy   | = ---- * |   KliP[1]   |
		//        ρH     -        1.0       -     ρH     -    1.0    -
		//
		//             1     -   -(1/fx) * (uH/fx - cx/fx)   -
		// ∂PH/∂fx = ---- * |               0                 |
		//            ρH     -              0                -3x1
		//
		//             1     -              0                -
		// ∂PH/∂fy = ---- * |    -(1/fy) * (vH/fx - cy/fy)    |
		//            ρH     -              0                -3x1
		//
		//             1     -           -(1/fx)             -
		// ∂PH/∂cx = ---- * |               0                 |
		//            ρH     -              0                -3x1
		//
		//             1     -              0                -
		// ∂PH/∂cy = ---- * |            -(1/fy)              |
		//            ρH     -              0                -3x1
		//
		// 因此,
		//
		//  ∂uT/∂fx = u + fx * (∂u/∂fx) = u + fx * [(∂u/∂PH)*(∂PH/∂fx)]
		//          = u + fx * (ρT/ρH) * (r11 - r31 * u) * [ -(1/fx) * (uH/fx - cx/fx) ]
		//          = u + drescale * (r31 * u - r11) * Klip[0]
		//
		//  ∂uT/∂fy = fx * (∂u/∂fy) = fx * [(∂u/∂PH)*(∂PH/∂fy)]
		//          = fx * (ρT/ρH) * (r12 - r32 * u) * [ -(1/fy) * (vH/fy - cy/fy) ]
		//          = fx * drescale * (r32 * u - r12) * (1/fy) * Klip[1]
		//
		//  ∂uT/∂cx = 1 + fx * (∂u/∂cx) = 1 + fx * [(∂u/∂PH)*(∂PH/∂cx)]
		//          = 1 + fx * (ρT/ρH) * (r11 - r31 * u) * [ -(1/fx) ]
		//          = 1 + drescale * (r31 * u - r11)
		//
		//  ∂uT/∂cy = fx * (∂u/∂cy) = fx * [(∂u/∂PH)*(∂PH/∂cy)]
		//          = fx * (ρT/ρH) * (r12 - r32 * u) * [ -(1/fy) ]
		//          = fx * drescale * (r32 * u - r12) * (1/fy)
		//
		//
		//  ∂vT/∂fx = fy * (∂v/∂fx) = fy * [(∂v/∂PH)*(∂PH/∂fx)]
		//          = fy * (ρT/ρH) * (r21 - r31 * v) * [ -(1/fx) * (uH/fx - cx/fx) ]
		//          = fy * drescale * (r31 * v - r21) * Klip[0] * (1/fx)
		//
		//  ∂vT/∂fy = v + fy * (∂v/∂fy) = v + fy * [(∂v/∂PH)*(∂PH/∂fy)]
		//          = v + fy * (ρT/ρH) * (r22 - r32 * v) * [ -(1/fy) * (vH/fy - cy/fy) ]
		//          = v + drescale * (r32 * v - r22) * Klip[1]
		//
		//  ∂vT/∂cx = fy * (∂v/∂cx) = fy * [(∂v/∂PH)*(∂PH/∂cx)]
		//          = fy * (ρT/ρH) * (r21 - r31 * v) * [ -(1/fx) ]
		//          = fy * drescale * (r31 * v - r21) * (1/fx)
		//
		//  ∂vT/∂cy = 1 + fy * (∂v/∂cy) = 1 + fy * [(∂v/∂PH)*(∂PH/∂cy)]
		//          = 1 + fy * (ρT/ρH) * (r22 - r32 * v) * [ -(1/fy) ]
		//          = 1 + drescale * (r32 * v - r22)
		//
		d_C_x[2] = drescale*(PRE_RTll_0(2,0)*u-PRE_RTll_0(0,0));
		d_C_x[3] = HCalib->fxl() * drescale*(PRE_RTll_0(2,1)*u-PRE_RTll_0(0,1)) * HCalib->fyli();
		d_C_x[0] = KliP[0]*d_C_x[2];
		d_C_x[1] = KliP[1]*d_C_x[3];

		d_C_y[2] = HCalib->fyl() * drescale*(PRE_RTll_0(2,0)*v-PRE_RTll_0(1,0)) * HCalib->fxli();
		d_C_y[3] = drescale*(PRE_RTll_0(2,1)*v-PRE_RTll_0(1,1));
		d_C_y[0] = KliP[0]*d_C_y[2];
		d_C_y[1] = KliP[1]*d_C_y[3];

		d_C_x[0] = (d_C_x[0]+u)*SCALE_F;
		d_C_x[1] *= SCALE_F;
		d_C_x[2] = (d_C_x[2]+1)*SCALE_C;
		d_C_x[3] *= SCALE_C;

		d_C_y[0] *= SCALE_F;
		d_C_y[1] = (d_C_y[1]+v)*SCALE_F;
		d_C_y[2] *= SCALE_C;
		d_C_y[3] = (d_C_y[3]+1)*SCALE_C;


		// (∂uT/∂ξ): 投影到target帧上的横坐标分别对李代数的6个项求导
		d_xi_x[0] = new_idepth*HCalib->fxl();
		d_xi_x[1] = 0;
		d_xi_x[2] = -new_idepth*u*HCalib->fxl();
		d_xi_x[3] = -u*v*HCalib->fxl();
		d_xi_x[4] = (1+u*u)*HCalib->fxl();
		d_xi_x[5] = -v*HCalib->fxl();
		// (∂vT/∂ξ): 投影到target帧上的纵坐标分别对李代数的6个项求导
		d_xi_y[0] = 0;
		d_xi_y[1] = new_idepth*HCalib->fyl();
		d_xi_y[2] = -new_idepth*v*HCalib->fyl();
		d_xi_y[3] = -(1+v*v)*HCalib->fyl();
		d_xi_y[4] = u*v*HCalib->fyl();
		d_xi_y[5] = u*HCalib->fyl();
	}


	{
        // 几何Jacobian仅计算中心点，pattern中其它点都以此为近似
		J->Jpdxi[0] = d_xi_x;
		J->Jpdxi[1] = d_xi_y;

		J->Jpdc[0] = d_C_x;
		J->Jpdc[1] = d_C_y;

		J->Jpdd[0] = d_d_x;
		J->Jpdd[1] = d_d_y;

	}






	float JIdxJIdx_00=0, JIdxJIdx_11=0, JIdxJIdx_10=0;
	float JabJIdx_00=0, JabJIdx_01=0, JabJIdx_10=0, JabJIdx_11=0;
	float JabJab_00=0, JabJab_01=0, JabJab_11=0;

	float wJI2_sum = 0;

	for(int idx=0;idx<patternNum;idx++)
	{
		float Ku, Kv;
        // 计算image derivative(∂I/xT)用current value
		if(!projectPoint(point->u+patternP[idx][0], point->v+patternP[idx][1], point->idepth_scaled, PRE_KRKiTll, PRE_KtTll, Ku, Kv))
			{ state_NewState = ResState::OOB; return state_energy; }

		projectedTo[idx][0] = Ku;
		projectedTo[idx][1] = Kv;


        Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));
        float residual = hitColor[0] - (float)(affLL[0] * color[idx] + affLL[1]);



		float drdA = (color[idx]-b0); // 光度部分也是用x=0时刻来计算FEJ
		if(!std::isfinite((float)hitColor[0]))
		{ state_NewState = ResState::OOB; return state_energy; }


		float w = sqrtf(setting_outlierTHSumComponent / (setting_outlierTHSumComponent + hitColor.tail<2>().squaredNorm()));
        w = 0.5f*(w + weights[idx]);



		float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
		energyLeft += w*w*hw *residual*residual*(2-hw);

		{
			if(hw < 1) hw = sqrtf(setting_hw_multiplier*hw);
			hw = hw*w;

			hitColor[1]*=hw;
			hitColor[2]*=hw;

            // 此处res尚未乘以J^T
			J->resF[idx] = residual*hw;

			J->JIdx[0][idx] = hitColor[1];
			J->JIdx[1][idx] = hitColor[2];
            // 计算光度部分Jacobian
            //                           tT * e^aT
            // r = w * [ (I[xT] - bT) - ----------- * (I[xH] - bH) ]
            //                           tH * e^aH
            //           tT * e^aT
            // 令 aTH = ------------ 为host到target的相对亮度密度变换参数
            //           tH * e^aH
            //
            //               tT * e^aT
            // (∂r/∂aT) = - ----------- * (I[xH] - bH) * w = -aTH * (I[xH] - bH) * w
            //               tH * e^aH                             ------- ↓↓ ------
            //                                                           JabF[0]
            // (∂r/∂bT) = -1 * 1 * w
            //                -- ↓ --
            //                JabF[1]
            //
            //             tT * e^aT
            // (∂r/∂aH) = ----------- * (I[xH] - bH) * w = aTH * (I[xH] - bH) * w
            //             tH * e^aH                             ------ ↓↓ ------
            //                                                        JabF[0]
            //             tT * e^aT
            // (∂r/∂bH) = ----------- * w = aTH * 1 * w
            //             tH * e^aH             -- ↓ --
            //                                   JabF[1]
            //
            // 这里代码中的JabF只记录了灰度部分，aTH在adHost和adTarget对角线上光度对应位置有记录
            //
			J->JabF[0][idx] = drdA*hw;
			J->JabF[1][idx] = hw;

			JIdxJIdx_00+=hitColor[1]*hitColor[1];
			JIdxJIdx_11+=hitColor[2]*hitColor[2];
			JIdxJIdx_10+=hitColor[1]*hitColor[2];

			JabJIdx_00+= drdA*hw * hitColor[1];
			JabJIdx_01+= drdA*hw * hitColor[2];
			JabJIdx_10+= hw * hitColor[1];
			JabJIdx_11+= hw * hitColor[2];

			JabJab_00+= drdA*drdA*hw*hw;
			JabJab_01+= drdA*hw*hw;
			JabJab_11+= hw*hw;


			wJI2_sum += hw*hw*(hitColor[1]*hitColor[1]+hitColor[2]*hitColor[2]);

			if(setting_affineOptModeA < 0) J->JabF[0][idx]=0;
			if(setting_affineOptModeB < 0) J->JabF[1][idx]=0;

		}
	}

	J->JIdx2(0,0) = JIdxJIdx_00;
	J->JIdx2(0,1) = JIdxJIdx_10;
	J->JIdx2(1,0) = JIdxJIdx_10;
	J->JIdx2(1,1) = JIdxJIdx_11;
	J->JabJIdx(0,0) = JabJIdx_00;
	J->JabJIdx(0,1) = JabJIdx_01;
	J->JabJIdx(1,0) = JabJIdx_10;
	J->JabJIdx(1,1) = JabJIdx_11;
	J->Jab2(0,0) = JabJab_00;
	J->Jab2(0,1) = JabJab_01;
	J->Jab2(1,0) = JabJab_01;
	J->Jab2(1,1) = JabJab_11;

	state_NewEnergyWithOutlier = energyLeft;

	if(energyLeft > std::max<float>(host->frameEnergyTH, target->frameEnergyTH) || wJI2_sum < 2)
	{
		energyLeft = std::max<float>(host->frameEnergyTH, target->frameEnergyTH);
		state_NewState = ResState::OUTLIER;
	}
	else
	{
		state_NewState = ResState::IN;
	}

	state_NewEnergy = energyLeft;
	return energyLeft;
}



void PointFrameResidual::debugPlot()
{
	if(state_state==ResState::OOB) return;
	Vec3b cT = Vec3b(0,0,0);

	if(freeDebugParam5==0)
	{
		float rT = 20*sqrt(state_energy/9);
		if(rT<0) rT=0; if(rT>255)rT=255;
		cT = Vec3b(0,255-rT,rT);
	}
	else
	{
		if(state_state == ResState::IN) cT = Vec3b(255,0,0);
		else if(state_state == ResState::OOB) cT = Vec3b(255,255,0);
		else if(state_state == ResState::OUTLIER) cT = Vec3b(0,0,255);
		else cT = Vec3b(255,255,255);
	}

	for(int i=0;i<patternNum;i++)
	{
		if((projectedTo[i][0] > 2 && projectedTo[i][1] > 2 && projectedTo[i][0] < wG[0]-3 && projectedTo[i][1] < hG[0]-3 ))
			target->debugImage->setPixel1((float)projectedTo[i][0], (float)projectedTo[i][1],cT);
	}
}



void PointFrameResidual::applyRes(bool copyJacobians)
{
	if(copyJacobians)
	{
		if(state_state == ResState::OOB)
		{
			assert(!efResidual->isActiveAndIsGoodNEW);
			return;	// can never go back from OOB
		}
		if(state_NewState == ResState::IN)// && )
		{
			efResidual->isActiveAndIsGoodNEW=true;
			efResidual->takeDataF();
		}
		else
		{
			efResidual->isActiveAndIsGoodNEW=false;
		}
	}

	setState(state_NewState);
	state_energy = state_NewEnergy;
}
}
