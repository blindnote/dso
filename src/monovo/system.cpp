//
// Created by Yin Rochelle on 11/27/17.
//
#include <vector>
#include <fstream>
#include <iomanip>
#include "util/globalCalib.h"
#include "monovo/system.h"
#include "FullSystem/ImmaturePoint.h"


namespace dso
{

System::System(const float* GInv, bool linear)
: linear_(linear), keyframe_ids_(0), status_(Status::NOT_STARTED_YET)
{
    SetGammaFunction(GInv);
    sp_initializer_ = std::make_shared<CoarseInitializer>(wG[0], hG[0]);
}


System::~System()
{

}


bool System::AddFrame(ImageAndExposure *p_img, int id)
{
    assert(p_img != nullptr);

    FrameHessian* new_fh = new FrameHessian();
    new_fh->ab_exposure = p_img->exposure_time;
//    if (id == 1)
//    {
//        std::ofstream myfile;
//        float *color = p_img->image;
//        myfile.open("/Users/yinr/Desktop/me_processed_float_img.txt");
//        for (int i = 0; i < p_img->w * p_img->h; i++) {
//            myfile << "[" << i << "]" << std::setprecision(8) << color[i] << ",";
//            if ((i + 1) % p_img->w == 0) myfile << std::endl;
//        }
//        myfile.close();
//    }
    new_fh->makeImages(p_img->image, &hessian_calib_);

    // Display purpose Only!
//    {
//        for(auto lvl = 1; lvl < pyrLevelsUsed; lvl++)
//        {
//            float* pyr_lvl = new float[wG[lvl] * hG[lvl]];
//            for(auto p = 0; p < wG[lvl] * hG[lvl]; p++)
//            {
//                pyr_lvl[p] = p_fh->dIp[lvl][p][0];
//            }
//
//            char buf[20];
//            snprintf(buf, 20, "level_%d", lvl);
//            IOWrap::displayImage(buf, pyr_lvl, wG[lvl], hG[lvl], true);
//
//            delete pyr_lvl;
//        }
//    }

    std::vector<IOWrap::Output3DWrapper*> vec;

    switch (GetStatus())
    {
        case Status::NOT_STARTED_YET:
            printf("Try setting 1st frame_%d ...", id);
            sp_initializer_->setFirst(&hessian_calib_, new_fh);
            SetStatus(Status::START_INITIALIZATION);
            printf("Done!\n");
            break;

        case Status::START_INITIALIZATION:
            printf("Try initializing frame_%d ...\n", id);
            if (sp_initializer_->trackFrame(new_fh, vec))
            {
                InitializeWithPointsSorted(new_fh);
                SetStatus(Status::INITIALIZED);
            }
            break;

        case Status::INITIALIZED:
            printf("Start tracking on frame_%d ...\n", id);
            SetStatus(Status::ON_TRACK);
            break;

        case Status::ON_TRACK:
            break;

        case Status::LOST:
            printf("Already lost!! Stop processing frame_%d ...\n", id);
            return false;
    }

    return true;
}


void System::SetGammaFunction(const float* BInv)
{
    if (BInv == 0) return;

    // copy BInv.
    memcpy(hessian_calib_.Binv, BInv, sizeof(float) * 256);

    // invert.
    for (int i = 1; i < 255; i++)
    {
        // find val, such that Binv[val] = i.
        // I dont care about speed for this, so do it the stupid way.

        for (int s = 1; s < 255; s++)
        {
            if (BInv[s] <= i && BInv[s+1] >= i)
            {
                hessian_calib_.B[i] = s + (i - BInv[s]) / (BInv[s + 1] - BInv[s]);
                break;
            }
        }
    }

    hessian_calib_.B[0] = 0;
    hessian_calib_.B[255] = 255;
}


void System::InitializeWithPointsSorted(FrameHessian *p_fh)
{
    FrameHessian* firstFrame = sp_initializer_->firstFrame;
    firstFrame->idx = vec_hessian_frames.size();
    vec_hessian_frames.push_back(firstFrame);
    firstFrame->frameID = keyframe_ids_++;

    firstFrame->pointHessians.reserve(wG[0]*hG[0]*0.2f);
    firstFrame->pointHessiansMarginalized.reserve(wG[0]*hG[0]*0.2f);
    firstFrame->pointHessiansOut.reserve(wG[0]*hG[0]*0.2f);

    float sumID=1e-5, numID=1e-5;
    for (int i = 0; i < sp_initializer_->numPoints[0]; i++)
    {
        sumID += sp_initializer_->points[0][i].iR;
        numID++;
    }
    float rescaleFactor = 1 / (sumID / numID);

    // randomly sub-select the points I need.
    float keepPercentage = setting_desiredPointDensity / sp_initializer_->numPoints[0];

//    if (!setting_debugout_runquiet)
        printf("Initialization: keep %.1f%% (need %d, have %d)!\n", 100*keepPercentage,
           (int)(setting_desiredPointDensity), sp_initializer_->numPoints[0] );

    /////////////////////////////////////////////////////////////////
    struct tmpPnt
    {
        tmpPnt(bool quality, float U, float V, float type, float ir, float energy, int idx)
        {
            good = quality;
            u = U;
            v = V;
            my_type = type;
            iR = ir;
            energy0 = energy;
            index = idx;
        }
        bool good;
        float u, v;
        float my_type;
        float iR;
        float energy0;
        int index;

        bool operator<(const tmpPnt& p) const
        {
            return energy0 < p.energy0;
        }
    };

    std::vector<tmpPnt> points, bad_points;
    for(int i = 0; i < sp_initializer_->numPoints[0]; i++)
    {
        Pnt* point = sp_initializer_->points[0] + i;
        if (point->isGood)
            points.push_back(tmpPnt(point->isGood,
                                    point->u, point->v,
                                    point->my_type, point->iR,
                                    point->energy[0], i));
        else
            bad_points.push_back(tmpPnt(point->isGood,
                                         point->u, point->v,
                                         point->my_type, point->iR,
                                         point->energy[0], i));
    }
    std::sort(points.begin(), points.end());
    printf("good: %d\n", points.size());


    points.insert(std::end(points), std::begin(bad_points), std::end(bad_points));


    std::ofstream myfile;
    myfile.open ("/Users/yinr/Desktop/two_table_points_sorted.txt");
    myfile << std::setprecision(4);
    for(auto i = 0; i < points.size(); i++)
    {
        tmpPnt point = points[i];
        myfile << "[" << i << "]" << point.u << ", " << point.v
               << ", " << (point.good ? "true" : "false")
               << ", " << point.energy0 << std::endl;
    }
    myfile.close();
    /////////////////////////////////////////////////////////////////

    int chosen_count = 0;
    int bad_chosen_count = 0;
    std::vector<int> indexes;
    for (int i = 0; i < points.size() && chosen_count <= setting_desiredPointDensity; i++)
    {
        tmpPnt point = points[i];

        if (!point.good &&
            rand()/(float)RAND_MAX > keepPercentage)
            continue;

        ImmaturePoint* pt = new ImmaturePoint(point.u+0.5f, point.v+0.5f, firstFrame, point.my_type, &hessian_calib_);

        if (!std::isfinite(pt->energyTH)) { delete pt; continue; }


        pt->idepth_max=pt->idepth_min=1;
        PointHessian* ph = new PointHessian(pt, &hessian_calib_);
        delete pt;
        if(!std::isfinite(ph->energyTH)) {delete ph; continue;}

        chosen_count++;
        if (!point.good) bad_chosen_count++;
        indexes.push_back(point.index);

        ph->setIdepthScaled(point.iR * rescaleFactor);
        ph->setIdepthZero(ph->idepth);
        ph->hasDepthPrior = true;
        ph->setPointStatus(PointHessian::ACTIVE);

        firstFrame->pointHessians.push_back(ph);
//        ef->insertPoint(ph);
    }

    printf("INITIALIZE FROM INITIALIZER (%d pts: good:%d, bad:%d)!\n",
           (int)firstFrame->pointHessians.size(),
           (int)firstFrame->pointHessians.size() - bad_chosen_count,
           bad_chosen_count);

    sp_initializer_->DislayChosenPoints(0, indexes);
}


void System::Initialize(FrameHessian *p_fh)
{
    // add firstframe.
    FrameHessian* firstFrame = sp_initializer_->firstFrame;
    firstFrame->idx = vec_hessian_frames.size();
    vec_hessian_frames.push_back(firstFrame);
//    firstFrame->frameID = allKeyFramesHistory.size();
//    allKeyFramesHistory.push_back(firstFrame->shell);
    firstFrame->frameID = keyframe_ids_++;
//    ef->insertFrame(firstFrame, &hessian_calib_);
//    setPrecalcValues();

    firstFrame->pointHessians.reserve(wG[0]*hG[0]*0.2f);
    firstFrame->pointHessiansMarginalized.reserve(wG[0]*hG[0]*0.2f);
    firstFrame->pointHessiansOut.reserve(wG[0]*hG[0]*0.2f);


    float sumID=1e-5, numID=1e-5;
    for (int i = 0; i < sp_initializer_->numPoints[0]; i++)
    {
        sumID += sp_initializer_->points[0][i].iR;
        numID++;
    }
    float rescaleFactor = 1 / (sumID / numID);

    // randomly sub-select the points I need.
    float keepPercentage = setting_desiredPointDensity / sp_initializer_->numPoints[0];

//    if (!setting_debugout_runquiet)
        printf("Initialization: keep %.1f%% (need %d, have %d)!\n", 100*keepPercentage,
               (int)(setting_desiredPointDensity), sp_initializer_->numPoints[0] );

//    std::ofstream myfile;
//    myfile.open ("/Users/yinr/Desktop/two_table_points.txt");
//    myfile << std::setprecision(4);
    std::vector<int> indexes;
    for (int i = 0; i < sp_initializer_->numPoints[0]; i++)
    {
        if (rand()/(float)RAND_MAX > keepPercentage) continue;

        Pnt* point = sp_initializer_->points[0]+i;
        ImmaturePoint* pt = new ImmaturePoint(point->u+0.5f, point->v+0.5f, firstFrame, point->my_type, &hessian_calib_);

        if (!std::isfinite(pt->energyTH)) { delete pt; continue; }


        pt->idepth_max=pt->idepth_min=1;
        PointHessian* ph = new PointHessian(pt, &hessian_calib_);
        delete pt;
        if(!std::isfinite(ph->energyTH)) {delete ph; continue;}

//        myfile << "[" << i << "]" << point->u << ", " << point->v << std::endl;
        indexes.push_back(i);

        ph->setIdepthScaled(point->iR * rescaleFactor);
        ph->setIdepthZero(ph->idepth);
        ph->hasDepthPrior = true;
        ph->setPointStatus(PointHessian::ACTIVE);

        firstFrame->pointHessians.push_back(ph);
//        ef->insertPoint(ph);
    }
//    myfile.close();

//    SE3 firstToNew = coarseInitializer->thisToNext;
//    firstToNew.translation() /= rescaleFactor;
//
//
//    // really no lock required, as we are initializing.
//    {
//        boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
//        firstFrame->shell->camToWorld = SE3();
//        firstFrame->shell->aff_g2l = AffLight(0,0);
//        firstFrame->setEvalPT_scaled(firstFrame->shell->camToWorld.inverse(),firstFrame->shell->aff_g2l);
//        firstFrame->shell->trackingRef=0;
//        firstFrame->shell->camToTrackingRef = SE3();
//
//        newFrame->shell->camToWorld = firstToNew.inverse();
//        newFrame->shell->aff_g2l = AffLight(0,0);
//        newFrame->setEvalPT_scaled(newFrame->shell->camToWorld.inverse(),newFrame->shell->aff_g2l);
//        newFrame->shell->trackingRef = firstFrame->shell;
//        newFrame->shell->camToTrackingRef = firstToNew.inverse();
//
//    }

    printf("INITIALIZE FROM INITIALIZER (%d pts)!\n", (int)firstFrame->pointHessians.size());
    sp_initializer_->DislayChosenPoints(0, indexes);
}

void System::SetPreCalculateValues()
{
    const auto fh_count = vec_hessian_frames.size();
    for(FrameHessian* fh : vec_hessian_frames)
    {
        fh->targetPrecalc.resize(fh_count);
        for (auto i = 0;i < fh_count; i++)
            fh->targetPrecalc[i].set(fh, vec_hessian_frames[i], &hessian_calib_);
    }

//    ef->setDeltaF(&hessian_calib_);
}

}