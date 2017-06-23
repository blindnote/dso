//
// Created by Yin Rochelle on 13/06/2017.
//

#include <list>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include "Initializer.h"
#include "iFeatureExtractor.h"
#include "iRichFeatureMatcher.h"

using namespace std;


const int   MIN_POINT_COUNT_FOR_HOMOGRAPHY         = 100;

Initializer::Initializer(const std::vector<cv::Mat>& images,
                         const std::vector<std::string>& image_names,
                         const std::shared_ptr<iEpipolarSolver> solver_ptr,
                         std::vector<CloudPoint>& cloud)
        : miFirstViewIndex(-1),
          miSecondViewIndex(-1),
          mCloud(cloud),
          mpEpipolarSolver(solver_ptr)
{
    convert_imgs_to_8U3(images, mImagesBgrVec, mImagesGrayVec);
//    for (unsigned int i = 0; i < images.size(); i++)
//    {
//        mImagesBgrVec.push_back(images[i]);
//        mImagesGrayVec.push_back(images[i]);
//    }
    mImageNamesVec.assign(image_names.begin(), image_names.end());
}

bool Initializer::RecoverDepth()
{
    this->MatchFeatures();

    std::cout << "\n======================================================================\n";
    std::cout << "======================== Depth Recovery Start ========================\n";
    std::cout << "======================================================================\n\n";

    this->ConstructBaseLine();
    //this->ConstructBaseLineSpecify();
    this->EnrichScene();

    std::cout << "\n======================================================================\n";
    std::cout << "========================= Depth Recovery DONE ========================\n";
    std::cout << "======================================================================\n\n";

    SaveCloudAndCamerasToPLY("/Users/yinr/ComputerVision/dataset/calib_narrowGamma_scene2/sfm");
    //SaveCamerasInvToPLY("/Users/yinr/ComputerVision/dataset/calib_narrowGamma_scene2/sfm_inv");

    std::cout << "******************* Finally got:" << mCloud.size() << std::endl << std::endl;

    return ( miFirstViewIndex != -1 );
}

Sophus::SE3d Initializer::GetFirstFramePose()
{
    if (miFirstViewIndex == -1)
       return Sophus::SE3d();

    return GetNthFramePose(miFirstViewIndex);
}

Sophus::SE3d Initializer::GetSecondFramePose()
{
    if (miSecondViewIndex == -1)
        return Sophus::SE3d();

    return GetNthFramePose(miSecondViewIndex);
}


void Initializer::BuildInitialSceneForDso(std::vector<InvDepthPnt>& idpts, Sophus::SE3d& thisToNext)
{
    if(miFirstViewIndex == -1 ||
       miSecondViewIndex == -1)
        return;

    idpts.clear();

    cv::Mat KP = mpEpipolarSolver->GetCameraMat() * (cv::Mat(Pmats[miFirstViewIndex]));
    //cv::Mat KP2 = mpEpipolarSolver->GetCameraMat() * (cv::Mat(Pmats[miSecondViewIndex]));

    for(const CloudPoint& p: mCloud)
    {
        int firstIdx = p.index_of_2d_origin[miFirstViewIndex];
        if (firstIdx == -1) continue;

        cv::KeyPoint& firstViewKp = mKeyPointsVec[miFirstViewIndex][firstIdx];
        InvDepthPnt pntInFirstView;
        pntInFirstView.u = firstViewKp.pt.x;
        pntInFirstView.v = firstViewKp.pt.y;

        cv::Mat homo_p3d = cv::Mat(4, 1, CV_64F);
        homo_p3d.at<double>(0,0) = p.p3d.x;
        homo_p3d.at<double>(1,0) = p.p3d.y;
        homo_p3d.at<double>(2,0) = p.p3d.z;
        homo_p3d.at<double>(3,0) = 1.0;

        cv::Mat p2d_in_img_mat = KP * homo_p3d;
        pntInFirstView.depth = p2d_in_img_mat.at<double>(2);
        pntInFirstView.idepth = (pntInFirstView.depth > 0.0) ? 1.0/pntInFirstView.depth : -1.0;
        pntInFirstView.isGood = (pntInFirstView.depth > 0.0) ? true : false;

        idpts.push_back(pntInFirstView);

        /*
        int secondIdx = p.index_of_2d_origin[miSecondViewIndex];
        if (secondIdx == -1) continue;

        cv::KeyPoint& secondViewKp = mKeyPointsVec[miSecondViewIndex][secondIdx];
        InvDepthPnt pntInSecondView;
        pntInSecondView.u = secondViewKp.pt.x;
        pntInSecondView.v = secondViewKp.pt.y;

        cv::Mat p2d_in_img2_mat = KP2 * homo_p3d;
        pntInSecondView.depth = p2d_in_img2_mat.at<double>(2);
        pntInSecondView.idepth = (pntInSecondView.depth > 0.0) ? 1.0/pntInSecondView.depth : -1.0;
        pntInSecondView.isGood = (pntInSecondView.depth > 0.0) ? true : false;

        idpts.push_back(pntInSecondView);
        */
    }

    //thisToNext = GetSecondFramePose().inverse();
    thisToNext = GetSecondFramePose();
}


Sophus::SE3d Initializer::GetNthFramePose(int id)
{
    if (id < 0 || id >= mImagesBgrVec.size())
        return Sophus::SE3d();

    cv::Matx34d pose = Pmats[id];
    return GetSE3From3x4Mat(pose);
}

Sophus::SE3d Initializer::GetSE3From3x4Mat(const cv::Matx34d& pose)
{
    Eigen::Matrix<double,4,4> midFormat;
    midFormat << pose(0, 0), pose(0, 1), pose(0, 2), pose(0, 3),
            pose(1, 0), pose(1, 1), pose(1, 2), pose(1, 3),
            pose(2, 0), pose(2, 1), pose(2, 2), pose(2, 3),
            0.0, 0.0, 0.0, 1.0;
    return Sophus::SE3d(midFormat);
}

cv::Matx34d Initializer::GetInvFromSophusPose(const Sophus::SE3d& pose_se3)
{
    Sophus::SE3d inv_pose = pose_se3.inverse();

    auto mat = inv_pose.matrix3x4();

    return cv::Matx34d(
            mat(0, 0), mat(0, 1), mat(0, 2), mat(0, 3),
            mat(1, 0), mat(1, 1), mat(1, 2), mat(1, 3),
            mat(2, 0), mat(2, 1), mat(2, 2), mat(2, 3)
    );
}

std::vector<iFrame> Initializer::GetBaseLineFrames()
{
    std::vector<iFrame> framesVec;

    for (int f = 0; f < mImageNamesVec.size(); f++)
    {
        iFrame frame(mImageNamesVec[f],
                    mImagesBgrVec[f], mImagesGrayVec[f],
                    mKeyPointsVec[f], Pmats[f]);
        framesVec.push_back(frame);
    }

    return framesVec;
}

void Initializer::MatchFeatures()
{
    std::vector<cv::Mat> descriptorsVec;

    std::shared_ptr<iFeatureExtractor> pExtractor = std::make_shared<iFeatureExtractor>(eDetectorType::ORB);
    pExtractor->ExtractFeaturesBatch(mImagesGrayVec, descriptorsVec, mKeyPointsVec);

    //std::shared_ptr<iRichFeatureMatcher> pMatcher = std::make_shared<iRichFeatureMatcher>(eMatcherType::FlannBased);
    std::shared_ptr<iRichFeatureMatcher> pMatcher = std::make_shared<iRichFeatureMatcher>(eMatcherType::BruteForce);

    for (int i = 0; i < (int)mImagesGrayVec.size() - 1; i++)
    {
        for (int j = i + 1; j < mImagesGrayVec.size(); j++)
        {
            std::cout << "*** Matching " << mImageNamesVec[i]
                      << " & " << mImageNamesVec[j];

            std::vector<cv::DMatch> matches, matchesFlipped;
            pMatcher->Match(descriptorsVec[i], descriptorsVec[j],
                            mKeyPointsVec[i], mKeyPointsVec[j],
                            matches);
            pMatcher->FlipMatches(matches, matchesFlipped);

            std::cout << ": " << matches.size() << std::endl;

            mMatchesTable[std::make_pair(i, j)] = matches;
            mMatchesTable[std::make_pair(j, i)] = matchesFlipped;

//            pMatcher->DrawMatches(mImageNamesVec[i] + " -> " + mImageNamesVec[j],
//                                  mImagesGrayVec[i], mImagesGrayVec[j],
//                                  mKeyPointsVec[i], mKeyPointsVec[j],
//                                  matches, 0.5);
        }
    }
}


int Initializer::findHomographyInliers(const std::vector<cv::KeyPoint>& left,
                                       const std::vector<cv::KeyPoint>& right,
                                       const std::vector<cv::DMatch>& matches)
{
    std::vector<cv::Point2d> alignedLeft;
    std::vector<cv::Point2d> alignedRight;
    get_aligned_point2ds(left, right, matches, alignedLeft, alignedRight);

    cv::Mat inlierMask;
    cv::Mat homography;
    if(matches.size() >= 4)
    {
        homography = findHomography(alignedLeft, alignedRight,
                                    cv::RANSAC, 10.0, inlierMask);
    }

    if(matches.size() < 4 || homography.empty())
    {
        return 0;
    }

    return countNonZero(inlierMask);
}


std::map<float, ImagePair> Initializer::sortViewsForBaseline()
{
    std::cout << "---------- Find Views Homography Inliers -----------" << std::endl;

    //sort pairwise matches to find the lxowest Homography inliers [Snavely07 4.2]
    std::map<float, ImagePair> matchesSizes;
    size_t numImages = mImagesGrayVec.size();

    for (size_t i = 0; (int)i < (int)numImages - 1; i++)
    {
        for (size_t j = i + 1; (int)j < (int)numImages; j++)
        {
            const std::vector<cv::DMatch>& matches = mMatchesTable[std::make_pair(i, j)];
            if (matches.size() < MIN_POINT_COUNT_FOR_HOMOGRAPHY)
            {
                //Not enough points in matching
                matchesSizes[1.0] = {i, j};
                continue;
            }

            //Find number of homography inliers
            const int numInliers = findHomographyInliers(mKeyPointsVec[i],
                                                         mKeyPointsVec[j],
                                                         matches);
            const float inliersRatio = (float)numInliers / (float)(matches.size());
            matchesSizes[inliersRatio] = {i, j};

            std::cout << "Homography inliers ratio: (" << i << ", " << j << ") -> " << inliersRatio << std::endl;
        }
    }

    return matchesSizes;
}


void Initializer::MatchFeatures2()
{
    std::vector<cv::Mat> descriptorsVec;

    std::shared_ptr<iFeatureExtractor> pExtractor = std::make_shared<iFeatureExtractor>(eDetectorType::ORB);
    pExtractor->ExtractFeaturesBatch(mImagesGrayVec, descriptorsVec, mKeyPointsVec);

    std::shared_ptr<iRichFeatureMatcher> pMatcher = std::make_shared<iRichFeatureMatcher>(eMatcherType::FlannBased);
    //std::shared_ptr<RichFeatureMatcher> pMatcher = std::make_shared<RichFeatureMatcher>(eMatcherType::BruteForce);

    for (int i = 0; i < (int)mImagesGrayVec.size() - 1; i++)
    {
        for (int j = i + 1; j < mImagesGrayVec.size(); j++)
        {
            std::cout << "*** Matching " << mImageNamesVec[i]
                      << ", " << mImageNamesVec[j];

            std::vector<cv::DMatch> matches, matchesFlipped;
            pMatcher->Match(descriptorsVec[i], descriptorsVec[j],
                            mKeyPointsVec[i], mKeyPointsVec[j],
                            matches);
            pMatcher->FlipMatches(matches, matchesFlipped);

            mMatchesTable[std::make_pair(i, j)] = matches;
            mMatchesTable[std::make_pair(j, i)] = matchesFlipped;

            std::cout << " : " << matches.size() << " ***" << std::endl;
        }
    }

    std::cout << std::endl;
}

void Initializer::ConstructBaseLine()
{
    cv::Mat camera = mpEpipolarSolver->GetCameraMat();

    std::cout << "----------- Find Baseline Triangulation ------------" << std::endl;

    std::map<float, ImagePair> pairsHomographyInliers = sortViewsForBaseline();

    cv::Matx34d Pleft  = cv::Matx34d::eye();
    cv::Matx34d Pright = cv::Matx34d::eye();
    std::vector<CloudPoint> new_cloud;
    for (auto& imagePair : pairsHomographyInliers)
    {
        if (fabs(imagePair.first - 1.0) < 1e-10) continue;

        std::cout << "Trying " << imagePair.second
                  << " " << mImageNamesVec[imagePair.second.left]
                  << " v.s. " << mImageNamesVec[imagePair.second.right]
                  << " ratio: " << imagePair.first << std::endl << std::flush;

        size_t i = imagePair.second.left;
        size_t j = imagePair.second.right;

        std::vector<cv::DMatch> old_matches = mMatchesTable[std::make_pair(i, j)];
        std::vector<cv::Point2d> left_aligned_point2ds_vec;
        std::vector<cv::Point2d> right_aligned_point2ds_vec;
        get_aligned_point2ds(mKeyPointsVec[i],
                             mKeyPointsVec[j],
                             old_matches,
                             left_aligned_point2ds_vec,
                             right_aligned_point2ds_vec);

//        iRichFeatureMatcher::DrawMatches("old",
//                                         mImagesGrayVec[i], mImagesGrayVec[j],
//                                         mKeyPointsVec[i], mKeyPointsVec[j],
//                                         old_matches);

        std::vector<cv::DMatch> prunedMatches;
        bool success = FindExtrinsicsFromMatches(camera,
                                                 left_aligned_point2ds_vec,
                                                 right_aligned_point2ds_vec,
                                                 old_matches, prunedMatches,
                                                 Pleft, Pright);

//        iRichFeatureMatcher::DrawMatches("essentail",
//                                         mImagesGrayVec[i], mImagesGrayVec[j],
//                                         mKeyPointsVec[i], mKeyPointsVec[j],
//                                         prunedMatches);

        if (not success)
        {
            std::cout << "stereo view could not be obtained " << imagePair.second
                      << ", go to next pair" << std::endl << std::flush;
            continue;
        }

        float poseInliersRatio = (float)prunedMatches.size() / (float)old_matches.size();
        std::cout << "pose inliers ratio " << poseInliersRatio << std::endl;

        if (poseInliersRatio < POSE_INLIERS_MINIMAL_RATIO)
        {
            std::cout << "insufficient pose inliers. skip." << std::endl;
            continue;
        }


//        iRichFeatureMatcher::DrawMatches("hello",
//                                        mImagesGrayVec[i], mImagesGrayVec[j],
//                                        mKeyPointsVec[i], mKeyPointsVec[j],
//                                        prunedMatches);


        mMatchesTable[std::make_pair(i, j)] = prunedMatches;

        ////??????????????????/////////
        //cv::Matx34d pleftInv = GetInvFromSophusPose(GetSE3From3x4Mat(Pleft));
        //cv::Matx34d prightInv = GetInvFromSophusPose(GetSE3From3x4Mat(Pright));
        ////??????????????????/////////

        new_cloud.clear();
        TriangulateViews(i, j,
                         prunedMatches,
                         Pleft, Pright,
                         //pleftInv, prightInv,
                         new_cloud);

        std::cout << "---- Triangulate from stereo views: " << imagePair.second << ", " << new_cloud.size() << " points"<< std::endl;

        miFirstViewIndex = i;
        miSecondViewIndex = j;

        Pmats[miFirstViewIndex] = Pleft;
        Pmats[miSecondViewIndex] = Pright;

        std::cout << "\n***** Taking baseline from " << mImageNamesVec[miFirstViewIndex]
                  << " and " << mImageNamesVec[miSecondViewIndex] << " *****" << std::endl;

        std::cout << "\n" << mImageNamesVec[miFirstViewIndex] << " pose:\n" << Pleft << std::endl;
        std::cout << "\n" << mImageNamesVec[miSecondViewIndex] << " pose:\n" << Pright << std::endl;

        mDoneViewsIdxVec.push_back(miFirstViewIndex);
        mDoneViewsIdxVec.push_back(miSecondViewIndex);

        break;
    }


    cv::Mat KP2 = mpEpipolarSolver->GetCameraMat() * (cv::Mat(Pmats[miSecondViewIndex]));
    std::vector<cv::DMatch> matches_chosen = mMatchesTable[std::make_pair(miFirstViewIndex, miSecondViewIndex)];
    for (auto i = 0; i < new_cloud.size(); i++)
    {
        CloudPoint clp;
        clp.p3d = new_cloud[i].p3d;
        clp.index_of_2d_origin = std::vector<int>(mImagesBgrVec.size(), -1);
        clp.index_of_2d_origin[miFirstViewIndex] = matches_chosen[i].queryIdx;
        clp.index_of_2d_origin[miSecondViewIndex] = matches_chosen[i].trainIdx;

        //cv::Mat homo_p3d = cv::Mat(4, 1, CV_64F, {clp.p3d.x, clp.p3d.y, clp.p3d.z, 1.0});
        cv::Mat homo_p3d = cv::Mat(4, 1, CV_64F);
        homo_p3d.at<double>(0,0) = clp.p3d.x;
        homo_p3d.at<double>(1,0) = clp.p3d.y;
        homo_p3d.at<double>(2,0) = clp.p3d.z;
        homo_p3d.at<double>(3,0) = 1.0;
        cv::Mat p2d_in_img2_mat = KP2 * homo_p3d;
        cv::Point2f p2d_in_img2(p2d_in_img2_mat.at<double>(0) / p2d_in_img2_mat.at<double>(2),
                                p2d_in_img2_mat.at<double>(1) / p2d_in_img2_mat.at<double>(2));
        clp.reprojection_error = cv::norm(p2d_in_img2 - mKeyPointsVec[miSecondViewIndex][matches_chosen[i].trainIdx].pt);

        mCloud.push_back(clp);
    }

    //adjustCurrentBundle();

    std::cout << "\n***** Initial point cloud recovered: " << mCloud.size() << " *****" << std::endl << std::endl;
}

void Initializer::ConstructBaseLineSpecify()
{
    cv::Mat camera = mpEpipolarSolver->GetCameraMat();

    std::cout << "----------- Find Baseline Triangulation ------------" << std::endl;

    //std::map<float, ImagePair> pairsHomographyInliers = sortViewsForBaseline();

    cv::Matx34d Pleft = cv::Matx34d::eye();
    cv::Matx34d Pright = cv::Matx34d::eye();
    std::vector<CloudPoint> new_cloud;
    /*
    for (auto &imagePair : pairsHomographyInliers)
    {
        std::cout << "Trying " << imagePair.second
                  << " " << mImageNamesVec[imagePair.second.left]
                  << " v.s. " << mImageNamesVec[imagePair.second.right]
                  << " ratio: " << imagePair.first << std::endl << std::flush;

        size_t i = imagePair.second.left;
        size_t j = imagePair.second.right;
    */
        size_t i = 7;
        size_t j = 8;

        std::vector<cv::DMatch> old_matches = mMatchesTable[std::make_pair(i, j)];
        std::vector<cv::Point2d> left_aligned_point2ds_vec;
        std::vector<cv::Point2d> right_aligned_point2ds_vec;
        get_aligned_point2ds(mKeyPointsVec[i],
                             mKeyPointsVec[j],
                             old_matches,
                             left_aligned_point2ds_vec,
                             right_aligned_point2ds_vec);

        std::vector<cv::DMatch> prunedMatches;
        bool success = FindExtrinsicsFromMatches(camera,
                                                 left_aligned_point2ds_vec,
                                                 right_aligned_point2ds_vec,
                                                 old_matches, prunedMatches,
                                                 Pleft, Pright);

        if (not success) {
            std::cout << "stereo view could not be obtained " << i << ", " << j
                      << ", go to next pair" << std::endl << std::flush;
            return;
        }

        float poseInliersRatio = (float) prunedMatches.size() / (float) old_matches.size();
        std::cout << "pose inliers ratio " << poseInliersRatio << std::endl;

//        if (poseInliersRatio < POSE_INLIERS_MINIMAL_RATIO) {
//            std::cout << "insufficient pose inliers. skip." << std::endl;
//            return;
//        }

        mMatchesTable[std::make_pair(i, j)] = prunedMatches;

        new_cloud.clear();
        TriangulateViews(i, j,
                         prunedMatches,
                         Pleft, Pright,
                         new_cloud);

        std::cout << "---- Triangulate from stereo views: " << i << ", " << j << ", " << new_cloud.size() << " points"
                  << std::endl;

        miFirstViewIndex = i;
        miSecondViewIndex = j;

        Pmats[miFirstViewIndex] = Pleft;
        Pmats[miSecondViewIndex] = Pright;

        std::cout << "\n***** Taking baseline from " << mImageNamesVec[miFirstViewIndex]
                  << " and " << mImageNamesVec[miSecondViewIndex] << " *****" << std::endl;

        std::cout << "\n" << mImageNamesVec[miFirstViewIndex] << " pose:\n" << Pleft << std::endl;
        std::cout << "\n" << mImageNamesVec[miSecondViewIndex] << " pose:\n" << Pright << std::endl;

        mDoneViewsIdxVec.push_back(miFirstViewIndex);
        mDoneViewsIdxVec.push_back(miSecondViewIndex);

 //       break;
 //   }


    cv::Mat KP2 = mpEpipolarSolver->GetCameraMat() * (cv::Mat(Pmats[miSecondViewIndex]));
    std::vector<cv::DMatch> matches_chosen = mMatchesTable[std::make_pair(miFirstViewIndex, miSecondViewIndex)];
    for (auto i = 0; i < new_cloud.size(); i++) {
        CloudPoint clp;
        clp.p3d = new_cloud[i].p3d;
        clp.index_of_2d_origin = std::vector<int>(mImagesBgrVec.size(), -1);
        clp.index_of_2d_origin[miFirstViewIndex] = matches_chosen[i].queryIdx;
        clp.index_of_2d_origin[miSecondViewIndex] = matches_chosen[i].trainIdx;

        //cv::Mat homo_p3d = cv::Mat(4, 1, CV_64F, {clp.p3d.x, clp.p3d.y, clp.p3d.z, 1.0});
        cv::Mat homo_p3d = cv::Mat(4, 1, CV_64F);
        homo_p3d.at<double>(0, 0) = clp.p3d.x;
        homo_p3d.at<double>(1, 0) = clp.p3d.y;
        homo_p3d.at<double>(2, 0) = clp.p3d.z;
        homo_p3d.at<double>(3, 0) = 1.0;
        cv::Mat p2d_in_img2_mat = KP2 * homo_p3d;
        cv::Point2f p2d_in_img2(p2d_in_img2_mat.at<double>(0) / p2d_in_img2_mat.at<double>(2),
                                p2d_in_img2_mat.at<double>(1) / p2d_in_img2_mat.at<double>(2));
        clp.reprojection_error = cv::norm(
                p2d_in_img2 - mKeyPointsVec[miSecondViewIndex][matches_chosen[i].trainIdx].pt);

        mCloud.push_back(clp);
    }

    //adjustCurrentBundle();

    std::cout << "***** Initial point cloud recovered: " << mCloud.size() << " *****" << std::endl << std::endl;
}

void Initializer::EnrichScene()
{
    if (mCloud.size() < 1) return;

    cv::Mat camera = mpEpipolarSolver->GetCameraMat();
    cv::Mat distort_coeffs = mpEpipolarSolver->GetDistortCoeffs();

    cv::Mat R, t;
    auto views_count = mImagesBgrVec.size();
    while (mDoneViewsIdxVec.size() != views_count)
    {
        int max_3d2d_view_idx = -1;
        int max_3d2d_corresp_cnt = 0;
        std::vector<cv::Point3d> max_3ds_vec;
        std::vector<cv::Point2d> max_2ds_vec;
        //find image with highest 2d-3d correspondance [Snavely07 4.2]
        for (int view_idx = 0; view_idx < views_count; view_idx++)
        {
            //already done with this view
            if(std::find(mDoneViewsIdxVec.begin(), mDoneViewsIdxVec.end(), view_idx) != mDoneViewsIdxVec.end())
                continue;

            std::cout << mImageNamesVec[view_idx] << ": ";

            std::vector<cv::Point3d> tmp_3ds_vec;
            std::vector<cv::Point2d> tmp_2ds_vec;
            Find3d2dCorrespondences(view_idx, tmp_3ds_vec, tmp_2ds_vec);

            if(tmp_3ds_vec.size() > max_3d2d_corresp_cnt)
            {
                max_3d2d_corresp_cnt = tmp_3ds_vec.size();
                max_3d2d_view_idx = view_idx;
                max_3ds_vec = tmp_3ds_vec;
                max_2ds_vec = tmp_2ds_vec;
            }
        }

        //most 2d3d correspondences view
        int best_view_idx = max_3d2d_view_idx;
        if (best_view_idx == -1) continue;

        // std::cout << "Best view " << working_view_idx << " has " << max_3d2d_corresp_cnt << " correspondences" << std::endl;
        std::cout << "Adding " << best_view_idx << " to existing " << cv::Mat(std::vector<int>(mDoneViewsIdxVec.begin(), mDoneViewsIdxVec.end())).t() << std::endl;
        std::cout << "-------------------------- " << mImageNamesVec[best_view_idx] << " --------------------------\n";
        mDoneViewsIdxVec.push_back(best_view_idx); // don't repeat it for now

        // solvePnP
        bool pose_estimated = FindPoseEstimation(camera, distort_coeffs,
                                                 R, t,
                                                 max_3ds_vec, max_2ds_vec);
        if(!pose_estimated)
        {
            std::cout << "Cannot recover camera pose for view: " << best_view_idx << std::endl;
            continue;
        }

        //store estimated pose
        cv::Matx34d& pose = Pmats[best_view_idx];
        R.copyTo(cv::Mat(3, 4, CV_64FC1, pose.val)(ROT));
        t.copyTo(cv::Mat(3, 4, CV_64FC1, pose.val)(TRA));

        std::cout << "New view " << best_view_idx << " pose " << std::endl << Pmats[best_view_idx] << std::endl;

        std::vector<CloudPoint> new_filtered_cloud;
        for (const int good_view_idx : mDoneViewsIdxVec)
        {
            if (good_view_idx == best_view_idx) continue; // skip current...

            std::cout << " -> " << mImageNamesVec[good_view_idx] << std::endl;

            size_t leftViewIdx  = (good_view_idx < best_view_idx) ? good_view_idx : best_view_idx;
            size_t rightViewIdx = (good_view_idx < best_view_idx) ? best_view_idx : good_view_idx;

            std::vector<cv::DMatch> old_matches = mMatchesTable[std::make_pair(leftViewIdx, rightViewIdx)];
            std::vector<cv::Point2d> left_aligned_point2ds_vec;
            std::vector<cv::Point2d> right_aligned_point2ds_vec;
            get_aligned_point2ds(mKeyPointsVec[leftViewIdx],
                                 mKeyPointsVec[rightViewIdx],
                                 old_matches,
                                 left_aligned_point2ds_vec,
                                 right_aligned_point2ds_vec);

            std::vector<cv::DMatch> prunedMatches;
            cv::Matx34d Pleft = cv::Matx34d::eye();
            cv::Matx34d Pright = cv::Matx34d::eye();

            //std::cout << "!!!!!!! before pruning: " << old_matches.size() << std::endl;
            //use the essential matrix recovery to prune the matches
            bool success = FindExtrinsicsFromMatches(camera,
                                                     left_aligned_point2ds_vec,
                                                     right_aligned_point2ds_vec,
                                                     old_matches, prunedMatches,
                                                     Pleft, Pright);

            std::vector<cv::DMatch> prunedMatchesFlipped;
            iRichFeatureMatcher::FlipMatches(prunedMatches, prunedMatchesFlipped);
            mMatchesTable[std::make_pair(leftViewIdx, rightViewIdx)] = prunedMatches;
            mMatchesTable[std::make_pair(rightViewIdx, leftViewIdx)] = prunedMatchesFlipped;
            //std::cout << "!!!!!!! After pruning: " << prunedMatches.size() << std::endl;

            /*
            RichFeatureMatcher::DrawMatches(leftViewIdx + " - " + rightViewIdx,
                                            mImagesBgrVec[leftViewIdx],
                                            mImagesBgrVec[rightViewIdx],
                                            mKeyPointsVec[leftViewIdx],
                                            mKeyPointsVec[rightViewIdx],
                                            prunedMatches);
             */


            std::vector<CloudPoint> new_cloud;
            /*
            std::vector<int> add_to_cloud;

            bool good_triangulation = TriangulatePointsBetweenViews(leftViewIdx,
                                                                    rightViewIdx,
                                                                    new_cloud,
                                                                    add_to_cloud);
            */
            std::cout << "best_view_idx: " << best_view_idx
                      << ", leftViewIdx: " << leftViewIdx
                      << ", rightViewIdx: " << rightViewIdx << std::endl;

            ////??????????????????/////////
            //cv::Matx34d pleftInv = GetInvFromSophusPose(GetSE3From3x4Mat(Pmats[leftViewIdx]));
            //cv::Matx34d prightInv = GetInvFromSophusPose(GetSE3From3x4Mat(Pmats[rightViewIdx]));
            ////??????????????????/////////


            TriangulateViews(leftViewIdx, rightViewIdx,
                             prunedMatches,
                             Pmats[leftViewIdx],
                             Pmats[rightViewIdx],
                             //pleftInv,
                             //prightInv,
                             new_cloud);

            std::cout << "---- Triangulate from stereo views: "
                      << leftViewIdx << "," << rightViewIdx
                      << ": " << new_cloud.size() << " points"<< std::endl;

            std::cout << "Merge triangulation between " << leftViewIdx << " and " << rightViewIdx
                      << " (# matching pts = " << (mMatchesTable[std::make_pair(leftViewIdx,rightViewIdx)].size()) << ") ";

            MergeNewPointCloud(new_cloud, new_filtered_cloud);

            /*
            std::cout << "before triangulation: " << mCloud.size();
            for (int j = 0; j < add_to_cloud.size(); j++)
            {
              if(add_to_cloud[j] == 1)
              {
                mCloud.push_back(new_cloud[j]);
                new_filtered_cloud.push_back(new_cloud[j]);
              }
            }
            std::cout << " after " << mCloud.size() << std::endl;
            */
            /*
            mCloud.reserve(mCloud.size() + new_cloud.size());
            mCloud.insert(mCloud.end(), new_cloud.begin(), new_cloud.end());

            new_filtered_cloud.reserve(new_filtered_cloud.size() + new_cloud.size());
            new_filtered_cloud.insert(new_filtered_cloud.end(), new_cloud.begin(), new_cloud.end());
            */
        }
    }
}


void Initializer::Find3d2dCorrespondences(int curr_view_index,
                                          std::vector<cv::Point3d>& point3ds_vec,
                                          std::vector<cv::Point2d>& point2ds_vec)
{
    point3ds_vec.clear();
    point2ds_vec.clear();

    const int total_view_cnt = mImagesBgrVec.size();
    for (int i = 0; i < mCloud.size(); i++)
    {
        CloudPoint clp = mCloud[i];

        bool found2DPoint = false;
        for (int viewIdx = 0; viewIdx < total_view_cnt; viewIdx++)
        {
            if (clp.index_of_2d_origin[viewIdx] == -1)
                continue;

            int originKpIdx = clp.index_of_2d_origin[viewIdx];

            int leftViewIdx  = (viewIdx < curr_view_index) ? viewIdx : curr_view_index;
            int rightViewIdx = (viewIdx < curr_view_index) ? curr_view_index : viewIdx;

            std::vector<cv::DMatch> matches = mMatchesTable[std::make_pair(leftViewIdx, rightViewIdx)];
            for (const cv::DMatch& m : matches)
            {
                int matched2DPointKpIdx = -1;
                if (viewIdx < curr_view_index) // originating view is 'left'
                {
                    if (m.queryIdx == originKpIdx)
                    {
                        matched2DPointKpIdx = m.trainIdx;
                    }
                }
                else // originating view is 'right'
                {
                    if (m.trainIdx == originKpIdx)
                    {
                        matched2DPointKpIdx = m.queryIdx;
                    }
                }

                if (matched2DPointKpIdx >= 0)
                {
                    point3ds_vec.push_back(clp.p3d);
                    point2ds_vec.push_back(mKeyPointsVec[curr_view_index][matched2DPointKpIdx].pt);
                    found2DPoint = true;
                    break;
                }
            } // end for(m: matches)

            if (found2DPoint)
            {
                break;
            }
        } // end for(kpIdx:  index_of_2d_origin)
    }

    std::cout << "found " << point3ds_vec.size() << " 3d-2d correspondences" << std::endl;
}

bool Initializer::FindPoseEstimation(const cv::Mat& camera,
                                     const cv::Mat& distortion,
                                     cv::Mat& R, cv::Mat& tvec,
                                     std::vector<cv::Point3d> point3ds_vec,
                                     std::vector<cv::Point2d> point2ds_vec)
{
    /*
    if(point3ds_vec.size() < 4 ||
       point2ds_vec.size() < 4 ||
       point3ds_vec.size() != point2ds_vec.size())
    {
      //something went wrong aligning 3D to 2D points..
      std::cerr << "couldn't find [enough] correspondence points... (only " << point3ds_vec.size() << ")" <<endl;
      return false;
    }
    */

//  cv::Mat camera = mpEpipolarSolver->GetCameraMat();
//  cv::Mat distort_coeffs = mpEpipolarSolver->GetDistortCoeffs();

    // double minVal,maxVal;
    // cv::minMaxIdx(point2ds_vec, &minVal, &maxVal);

    if (!point3ds_vec.size() || !point2ds_vec.size())
    {
        std::cout << "Empty 3d-2d correspondence!" << std::endl;
        return false;
    }


    cv::Mat rvec;
    std::vector<int> inliers;
    bool res = cv::solvePnPRansac(point3ds_vec, point2ds_vec,
                                  camera, distortion,
                                  rvec, tvec,
                                  false,
                                  100,   // 1000?
                                  RANSAC_THRESHOLD,   // 0.006 * maxVal, default: 8.0
                                  0.99,  // 0.25 * (double)(point2ds_vec.size())
                                  inliers,
                                  cv::SOLVEPNP_EPNP);    // SfmToyLib: cv::Iterative

    std::cout << "EPNP result: " << res << std::endl;
    std::cout << "inliers: " << cv::countNonZero(inliers) << std::endl;
    std::cout << "rvec=" << rvec << std::endl;
    //std::cout << "tvec=" << tvec << std::endl;

    if (!res)
    {
        std::cout << "solvePnP return false!" << std::endl;
        return false;
    }

    /*
    if(inliers.size() < (double)(point2ds_vec.size()) / 5.0)
    {
      std::cout << "not enough inliers to consider a good pose (" << inliers.size() << "/" <<point2ds_vec.size()<<")" << endl;
      return false;
    }
     */
    double inliers_ratio = (float)cv::countNonZero(inliers) / (float)point2ds_vec.size();
    if (inliers_ratio < POSE_INLIERS_MINIMAL_RATIO)
    {
        std::cout << "!!!Inliers ratio is too small: " << cv::countNonZero(inliers) << " / "
                  << point2ds_vec.size() << ": " << inliers_ratio << std::endl;
        return false;
    }

    cv::Rodrigues(rvec, R);
    /*
    if(!mpEpipolarSolver->CheckCoherentRotation(R))
    {
      cerr << "rotation is incoherent. we should try a different base view..." << endl;
      return false;
    }
    */
    std::cout << "tvec = " << tvec << "\nR = \n"<< R <<std::endl;


    return true;
}

bool Initializer::FindExtrinsicsFromMatches(const cv::Mat& K,
                                            const std::vector<cv::Point2d>& left_point2ds,
                                            const std::vector<cv::Point2d>& right_point2ds,
                                            const std::vector<cv::DMatch>& matches,
                                            std::vector<cv::DMatch>& prunedMatches,
                                            cv::Matx34d& Pleft, cv::Matx34d& Pright)
{
    prunedMatches.clear();

    //double focal = K.at<double>(0, 0); // Note: assuming fx = fy
    double focal = (K.at<double>(0, 0) + K.at<double>(1, 1))/2;
    cv::Point2d pp(K.at<double>(0, 2), K.at<double>(1, 2));

    cv::Mat E, R, t;
    cv::Mat mask;
   // E = findEssentialMat(left_point2ds, right_point2ds, focal, pp, cv::RANSAC, 0.999, 1.0, mask);
    E = findEssentialMat(left_point2ds, right_point2ds, K, cv::RANSAC, 0.999, 1.0, mask);
    std::cout << "...... findEssentialMat: " << cv::countNonZero(mask) << std::endl;

    // Find Pright camera matrix from the essential matrix
    // Cheirality check (all points are in front of camera: i.e. the triangulated 3D points should have positive depth) is performed internally.
    //int recover_inliers_cnt = recoverPose(E, left_point2ds, right_point2ds, R, t, focal, pp, mask);
    // cv::Mat mask2;
    int recover_inliers_cnt = recoverPose(E, left_point2ds, right_point2ds, K, R, t, mask);
    std::cout << "...... recoverPose: " << cv::countNonZero(mask) << std::endl;

    std::cout << "recoverPose inliers:" << recover_inliers_cnt << std::endl;
    if (recover_inliers_cnt < 1)
    {
        return false;
    }

    //TODO: stratify over Pleft
    Pleft = cv::Matx34d::eye();
    Pright = cv::Matx34d(R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0),
                         R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1),
                         R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2));

    //populate pruned matches
    for (size_t i = 0; i < mask.rows; i++)
    {
        if (!mask.at<uchar>(i)) continue;

        prunedMatches.push_back(matches[i]);
    }

    return true;
}


void Initializer::TriangulateViews(int left_view_index,
                                   int right_view_index,
                                   const std::vector<cv::DMatch>& matches,
                                   const cv::Matx34d& Pleft,
                                   const cv::Matx34d& Pright,
                                   std::vector<CloudPoint>& new_cloud)
{
    std::vector<cv::Point2d> left_point_2ds, right_point_2ds;
    get_aligned_point2ds(mKeyPointsVec[left_view_index],
                         mKeyPointsVec[right_view_index],
                         matches,
                         left_point_2ds,
                         right_point_2ds);

    if (left_point_2ds.size() < 1 || right_point_2ds.size() < 1) return;

    cv::Mat K = mpEpipolarSolver->GetCameraMat();
    cv::Mat normalizedLeftPts, normalizedRightPts;
    //std::cout << "............... left_point_2ds.size():" << left_point_2ds.size() << std::endl;
    //std::cout << "............... right_point_2ds.size():" << right_point_2ds.size() << std::endl;
    undistortPoints(left_point_2ds,  normalizedLeftPts,  K, cv::Mat());
    undistortPoints(right_point_2ds, normalizedRightPts, K, cv::Mat());

    cv::Mat points3dHomogeneous;
    triangulatePoints(Pleft, Pright, normalizedLeftPts, normalizedRightPts, points3dHomogeneous);

    cv::Mat points3d;
    convertPointsFromHomogeneous(points3dHomogeneous.t(), points3d);

    cv::Mat rvecLeft;
    Rodrigues(Pleft.get_minor<3, 3>(0, 0), rvecLeft);
    cv::Mat tvecLeft(Pleft.get_minor<3, 1>(0, 3).t());

    std::vector<cv::Point2d> projectedOnLeft(left_point_2ds.size());
    projectPoints(points3d, rvecLeft, tvecLeft, K, cv::Mat(), projectedOnLeft);

    cv::Mat rvecRight;
    Rodrigues(Pright.get_minor<3, 3>(0, 0), rvecRight);
    cv::Mat tvecRight(Pright.get_minor<3, 1>(0, 3).t());

    std::vector<cv::Point2d> projectedOnRight(right_point_2ds.size());
    projectPoints(points3d, rvecRight, tvecRight, K, cv::Mat(), projectedOnRight);

    //Note: cheirality check (all points z > 0) was already performed at camera pose calculation

    for (size_t i = 0; i < points3d.rows; i++)
    {
        //check if point reprojection error is small enough
        double l_reprojection_error = cv::norm(projectedOnLeft[i]  - left_point_2ds[i]);
        double r_reprojection_error = cv::norm(projectedOnRight[i] - right_point_2ds[i]);
        //std::cout << "left reprojection error:" << l_reprojection_error << std::endl;
        //std::cout << "right reprojection error:" << r_reprojection_error << std::endl;
        if (l_reprojection_error > MIN_REPROJECTION_ERROR or
            r_reprojection_error > MIN_REPROJECTION_ERROR)
        {
            continue;
        }

        CloudPoint clp;
        clp.p3d = cv::Point3d(points3d.at<double>(i, 0),
                              points3d.at<double>(i, 1),
                              points3d.at<double>(i, 2));
        clp.index_of_2d_origin = std::vector<int>(mImagesBgrVec.size(), -1);
        clp.index_of_2d_origin[left_view_index] = matches[i].queryIdx;
        clp.index_of_2d_origin[right_view_index] = matches[i].trainIdx;
        clp.reprojection_error = (l_reprojection_error + r_reprojection_error) / 2;

        new_cloud.push_back(clp);
    }
}

void Initializer::MergeNewPointCloud(const std::vector<CloudPoint>& new_cloud,
                                     std::vector<CloudPoint>& new_cloud_filtered)
{
    const size_t numImages = mImagesGrayVec.size();

    std::map<std::pair<int, int>, std::vector<cv::DMatch>> mergeMatchesTable;

    size_t newPoints = 0;
    size_t mergedPoints = 0;

    for (const CloudPoint& newPoint : new_cloud)
    {
        //const cv::Point3d newPoint = p.p3d;   // new 3D point

        bool foundMatching3DPoint = false;
        bool foundAnyMatchingExistingViews = false;
        for(CloudPoint& existingPoint : this->mCloud)
        {
            if (norm(existingPoint.p3d - newPoint.p3d) > MERGE_CLOUD_POINT_MIN_MATCH_DISTANCE)
                continue;

            //This point is very close to an existing 3D cloud point
            foundMatching3DPoint = true;

            //Look for common 2D features to confirm match
            for (int new_origin_view_idx = 0; new_origin_view_idx < numImages; new_origin_view_idx++)
            {
                int newPoint_origin_kp_idx = newPoint.index_of_2d_origin[new_origin_view_idx];
                if (newPoint_origin_kp_idx == -1)
                    continue;

                for (int existing_origin_view_idx = 0; existing_origin_view_idx < numImages; existing_origin_view_idx++)
                {
                    int existingPoint_origin_kp_idx = existingPoint.index_of_2d_origin[existing_origin_view_idx];
                    if (existingPoint_origin_kp_idx == -1)
                        continue;

                    bool newIsLeft = new_origin_view_idx < existing_origin_view_idx;
                    int leftViewIdx         = (newIsLeft) ? new_origin_view_idx : existing_origin_view_idx;
                    int leftViewFeatureIdx  = (newIsLeft) ? newPoint_origin_kp_idx : existingPoint_origin_kp_idx;
                    int rightViewIdx        = (newIsLeft) ? existing_origin_view_idx : new_origin_view_idx;
                    int rightViewFeatureIdx = (newIsLeft) ? existingPoint_origin_kp_idx : newPoint_origin_kp_idx;

                    bool foundMatchingFeature = false;

                    std::vector<cv::DMatch>& matches = mMatchesTable[std::make_pair(leftViewIdx, rightViewIdx)];
                    for (const cv::DMatch& match : matches)
                    {
                        if (    match.queryIdx == leftViewFeatureIdx
                                and match.trainIdx == rightViewFeatureIdx
                                and match.distance < MERGE_CLOUD_FEATURE_MIN_MATCH_DISTANCE)
                        {
                            mergeMatchesTable[std::make_pair(leftViewIdx, rightViewIdx)].push_back(match);

                            //Found a 2D feature match for the two 3D points - merge
                            foundMatchingFeature = true;
                            break;
                        }
                    }

                    if (foundMatchingFeature)
                    {
                        // Add the new originating view, and feature index
                        // merge happens here
                        existingPoint.index_of_2d_origin[new_origin_view_idx] = newPoint_origin_kp_idx;
                        foundAnyMatchingExistingViews = true;
                    }

                } // end origins of existing point  loop
            }  // end origins of new point loop

            if (foundAnyMatchingExistingViews)
            {
                mergedPoints++;
                break; //Stop looking for more matching cloud points
            }
        } // end existing point loop

        if (not foundAnyMatchingExistingViews and not foundMatching3DPoint)
        {
            //This point did not match any existing cloud points - add it as new.
            mCloud.push_back(newPoint);
            new_cloud_filtered.push_back(newPoint);
            newPoints++;
        }
    } // end new point loop

    std::cout << " adding: " << new_cloud.size()
              << " (new: " << newPoints << ", merged: "
              << mergedPoints << ")" << std::endl;
}

bool Initializer::TriangulatePointsBetweenViews(int curr_view_index,
                                                int other_view_index,
                                                std::vector<CloudPoint>& new_cloud,
                                                std::vector<int>& add_to_cloud)
{
    std::cout << " *** Triangulate " << mImageNamesVec[curr_view_index] << " and " << mImageNamesVec[other_view_index] << ":" << std::endl;

    if (Pmats.find(other_view_index) == Pmats.end())
    {
        std::cout << "Can not find Pmats[" << other_view_index << "]" << std::endl;
        return false;
    }

    cv::Matx34d P1 = Pmats[other_view_index];
    cv::Matx34d P2 = Pmats[curr_view_index];

    std::vector<cv::Point2d> point_2ds_1, point_2ds_2;
    std::vector<cv::DMatch> matches = mMatchesTable[std::make_pair(other_view_index, curr_view_index)];
    get_aligned_point2ds(mKeyPointsVec[other_view_index],
                         mKeyPointsVec[curr_view_index],
                         matches,
                         point_2ds_1,
                         point_2ds_2);

    cv::Mat triangulated_pts_4d_mat;
    cv::triangulatePoints(P1, P2, point_2ds_1, point_2ds_2, triangulated_pts_4d_mat);

    std::vector<cv::Point3d> point3ds_vec;
    mpEpipolarSolver->Point4DMatToPoint3dVec(triangulated_pts_4d_mat, point3ds_vec);

    // construct new points
    cv::Mat KP = mpEpipolarSolver->GetCameraMat() * cv::Mat(Pmats[curr_view_index]);
    new_cloud.clear();
    std::vector<double> reprj_errors_vec;
    for (auto i = 0; i < point3ds_vec.size(); i++)
    {
        CloudPoint clp;
        clp.p3d = point3ds_vec[i];
        clp.index_of_2d_origin = std::vector<int>(mImagesBgrVec.size(), -1);
        clp.index_of_2d_origin[other_view_index] = matches[i].queryIdx;
        clp.index_of_2d_origin[curr_view_index] = matches[i].trainIdx;

        cv::Mat homo_p3d = cv::Mat(4, 1, CV_64F, {clp.p3d.x, clp.p3d.y, clp.p3d.z, 1.0});
        cv::Mat p2d_in_img2_mat = KP * homo_p3d;
//    std::cout << point3ds_vec[i] << " --> (" << p2d_in_img2_mat << ")"
//              << "kp:" << mKeyPointsVec[curr_view_index][matches[i].trainIdx].pt << std::endl;
        cv::Point2f p2d_in_img2(p2d_in_img2_mat.at<double>(0) / p2d_in_img2_mat.at<double>(2),
                                p2d_in_img2_mat.at<double>(1) / p2d_in_img2_mat.at<double>(2));
        clp.reprojection_error = cv::norm(p2d_in_img2 - mKeyPointsVec[curr_view_index][matches[i].trainIdx].pt);

        new_cloud.push_back(clp);
        reprj_errors_vec.push_back(clp.reprojection_error);
    }
    std::sort(reprj_errors_vec.begin(), reprj_errors_vec.end());

    // filter outstanding reprojection_errors
    // get the 80% precentile
    //threshold from Snavely07 4.2
    double reprj_err_cutoff = reprj_errors_vec[4 * reprj_errors_vec.size() / 5] * 2.4;

    add_to_cloud.clear();
    add_to_cloud.resize(new_cloud.size(), 1);
    for(int i = 0;i < new_cloud.size(); i++)
    {
        // std::cout << "reprojection_error[" << i << "]:" << new_cloud[i].reprojection_error << std::endl;

        if(new_cloud[i].reprojection_error > 1000.0)  // > 16
        {
            add_to_cloud[i] = 0;
            continue; //reject point
        }

        /*
        if (new_cloud[i].reprojection_error >= 4.0 &&
            new_cloud[i].reprojection_error >= reprj_err_cutoff)
        {
          add_to_cloud[i] = 0;
          continue;
        }
         */
    }
    std::cout << "filtered out " << (new_cloud.size() - cv::countNonZero(add_to_cloud)) << " high-error points" << std::endl;

    //all points filtered?
    if(cv::countNonZero(add_to_cloud) == 0)
    {
        std::cout << "All points filtered out!!!" << std::endl;
        return false;
    }

    //scan new triangulated points, if they were already triangulated before - strengthen cloud
    int found_other_views_count = 0;
    int num_views = mImagesBgrVec.size();
    for (int j = 0; j < new_cloud.size(); j++)
    {
        if (add_to_cloud[j] == 0) continue;

        bool found_in_other_view = false;
        for (auto view_idx = 0; view_idx < num_views; view_idx++)
        {
            if (view_idx == other_view_index ||
                view_idx == curr_view_index)
                continue;

            std::vector<cv::DMatch> submatches = mMatchesTable[std::make_pair(view_idx, curr_view_index)];

            for (auto m = 0; m < submatches.size(); m++)
            {
                if (found_in_other_view) break;

                if (submatches[m].trainIdx != matches[j].trainIdx)
                    continue;

                new_cloud[j].index_of_2d_origin[view_idx] = submatches[m].queryIdx;

                for (auto pt_idx = 0; pt_idx < mCloud.size(); pt_idx++)
                {
                    if (mCloud[pt_idx].index_of_2d_origin[view_idx] != submatches[m].queryIdx)
                        continue;

                    mCloud[pt_idx].index_of_2d_origin[curr_view_index] = matches[j].trainIdx;
                    mCloud[pt_idx].index_of_2d_origin[other_view_index] = matches[j].queryIdx;
                    found_in_other_view = true;
                    break;
                }
            }
        }

        if (found_in_other_view)
        {
            found_other_views_count++;
        }
        else
        {
            add_to_cloud[j] = 1;
        }
    }

    std::cout << found_other_views_count << "/" << new_cloud.size() << " points were found in other views, adding " << cv::countNonZero(add_to_cloud) << " new\n";
    return true;
}

void Initializer::SaveCloudAndCamerasToPLY(const std::string& prefix)
{
    std::ofstream ofs(prefix + "_points.ply");

    ofs << "ply" << std::endl <<
        "format ascii 1.0" << endl <<
        "element vertex " << mCloud.size() << std::endl <<
        "property float x" << std::endl <<
        "property float y" << std::endl <<
        "property float z" << std::endl <<
        "property uchar red" << std::endl <<
        "property uchar green" << std::endl <<
        "property uchar blue" << std::endl <<
        "end_header" << std::endl;

    for (const CloudPoint& p : mCloud)
    {
        //get color from first originating view
        /*
        auto originatingView = p.index_of_2d_origin.begin();
        const int viewIdx = originatingView->first;
        cv::Point2f p2d = mImageFeatures[viewIdx].points[originatingView->second];
        cv::Vec3b pointColor = mImagesBgrVec[viewIdx].at<Vec3b>(p2d);
         */

        std::vector<cv::Vec3b> colors;
        auto view = 0;
        for (view = 0; view < p.index_of_2d_origin.size(); view++)
        {
            int kp_idx = p.index_of_2d_origin[view];
            if (kp_idx != -1)
            {
                cv::Point point2d = mKeyPointsVec[view][kp_idx].pt;
                colors.push_back(((cv::Mat)(mImagesBgrVec[view])).at<cv::Vec3b>(point2d));
            }
        }

        //cv::Scalar res_color = cv::mean(colors);
        //cv::Vec3b pointColor = (cv::Vec3b(res_color[2],res_color[1],res_color[0])); // bgr -> rgb
        cv::Scalar pointColor = cv::mean(colors);

        ofs << p.p3d.x       << " " <<
            p.p3d.y              << " " <<
            p.p3d.z              << " " <<
            (int)pointColor(2) << " " <<
            (int)pointColor(1) << " " <<
            (int)pointColor(0) << std::endl;
    }

    ofs.close();

    ofstream ofsc(prefix + "_cameras.ply");

    ofsc << "ply" << std::endl <<
         "format ascii 1.0" << std::endl <<
         "element vertex " << (Pmats.size() * 4) << std::endl <<
         "property float x" << std::endl <<
         "property float y" << std::endl <<
         "property float z" << std::endl <<
         "element edge " << (Pmats.size() * 3) << std::endl <<
         "property int vertex1" << std::endl <<
         "property int vertex2" << std::endl <<
         "property uchar red" << std::endl <<
         "property uchar green" << std::endl <<
         "property uchar blue" << std::endl <<
         "end_header" << std::endl;

    for (auto it = Pmats.begin(); it != Pmats.end(); it++)
    {
        cv::Matx34d pose = it->second;
        //std::cout << (it->first) << ":" << it->second << std::endl;
        cv::Point3d c(pose(0, 3), pose(1, 3), pose(2, 3));
        cv::Point3d cx = c + cv::Point3d(pose(0, 0), pose(1, 0), pose(2, 0)) * 0.2;
        cv::Point3d cy = c + cv::Point3d(pose(0, 1), pose(1, 1), pose(2, 1)) * 0.2;
        cv::Point3d cz = c + cv::Point3d(pose(0, 2), pose(1, 2), pose(2, 2)) * 0.2;

        ofsc << c.x  << " " << c.y  << " " << c.z  << endl;
        ofsc << cx.x << " " << cx.y << " " << cx.z << endl;
        ofsc << cy.x << " " << cy.y << " " << cy.z << endl;
        ofsc << cz.x << " " << cz.y << " " << cz.z << endl;
    }

    //const int camVertexStartIndex = mCloud.size();

    for (size_t i = 0; i < Pmats.size(); i++) {
        ofsc << (i * 4 + 0) << " " <<
             (i * 4 + 1) << " " <<
             "255 0 0" << endl;
        ofsc << (i * 4 + 0) << " " <<
             (i * 4 + 2) << " " <<
             "0 255 0" << endl;
        ofsc << (i * 4 + 0) << " " <<
             (i * 4 + 3) << " " <<
             "0 0 255" << endl;
    }

    ofsc.close();
}

void Initializer::SaveCamerasInvToPLY(const std::string& prefix)
{
    ofstream ofsc(prefix + "_cameras.ply");

    ofsc << "ply" << std::endl <<
         "format ascii 1.0" << std::endl <<
         "element vertex " << (Pmats.size() * 4) << std::endl <<
         "property float x" << std::endl <<
         "property float y" << std::endl <<
         "property float z" << std::endl <<
         "element edge " << (Pmats.size() * 3) << std::endl <<
         "property int vertex1" << std::endl <<
         "property int vertex2" << std::endl <<
         "property uchar red" << std::endl <<
         "property uchar green" << std::endl <<
         "property uchar blue" << std::endl <<
         "end_header" << std::endl;

    for (int i = 0; i < Pmats.size(); i++)
    {
        Sophus::SE3d pose = GetNthFramePose(i).inverse();
        auto rotation = pose.rotationMatrix();
        auto translation = pose.translation();

        cv::Point3d c(translation(0, 0), translation(1, 0), translation(2, 0));
        cv::Point3d cx = c + cv::Point3d(rotation(0, 0), rotation(1, 0), rotation(2, 0)) * 0.2;
        cv::Point3d cy = c + cv::Point3d(rotation(0, 1), rotation(1, 1), rotation(2, 1)) * 0.2;
        cv::Point3d cz = c + cv::Point3d(rotation(0, 2), rotation(1, 2), rotation(2, 2)) * 0.2;

        ofsc << c.x  << " " << c.y  << " " << c.z  << endl;
        ofsc << cx.x << " " << cx.y << " " << cx.z << endl;
        ofsc << cy.x << " " << cy.y << " " << cy.z << endl;
        ofsc << cz.x << " " << cz.y << " " << cz.z << endl;
    }

    //const int camVertexStartIndex = mCloud.size();

    for (size_t i = 0; i < Pmats.size(); i++) {
        ofsc << (i * 4 + 0) << " " <<
             (i * 4 + 1) << " " <<
             "255 0 0" << endl;
        ofsc << (i * 4 + 0) << " " <<
             (i * 4 + 2) << " " <<
             "0 255 0" << endl;
        ofsc << (i * 4 + 0) << " " <<
             (i * 4 + 3) << " " <<
             "0 0 255" << endl;
    }

    ofsc.close();
}

