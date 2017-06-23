//
// Created by Yin Rochelle on 13/06/2017.
//

#ifndef DSO_INITIALIZER_H
#define DSO_INITIALIZER_H

#include <memory>
#include <vector>
#include <map>
#include <opencv2/core.hpp>
#include <sophus/se3.hpp>
#include "iCommon.h"
#include "iFrame.h"
#include "iEpipolarSolver.h"


class Initializer
{
public:
    Initializer(const std::vector<cv::Mat>& images,
                const std::vector<std::string>& image_names,
                const std::shared_ptr<iEpipolarSolver> solver_ptr,
                std::vector<CloudPoint>& cloud);

    static bool FindPoseEstimation(const cv::Mat& K,
                                   const cv::Mat& distortion,
                                   cv::Mat& R, cv::Mat& tvec,
                                   std::vector<cv::Point3d> point3ds_vec,
                                   std::vector<cv::Point2d> point2ds_vec);

    static bool FindExtrinsicsFromMatches(const cv::Mat& K,
                                          const std::vector<cv::Point2d>& left_point2ds,
                                          const std::vector<cv::Point2d>& right_point2ds,
                                          const std::vector<cv::DMatch>& matches,
                                          std::vector<cv::DMatch>& prunedMatches,
                                          cv::Matx34d& Pleft, cv::Matx34d& Pright);

    bool RecoverDepth();

    std::vector<iFrame> GetBaseLineFrames();
    const std::vector<std::string>& GetImageFileNames() { return mImageNamesVec; }

    Sophus::SE3d GetFirstFramePose();
    Sophus::SE3d GetSecondFramePose();
    inline int GetFirstViewIndex() { return miFirstViewIndex; }
    inline int GetSecondViewIndex() { return miSecondViewIndex; }

    void BuildInitialSceneForDso(std::vector<InvDepthPnt>& idpts, Sophus::SE3d& thisToNext);

private:
    void MatchFeatures();       // brute-force
    int findHomographyInliers(const std::vector<cv::KeyPoint>& left,
                              const std::vector<cv::KeyPoint>& right,
                              const std::vector<cv::DMatch>& matches);
    std::map<float, ImagePair> sortViewsForBaseline();
    //std::map<float, ImagePair, std::greater<float>> sortViewsForBaseline();

    void MatchFeatures2();       // flann
    void ConstructBaseLine();
    void ConstructBaseLineSpecify();
    void EnrichScene();

    void Find3d2dCorrespondences(int curr_view_index,
                                 std::vector<cv::Point3d>& point3ds_vec,
                                 std::vector<cv::Point2d>& point2ds_vec);

    bool TriangulatePointsBetweenViews(int curr_view_index,
                                       int other_view_index,
                                       std::vector<CloudPoint>& new_cloud,
                                       std::vector<int>& add_to_cloud);

    void TriangulateViews(int left_view_index, int right_view_index,
                          const std::vector<cv::DMatch>& matches,
                          const cv::Matx34d& Pleft, const cv::Matx34d& Pright,
                          std::vector<CloudPoint>& new_cloud);

    void MergeNewPointCloud(const std::vector<CloudPoint>& new_cloud,
                            std::vector<CloudPoint>& new_cloud_filtered);

    void SaveCloudAndCamerasToPLY(const std::string& prefix);
    void SaveCamerasInvToPLY(const std::string& prefix);

    Sophus::SE3d GetNthFramePose(int id);
    Sophus::SE3d GetSE3From3x4Mat(const cv::Matx34d& pose);
    cv::Matx34d GetInvFromSophusPose(const Sophus::SE3d& pose_se3);


private:
    std::vector<cv::Mat> mImagesBgrVec, mImagesGrayVec;
    std::vector<std::string> mImageNamesVec;

    std::vector<std::vector<cv::KeyPoint>> mKeyPointsVec;
    std::map<std::pair<int, int>, std::vector<cv::DMatch>> mMatchesTable;

    std::map<int, cv::Matx34d> Pmats;   // right->left OR camToWorld
    std::vector<int> mDoneViewsIdxVec;
    std::vector<CloudPoint>& mCloud;

    int miFirstViewIndex, miSecondViewIndex;

    std::shared_ptr<iEpipolarSolver> mpEpipolarSolver;
};

#endif //DSO_INITIALIZER_H
