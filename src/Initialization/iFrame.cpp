//
// Created by Yin Rochelle on 13/06/2017.
//

#include "iFrame.h"
#include "iFeatureExtractor.h"

iFrame::iFrame(const std::string& name,
             const cv::Mat& img_bgr,
             const cv::Mat& img_gray)
        : mImageName(name)
{
    mImageBgr = img_bgr.clone();
    mImageGray = img_gray.clone();

    SetDesp(true);
    mPoseTransform = cv::Matx34d(3, 4, CV_64F);
}

iFrame::iFrame(const std::string& name,
             const cv::Mat& img_bgr,
             const cv::Mat& img_gray,
             const std::vector<cv::KeyPoint>& keypoints,
             const cv::Matx34d& pose)
        : mImageName(name)
{
    // img_bgr.copyTo(mImageBgr);
    // img_gray.copyTo(mImageGray);
    mImageBgr = img_bgr.clone();
    mImageGray = img_gray.clone();

    mKeyPoints.assign(keypoints.begin(), keypoints.end());
    SetDesp(false);
    SetPose(pose);
}

void iFrame::SetDesp(bool replace)
{
    iFeatureExtractor extractor(eDetectorType::ORB);
    if (replace)
    {
        extractor.ExtractFeatures(mImageGray, mDescriptors, mKeyPoints);
        return;
    }

    std::vector<cv::KeyPoint> tmpKeypoints;
    extractor.ExtractFeatures(mImageGray, mDescriptors, tmpKeypoints);
}

void iFrame::SetPose(const cv::Matx34d& pose)
{
    (cv::Mat(pose)).copyTo(mPoseTransform);
}
