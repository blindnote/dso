//
// Created by Yin Rochelle on 13/06/2017.
//

#ifndef DSO_IFRAME_H
#define DSO_IFRAME_H

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

class iFrame
{
 public:
  iFrame(const std::string& name,
        const cv::Mat& img_bgr, const cv::Mat& img_gray,
        const std::vector<cv::KeyPoint>& keypoints,
        const cv::Matx34d& pose = cv::Mat(3, 4, CV_64F));

    iFrame(const std::string& name,
        const cv::Mat& img_bgr, const cv::Mat& img_gray);

  void SetPose(const cv::Matx34d& pose);
  void SetDesp(bool replace = true);

  iFrame& operator=(const iFrame& rhs)
  {
    if (this == &rhs)
      return *this;

    this->mImageName = rhs.mImageName;
    this->mImageBgr = rhs.mImageBgr.clone();
    this->mImageGray = rhs.mImageGray.clone();
    this->mDescriptors = rhs.mDescriptors.clone();
    this->mKeyPoints.clear();
    this->mKeyPoints.insert(this->mKeyPoints.end(), rhs.mKeyPoints.begin(), rhs.mKeyPoints.end());
    ((cv::Mat)(rhs.mPoseTransform)).copyTo(this->mPoseTransform);

    return *this;
  }

public:
  std::string mImageName;
  cv::Mat mImageBgr, mImageGray;

  cv::Mat mDescriptors;
  std::vector<cv::KeyPoint> mKeyPoints;

  cv::Matx34d mPoseTransform;
};


#endif //DSO_IFRAME_H
