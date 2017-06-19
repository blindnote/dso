//
// Created by Yin Rochelle on 13/06/2017.
//

#ifndef DSO_IFEATUREEXTRACTOR_H
#define DSO_IFEATUREEXTRACTOR_H

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

enum eDetectorType
{
  ORB = 0,
  AKAZE = 1
};

class iFeatureExtractor
{
public:
  iFeatureExtractor(eDetectorType type = eDetectorType::ORB);

  void ExtractFeatures(const cv::Mat& image,
                       cv::Mat& descriptors,
                       std::vector<cv::KeyPoint>& keypoints);

  void ExtractFeaturesBatch(const std::vector<cv::Mat>& imagesVec,
                            std::vector<cv::Mat>& descriptorsVec,
                            std::vector<std::vector<cv::KeyPoint>>& keypointsVec);

  void DrawKeyPoints(const std::string name, const cv::Mat& image,
                     const std::vector<cv::KeyPoint>& keypoints);
  void DrawKeyPointsBatch(const std::vector<cv::Mat>& imagesVec,
                          const std::vector<std::vector<cv::KeyPoint>>& keypointsVec);

private:
  cv::Ptr<cv::FeatureDetector> mpDetector;
};

#endif //DSO_IFEATUREEXTRACTOR_H
