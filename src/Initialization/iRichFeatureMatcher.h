//
// Created by Yin Rochelle on 13/06/2017.
//

#ifndef DSO_IRICHFEATUREMATCHER_H
#define DSO_IRICHFEATUREMATCHER_H


#include <memory>
#include "opencv2/features2d.hpp"
#include "iFeatureMatcher.h"

enum eMatcherType
{
  FlannBased = 0,
  BruteForce = 1
};

class iRichFeatureMatcher : public iFeatureMatcher
{
public:
  iRichFeatureMatcher(eMatcherType type = eMatcherType::FlannBased);

  void Match(const cv::Mat& desp_i,
             const cv::Mat& desp_j,
             const std::vector<cv::KeyPoint>& kp_i,
             const std::vector<cv::KeyPoint>& kp_j,
             std::vector<cv::DMatch>& matches);

  void Match_Homo(const cv::Mat& desp_i,
                  const cv::Mat& desp_j,
                  const std::vector<cv::KeyPoint>& kp_i,
                  const std::vector<cv::KeyPoint>& kp_j,
                  std::vector<cv::DMatch>& matches);

  static void FlipMatches(const std::vector<cv::DMatch>& matches,
                          std::vector<cv::DMatch>& matches_flipped);

  static void DrawMatches(const std::string name,
                          const cv::Mat& img_i, const cv::Mat& img_j,
                          const std::vector<cv::KeyPoint>& kp_i,
                          const std::vector<cv::KeyPoint>& kp_j,
                          const std::vector<cv::DMatch>& matches,
                          float scale = 1.0);

private:
  const int Min_Matches_Num = 8;
  const double Knn_Match_Ratio = 0.8; // 1.f / 1.5f
  const double Reprojection_Threshhold = 2.5f;

  std::shared_ptr<cv::DescriptorMatcher> mpMatcher;
};

#endif //DSO_IRICHFEATUREMATCHER_H
