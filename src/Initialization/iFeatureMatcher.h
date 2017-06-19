//
// Created by Yin Rochelle on 13/06/2017.
//

#ifndef DSO_IFEATUREMATCHER_H
#define DSO_IFEATUREMATCHER_H

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

class iFeatureMatcher
{
public:
  virtual void Match(const cv::Mat& desp_i,
                     const cv::Mat& desp_j,
                     const std::vector<cv::KeyPoint>& kp_i,
                     const std::vector<cv::KeyPoint>& kp_j,
                     std::vector<cv::DMatch>& matches) = 0;
};


#endif //DSO_IFEATUREMATCHER_H
