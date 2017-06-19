//
// Created by Yin Rochelle on 13/06/2017.
//

#ifndef DSO_IEPIPOLARSOLVER_H
#define DSO_IEPIPOLARSOLVER_H


#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "iCommon.h"


class iEpipolarSolver
{
public:
  iEpipolarSolver(const Eigen::Matrix<double,3,3>& k);

  bool RecoverFromAlignedPoints(const std::vector<cv::Point2d>& point2d_i_vec,
                                const std::vector<cv::Point2d>& point2d_j_vec,
                                cv::Mat& R, cv::Mat& t,
                                std::vector<cv::Point3d>& point3ds_vec,
                                std::vector<uchar>& recover_masks);

  void Point4DMatToPoint3dVec(const cv::Mat& points4D,
                              // const std::vector<uchar>& masks,
                              std::vector<cv::Point3d>& point3dsVec);

  bool CheckCoherentRotation(cv::Mat& R);

  cv::Mat GetCameraMat() { return mK; };
  cv::Mat GetDistortCoeffs() { return mDistortionCoeffs; };


private:
  cv::Mat CalcEssentialMat(const std::vector<cv::Point2d>& points2d_i_vec,
                           const std::vector<cv::Point2d>& points2d_j_vec,
                           std::vector<uchar>& inliers);

private:
  bool mbKUnknown;
  cv::Mat mK;
  cv::Mat mDistortionCoeffs;
};


#endif //DSO_IEPIPOLARSOLVER_H
