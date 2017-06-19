//
// Created by Yin Rochelle on 13/06/2017.
//

#include <iostream>
#include <opencv2/calib3d.hpp>
#include "iEpipolarSolver.h"

iEpipolarSolver::iEpipolarSolver(const Eigen::Matrix<double,3,3>& k)
{
  mbKUnknown = false;
  mK = (cv::Mat_<double>(3,3) << k(0, 0), k(0, 1), k(0, 2),
          k(1, 0), k(1, 1), k(1, 2),
          k(2, 0), k(2, 1), k(2, 2));
  mDistortionCoeffs = cv::Mat_<double>::zeros(1,4);

  std::cout << std::endl << "K:" << std::endl << mK << std::endl;
}

bool iEpipolarSolver::RecoverFromAlignedPoints(const std::vector<cv::Point2d>& point2d_i_vec,
                                              const std::vector<cv::Point2d>& point2d_j_vec,
                                              cv::Mat& R, cv::Mat& t,
                                              std::vector<cv::Point3d>& point3ds_vec,
                                              std::vector<uchar>& recover_masks)
{
  std::vector<uchar> calc_essential_masks;
  cv::Mat E = CalcEssentialMat(point2d_i_vec,
                               point2d_j_vec,
                               calc_essential_masks);
  if (cv::countNonZero(calc_essential_masks) < 6)
  {
    std::cout << "Fail to find enough inliers when calcuating essential matrix!" << std::endl;
    return false;
  }

  //according to http://en.wikipedia.org/wiki/Essential_matrix#Properties_of_the_essential_matrix
  if(fabs(determinant(E)) > 1e-07)
  {
    std::cout << "[!!]det(E) != 0 : " << determinant(E) << "\n";
  //  P1 = 0;
    return false;
  }

//  cv::Mat R1(3, 3, CV_64FC1), R2(3, 3, CV_64FC1);
//  cv::Mat t1(1, 3, CV_64FC1), t2(1, 3, CV_64FC1);
  cv::Mat triangulated_pts_mat;
  int recover_pose_inliers_cnt = recoverPose(E, point2d_i_vec, point2d_j_vec,
                                             mK, R, t, 40.0,
                                             recover_masks, triangulated_pts_mat);

  double percent = 100 * recover_pose_inliers_cnt / recover_masks.size();
  if ((recover_pose_inliers_cnt) < 10 || (percent < 50))
  {
    std::cout << "[!!]recover pose doesn't get good results: inliers count:"
              << recover_pose_inliers_cnt << ", percent:" << percent << " %" << std::endl;
    return false;
  }

  std::cout << "R:" << std::endl << R << std::endl;
  std::cout << "t:" << std::endl << t << std::endl;
  std::cout << "recover pose got good results: inliers count:"
            << recover_pose_inliers_cnt << ", percent:" << percent << " %" << std::endl;
  Point4DMatToPoint3dVec(triangulated_pts_mat,
                   //      recover_masks,
                         point3ds_vec);
  /*
  std::cout << "Got " << recover_pose_inliers_cnt << " triangulated points:"<< std::endl;
  for (int i = 0; i < point3ds_vec.size(); i++)
  {
    if (!recover_masks[i]) continue;
    std::cout << i << ":" << point3ds_vec[i] << std::endl;
  }
  */

  return true;
}


bool iEpipolarSolver::CheckCoherentRotation(cv::Mat& R)
{
  if(fabs(determinant(R))-1.0 > 1e-07)
  {
    std::cout << "det(R) != +-1.0, this is not a rotation matrix" << std::endl;
    return false;
  }

  return true;
}

cv::Mat iEpipolarSolver::CalcEssentialMat(const std::vector<cv::Point2d>& points2d_i_vec,
                                         const std::vector<cv::Point2d>& points2d_j_vec,
                                         std::vector<uchar>& inliers)
{
  inliers.clear();

  cv::Mat E;
  if (mbKUnknown)
  {
    // find essential matrix via fundamental matrix
    cv::Mat F = findFundamentalMat(points2d_i_vec, points2d_j_vec,
                                   cv::FM_RANSAC, 3., 0.99, inliers);
    /*
     double minVal,maxVal;
     cv::minMaxIdx(points2d_i_vec, &minVal, &maxVal);
     cv::Mat F = findFundamentalMat(points2d_i_vec, points2d_j_vec,
     cv::FM_RANSAC, 0.006 * maxVal, 0.99, status);
     */
    if (F.data == NULL)
    {
      std::cout << "i_size:" << points2d_i_vec.size()
                << ", j_size:" << points2d_j_vec.size()
                << std::endl;

      inliers.clear();
      return E;
    }

    //Essential matrix: compute then extract cameras [R|t]
    E = mK.t() * F * mK; //according to HZ (9.12)
  }
  else
  {
    if (points2d_i_vec.size() < 5)
    {
      std::cout << "Unable to calculate Essential Matrix" << std::endl;
      return E;
    }

    // find essential matrix directly
    E = findEssentialMat(points2d_i_vec, points2d_j_vec, mK,
                         cv::RANSAC, 0.999, 1.0, inliers);
  }

  //std::cout << "E: " << std::endl << E << std::endl;
  return E;
}


void iEpipolarSolver::Point4DMatToPoint3dVec(const cv::Mat& points4D,
                                        //    const std::vector<uchar>& masks,
                                            std::vector<cv::Point3d>& point3dsVec)
{
  assert(points4D.rows == 4);

  for(int i = 0; i < points4D.cols; i++)
  {
  //  if (!masks[i]) continue;

    double w = points4D.at<double>(3, i);
    if (fabs(w) < 1e-07) continue;

    double x = points4D.at<double>(0, i);
    double y = points4D.at<double>(1, i);
    double z = points4D.at<double>(2, i);

    cv::Point3d point(x/w, y/w, z/w);
    point3dsVec.push_back(point);
  }
}
