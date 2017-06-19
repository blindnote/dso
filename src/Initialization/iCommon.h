//
// Created by Yin Rochelle on 13/06/2017.
//

#ifndef DSO_ICOMMON_H
#define DSO_ICOMMON_H

#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <Eigen/Eigen>

///Rotational element in a 3x4 matrix
const cv::Rect ROT(0, 0, 3, 3);

///Translational element in a 3x4 matrix
const cv::Rect TRA(3, 0, 1, 3);

const double RANSAC_THRESHOLD = 10.0;
const double POSE_INLIERS_MINIMAL_RATIO = 0.5;
const float MIN_REPROJECTION_ERROR = 10.0;
const float MERGE_CLOUD_POINT_MIN_MATCH_DISTANCE   = 0.01;
const float MERGE_CLOUD_FEATURE_MIN_MATCH_DISTANCE = 20.0;

struct CloudPoint
{
  cv::Point3d p3d;
  std::vector<int> index_of_2d_origin;
  double reprojection_error;
};

struct InvDepthPnt
{
    float u,v;
    float idepth;
    float depth;
    bool isGood;
};

struct ImagePair
{
  size_t left, right;
};

std::ostream& operator<< (std::ostream& stream, const ImagePair& pair);

std::vector<cv::Point3d> cloudpoints_to_point3ds(const std::vector<CloudPoint>& cloud);

void load_imgs_from_dir(const char* dir,
                        std::vector<cv::Mat>& images,
                        std::vector<std::string>& image_names,
                        bool names_only = false);

void convert_imgs_to_8U3(const std::vector<cv::Mat>& imgs_vec_in,
                         std::vector<cv::Mat>& imgs_bgr_vec,
                         std::vector<cv::Mat>& imgs_gray_vec);
void convert_image_to_8U3(const cv::Mat img_in,
                          cv::Mat& img_bgr,
                          cv::Mat& img_gray);

void keypoints_to_point2ds(const std::vector<cv::KeyPoint>& keypoints,
                           std::vector<cv::Point2d>& point2ds_vec);

void get_aligned_keypoints(const std::vector<cv::KeyPoint>& kp_i_vec,
                           const std::vector<cv::KeyPoint>& kp_j_vec,
                           const std::vector<cv::DMatch>& matches,
                           std::vector<cv::KeyPoint>& aligned_kp_i_vec,
                           std::vector<cv::KeyPoint>& aligned_kp_j_vec);

void get_aligned_point2ds(const std::vector<cv::KeyPoint>& kp_i_vec,
                          const std::vector<cv::KeyPoint>& kp_j_vec,
                          const std::vector<cv::DMatch>& matches,
                          std::vector<cv::Point2d>& point2ds_i_vec,
                          std::vector<cv::Point2d>& point2ds_j_vec);


#endif //DSO_ICOMMON_H
