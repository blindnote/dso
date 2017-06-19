//
// Created by Yin Rochelle on 13/06/2017.
//

#include "iCommon.h"

#include <dirent.h>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "iCommon.h"

std::ostream& operator<< (std::ostream& stream, const ImagePair& pair)
{
    return stream << "[" << pair.left << ", " << pair.right << "]";
}

bool hasEnding (std::string const &fullString, std::string const &ending)
{
    if (fullString.length() >= ending.length())
    {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    }
    else
    {
        return false;
    }
}

bool hasEndingLower (std::string const &fullString_, std::string const &_ending)
{
    std::string fullstring = fullString_, ending = _ending;
    transform(fullString_.begin(),fullString_.end(),fullstring.begin(),::tolower); // to lower
    return hasEnding(fullstring,ending);
}

void load_imgs_from_dir(const char* dir,
                        std::vector<cv::Mat>& images,
                        std::vector<std::string>& image_names,
                        bool names_only)
{
    std::cout << "========================== Load Images ==========================" << std::endl;

    std::string dir_name = std::string(dir);
    std::vector<std::string> files_;

    DIR *dp;
    struct dirent *ep;
    dp = opendir(dir);

    if (dp != NULL)
    {
        while (ep = readdir(dp))
        {
            if (ep->d_name[0] != '.')
                files_.push_back(ep->d_name);
        }

        (void) closedir (dp);
    }
    else
    {
        std::cerr << ("Couldn't open the directory");
        return;
    }

    for (unsigned int i = 0; i < files_.size(); i++)
    {
        if (files_[i][0] == '.' || !(hasEndingLower(files_[i],"jpg")|| hasEndingLower(files_[i],"png")))
        {
            continue;
        }

        if (!names_only)
        {
            //std::cout << std::string(dir_name).append("/").append(files_[i]) << std::endl;
            cv::Mat m_ = cv::imread(std::string(dir_name).append("/").append(files_[i]));
            images.push_back(m_);
        }
        image_names.push_back(files_[i]);

        std::cout << ".";

    }

    std::sort(image_names.begin(), image_names.end());
    std::cout << "  load " <<  image_names.size() << " images" << std::endl;
}

void convert_image_to_8U3(const cv::Mat img_in,
                          cv::Mat& img_bgr,
                          cv::Mat& img_gray)
{
    if (img_in.type() == CV_8UC1)
    {
        cvtColor(img_in, img_bgr, CV_GRAY2BGR);
    }
    else if (img_in.type() == CV_32FC3 ||
             img_in.type() == CV_64FC3)
    {
        img_in.convertTo(img_bgr, CV_8UC3, 255.0);
    }
    else
    {
        //std::cout << "hullala" << std::endl;
        //img_in.copyTo(img_bgr);
        img_bgr = img_in.clone();
    }

    cvtColor(img_bgr, img_gray, CV_BGR2GRAY);
}

void convert_imgs_to_8U3(const std::vector<cv::Mat>& imgs_vec_in,
                         std::vector<cv::Mat>& imgs_bgr_vec,
                         std::vector<cv::Mat>& imgs_gray_vec)
{
    //std::vector<cv::Mat_<cv::Vec3b>> imgs_vec_bgr;
    // CV_8UC1:0  CV_8UC3:16  CV_32FC1:5  CV_32FC3:21   CV_64FC1:6   CV_64FC3:22
    // ensure images are CV_8UC3
    for (unsigned int i = 0; i < imgs_vec_in.size(); i++)
    {
        imgs_bgr_vec.push_back(cv::Mat_<cv::Vec3b>());
        imgs_gray_vec.push_back(cv::Mat());

        if (!imgs_vec_in[i].empty())
        {
            convert_image_to_8U3(imgs_vec_in[i],
                                 imgs_bgr_vec[i],
                                 imgs_gray_vec[i]);
        }

        /*
        cv::imshow("GRAY2BGR", imgs_bgr_vec[i]);
        cv::imshow("BGR2GRAY", imgs_gray_vec[i]);
        char key = cv::waitKey(0);
        cv::destroyWindow("GRAY2BGR");
        cv::destroyWindow("BGR2GRAY");
         */
    }
}

void keypoints_to_point2ds(const std::vector<cv::KeyPoint>& keypoints,
                           std::vector<cv::Point2d>& point2ds_vec)
{
    point2ds_vec.clear();
    for (auto i = 0; i < keypoints.size(); i++)
        point2ds_vec.push_back(keypoints[i].pt);
}

void get_aligned_keypoints(const std::vector<cv::KeyPoint>& kp_i_vec,
                           const std::vector<cv::KeyPoint>& kp_j_vec,
                           const std::vector<cv::DMatch>& matches,
                           std::vector<cv::KeyPoint>& aligned_kp_i_vec,
                           std::vector<cv::KeyPoint>& aligned_kp_j_vec)
{
    aligned_kp_i_vec.clear();
    aligned_kp_j_vec.clear();
    for (auto i = 0; i < matches.size(); i++)
    {
        aligned_kp_i_vec.push_back(kp_i_vec[matches[i].queryIdx]);
        aligned_kp_j_vec.push_back(kp_j_vec[matches[i].trainIdx]);
    }
}

void get_aligned_point2ds(const std::vector<cv::KeyPoint>& kp_i_vec,
                          const std::vector<cv::KeyPoint>& kp_j_vec,
                          const std::vector<cv::DMatch>& matches,
                          std::vector<cv::Point2d>& point2ds_i_vec,
                          std::vector<cv::Point2d>& point2ds_j_vec)
{
    point2ds_i_vec.clear();
    point2ds_j_vec.clear();
    for (auto i = 0; i < matches.size(); i++)
    {
        point2ds_i_vec.push_back(kp_i_vec[matches[i].queryIdx].pt);
        point2ds_j_vec.push_back(kp_j_vec[matches[i].trainIdx].pt);
    }
}

std::vector<cv::Point3d> cloudpoints_to_point3ds(const std::vector<CloudPoint>& cloud)
{
    std::vector<cv::Point3d> out;
    for (auto i=0; i<cloud.size(); i++)
    {
        out.push_back(cloud[i].p3d);
    }
    return out;
}
