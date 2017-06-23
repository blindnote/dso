//
// Created by Yin Rochelle on 13/06/2017.
//

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include "iCommon.h"
#include "iRichFeatureMatcher.h"

const double NN_MATCH_RATIO = 0.8f;

iRichFeatureMatcher::iRichFeatureMatcher(eMatcherType type)
{
    switch (type)
    {
        case eMatcherType::BruteForce:
            //mpMatcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
            //mpMatcher = std::make_shared<cv::BFMatcher>(cv::NORM_HAMMING, true);
            mpMatcher = std::make_shared<cv::BFMatcher>(cv::NORM_HAMMING);
            break;

        case eMatcherType::FlannBased:
        default:
            // mpMatcher = cv::DescriptorMatcher::create("FlannBased");
            mpMatcher = std::make_shared<cv::FlannBasedMatcher>(cv::makePtr<cv::flann::LshIndexParams>(6, 12, 1));
            break;
    }
}


void iRichFeatureMatcher::Match(const cv::Mat& desp_i,
                               const cv::Mat& desp_j,
                               const std::vector<cv::KeyPoint>& kp_i,
                               const std::vector<cv::KeyPoint>& kp_j,
                               std::vector<cv::DMatch>& matches)
{
    matches.clear();

    std::vector<std::vector<cv::DMatch>> knn_matches;
    mpMatcher->knnMatch(desp_i, desp_j, knn_matches, 2);

    //prune the matching using the ratio test
    for(unsigned i = 0; i < knn_matches.size(); i++)
    {
        std::vector<cv::DMatch>& ith_match = knn_matches[i];
        if (ith_match.size() < 1)
        {
            continue;
        }

        if (ith_match.size() < 2)
        {
            matches.push_back(ith_match[0]);
            continue;
        }

        if(knn_matches[i][0].distance < NN_MATCH_RATIO * knn_matches[i][1].distance)
        {
            matches.push_back(knn_matches[i][0]);
        }
    }
}


void iRichFeatureMatcher::Match_Homo(const cv::Mat& desp_i,
                                    const cv::Mat& desp_j,
                                    const std::vector<cv::KeyPoint>& kp_i,
                                    const std::vector<cv::KeyPoint>& kp_j,
                                    std::vector<cv::DMatch>& matches)
{
    matches.clear();

    std::vector<std::vector<cv::DMatch>> knn_matches;
    mpMatcher->knnMatch(desp_i, desp_j, knn_matches, 1);

    for(size_t i = 0; i < knn_matches.size(); i++)
    {
        std::vector<cv::DMatch>& ith_match = knn_matches[i];
        if (ith_match.size() < 1)
        {
            continue;
        }

        if (ith_match.size() < 2)
        {
            matches.push_back(ith_match[0]);
            continue;
        }

        cv::DMatch bestMatch = ith_match[0];
        cv::DMatch betterMatch = ith_match[1];

        if(bestMatch.distance < Knn_Match_Ratio * betterMatch.distance)
        {
            matches.push_back(bestMatch);
        }
    }

    //std::cout << "before refine: " << matches_num << std::endl;
    if (matches.size() < Min_Matches_Num) return;

    std::vector<cv::Point2d> srcPoints, dstPoints;
    // not natural order, but do give better results >_<??
    get_aligned_point2ds(kp_i, kp_j, matches, dstPoints, srcPoints);
    //get_aligned_point2ds(kp_i, kp_j, matches, srcPoints, dstPoints);

    std::vector<uchar> inliersMask(srcPoints.size());
    cv::Mat homography = findHomography(srcPoints, dstPoints, CV_FM_RANSAC,
                                        Reprojection_Threshhold, inliersMask);

    /* not good on pantry data
    double minVal, maxVal;
    cv::minMaxIdx(srcPoints, &minVal, &maxVal);
    // threshold from Snavely07
    cv::Mat homography = findHomography(srcPoints, dstPoints, inliersMask,
                                        CV_FM_RANSAC, 0.004 * maxVal);
    */
    std::vector<cv::DMatch> inliers;
    for (size_t i = 0; i < inliersMask.size(); i++)
    {
        if (inliersMask[i])
            inliers.push_back(matches[i]);
    }
    matches.swap(inliers);
    //std::cout << "after refine: " << matches.size() << std::endl;
}

void iRichFeatureMatcher::FlipMatches(const std::vector<cv::DMatch>& matches,
                                     std::vector<cv::DMatch>& matches_flipped)
{
    for(int i = 0; i < matches.size(); i++)
    {
        matches_flipped.push_back(matches[i]);
        std::swap(matches_flipped.back().queryIdx,
                  matches_flipped.back().trainIdx);
    }
}

void iRichFeatureMatcher::DrawMatches(const std::string name,
                                     const cv::Mat& img_i,
                                     const cv::Mat& img_j,
                                     const std::vector<cv::KeyPoint>& kp_i,
                                     const std::vector<cv::KeyPoint>& kp_j,
                                     const std::vector<cv::DMatch>& matches,
                                     float scale)
{
    cv::Mat matches_to_draw;
    cv::drawMatches(img_i, kp_i, img_j, kp_j, matches, matches_to_draw,
                    cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    cv::Mat scaled_image;
    cv::resize(matches_to_draw, scaled_image, cv::Size(), scale, scale);
    cv::imshow( name, scaled_image );
    cv::waitKey(0);
    cv::destroyWindow(name);
}
