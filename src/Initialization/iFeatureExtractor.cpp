//
// Created by Yin Rochelle on 13/06/2017.
//

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "iFeatureExtractor.h"


iFeatureExtractor::iFeatureExtractor(eDetectorType type)
{
    switch (type)
    {
        case eDetectorType::AKAZE:
            mpDetector = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 500);
            break;

        case eDetectorType::ORB:
        default:
            mpDetector = cv::ORB::create(500);
            break;
    }
}

void iFeatureExtractor::ExtractFeatures(const cv::Mat& image,
                                       cv::Mat& descriptors,
                                       std::vector<cv::KeyPoint>& keypoints)
{
    keypoints.clear();
    mpDetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
//  mpDetector->detect(image, keypoints);
//  mpDetector->compute(image, keypoints, descriptors);
//  std::cout << keypoints.size() << std::endl;
}

void iFeatureExtractor::ExtractFeaturesBatch(const std::vector<cv::Mat>& imagesVec,
                                            std::vector<cv::Mat>& descriptorsVec,
                                            std::vector<std::vector<cv::KeyPoint>>& keypointsVec)
{
    std::cout << "\n-------------- extract feature points for all images ------------" << std::endl;

    mpDetector->detect(imagesVec, keypointsVec);
    mpDetector->compute(imagesVec, keypointsVec, descriptorsVec);

    std::cout << "------------------------------ done -----------------------------\n" << std::endl;

    // DrawKeyPointsBatch(imagesVec, keypointsVec);
}

void iFeatureExtractor::DrawKeyPoints(const std::string name, const cv::Mat& image,
                                     const std::vector<cv::KeyPoint>& keypoints)
{
    cv::Mat outImage;
    cv::drawKeypoints(image, keypoints, outImage);
    cv::imshow(name, outImage);
    cv::waitKey(200);
    cv::destroyWindow(name);
}

void iFeatureExtractor::DrawKeyPointsBatch(const std::vector<cv::Mat>& imagesVec,
                                          const std::vector<std::vector<cv::KeyPoint>>& keypointsVec)
{
    for (int i = 0; i < imagesVec.size(); i++)
    {
        std::cout << "draw_" << i << ": " << keypointsVec[i].size() << std::endl;
        DrawKeyPoints("KeyPoints_" + std::to_string(i), imagesVec[i], keypointsVec[i]);
    }
}
