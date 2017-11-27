//
// Created by Yin Rochelle on 11/21/17.
//

#ifndef DSO_GEOMETRIC_CALIBRATION_H
#define DSO_GEOMETRIC_CALIBRATION_H


#include <string>

#include <opencv2/core.hpp>

#include "util/NumType.h"
#include "util/ImageAndExposure.h"
#include "util/MinimalImage.h"


namespace dso {

class GeometricCalibration
{
public:
    GeometricCalibration(const std::string& calibration_file,
                         const std::string& opencv_yaml = "");
    ~GeometricCalibration();

    inline const Mat33 GetOptimalK() const { return optimal_K_; }
    inline const Eigen::Vector2i GetOutputResolution() const { return Eigen::Vector2i(width_, height_); }
    inline const VecX GetOriginalParameters() const { return original_parameters_; }
    inline const Eigen::Vector2i GetInputResolution() { return Eigen::Vector2i(original_width_, original_height_); }
    inline bool Valid() { return valid_; }

    ImageAndExposure* Undistort(const ImageAndExposure* image) const;
    ImageAndExposure* UndistortOpencv(const ImageAndExposure* image) const;


protected:
    enum DistortionModel
    {
        kFov = 0,
        kRadTan,
        kEquidistant,
        kPinhole,
        kKB,
        kUnknown,
    };


private:
    enum RectifyOption
    {
        CROP = -1, FULL = -2, NONE = -3,
    };

    void LoadOpencvCalibration(const std::string& opencv_yaml);
    void ReadConfiguration(const std::string& calibration_file,
                           std::string& intrinsics,
                           std::string& input_resolution,
                           std::string& crop_option,
                           std::string& output_resolution);
    void ResolveDistortionModel(const std::string& intrinsics);
    void ResolveResolution(const std::string& resolution, bool input);
    void ResolveRectifyOption(const std::string& crop_info);
    void AdjustIfRelativeCalibrated();
    void ApplyRectification(const RectifyOption& rectify);
    void CalculateOptimalKIfCrop();
    void CalculateRectifyMap();

    void DistortCoordinates(float* in_x, float* in_y, float* out_x, float* out_y, int n);
    void DistortCoordinatesFov(float* in_x, float* in_y, float* out_x, float* out_y, int n);
    void DistortCoordinatesPinhole(float* in_x, float* in_y, float* out_x, float* out_y, int n);
    void DistortCoordinatesRadTan(float* in_x, float* in_y, float* out_x, float* out_y, int n);
    void DistortCoordinatesEquidistant(float* in_x, float* in_y, float* out_x, float* out_y, int n);
    void DistortCoordinatesKB(float* in_x, float* in_y, float* out_x, float* out_y, int n);

    void ApplyBlurNoise(float* image) const;


    bool valid_;
    bool passthrough_;

    DistortionModel model_;

    int width_, height_;
    int original_width_, original_height_;

    Mat33 optimal_K_;
    VecX original_parameters_;

    bool opencv_calib_enabled_;
    cv::Mat original_K_;
    cv::Mat original_distort_coeffs_;
    cv::Mat map1_;
    cv::Mat map2_;

    float* remap_x_;
    float* remap_y_;
};

}

#endif //DSO_GEOMETRIC_CALIBRATION_H
