//
// Created by Yin Rochelle on 11/21/17.
//
#include <iostream>
#include <fstream>
#include <iomanip>
#include <opencv2/imgproc.hpp>
#include <opencv2/ccalib.hpp>

#include "util/settings.h"
#include "util/globalFuncs.h"
#include "geometric_calibration.h"


namespace dso {

GeometricCalibration::GeometricCalibration(const std::string& calibration_file,
                                           const std::string& opencv_yaml)
: valid_(true), passthrough_(false), opencv_calib_enabled_(false),
  remap_x_(0), remap_y_(0)
{
    LoadOpencvCalibration(opencv_yaml);

    // read parameters
    std::string l1, l2, l3, l4;
    ReadConfiguration(calibration_file, l1, l2, l3, l4);

    ResolveResolution(l2, true);
    ResolveResolution(l4, false);
    ResolveDistortionModel(l1);
    ResolveRectifyOption(l3);
}

GeometricCalibration::~GeometricCalibration()
{
    if (remap_x_ != 0) delete[] remap_x_;
    if (remap_y_ != 0) delete[] remap_y_;
}


void GeometricCalibration::LoadOpencvCalibration(const std::string& opencv_calib_file)
{
    if (opencv_calib_file == "")
    {
        std::cout << "None opencv calib yaml, will disable opencv undistortion!" << std::endl;
        opencv_calib_enabled_ = false;
        return;
    }

    cv::FileStorage fs;
    fs.open(opencv_calib_file, cv::FileStorage::READ);

    if (!fs.isOpened())
    {
        std::cerr << "Fail to open " << opencv_calib_file << std::endl;
        opencv_calib_enabled_ = false;
        return;
    }

    fs["M1"] >> original_K_;
    fs["D1"] >> original_distort_coeffs_;

//    DistCoeffs.at<float>(7) = 0.0;

    std::cout << "M1:" << std::endl << std::setprecision(8) << original_K_ << std::endl;
    std::cout << "D1:" << std::endl << std::setprecision(8) << original_distort_coeffs_ << std::endl;
    opencv_calib_enabled_ = true;
}


ImageAndExposure* GeometricCalibration::Undistort(const ImageAndExposure* input_image) const
{
    if (input_image->w != original_width_ || input_image->h != original_height_)
    {
        printf("GeometricCalibration::Undistort: wrong image size (%d %d instead of %d %d) \n",
               input_image->w, input_image->h, original_width_, original_height_);
        return 0;
    }

    if (!valid_) return 0;

    ImageAndExposure* result = new ImageAndExposure(width_, height_, input_image->timestamp);

    if (!passthrough_)
    {
        float* out_data = result->image;
        float* in_data = input_image->image;

        float* noiseMapX = 0;
        float* noiseMapY = 0;
        if (benchmark_varNoise > 0)
        {
            int numnoise = (benchmark_noiseGridsize + 8) * (benchmark_noiseGridsize + 8);
            noiseMapX = new float[numnoise];
            noiseMapY = new float[numnoise];
            memset(noiseMapX,0,sizeof(float) * numnoise);
            memset(noiseMapY,0,sizeof(float) * numnoise);

            for (int i = 0; i < numnoise; i++)
            {
                noiseMapX[i] = 2 * benchmark_varNoise * (rand()/(float)RAND_MAX - 0.5f);
                noiseMapY[i] = 2 * benchmark_varNoise * (rand()/(float)RAND_MAX - 0.5f);
            }
        }


        for(int idx = width_ * height_ - 1; idx >= 0; idx--)
        {
            // get interp. values
            float xx = remap_x_[idx];
            float yy = remap_y_[idx];

            if (benchmark_varNoise > 0)
            {
                float deltax = getInterpolatedElement11BiCub(noiseMapX,
                                                             4 + (xx / (float)original_width_) * benchmark_noiseGridsize,
                                                             4 + (yy / (float)original_height_) * benchmark_noiseGridsize,
                                                             benchmark_noiseGridsize + 8);
                float deltay = getInterpolatedElement11BiCub(noiseMapY,
                                                             4 + (xx / (float)original_width_) * benchmark_noiseGridsize,
                                                             4 + (yy / (float)original_height_) * benchmark_noiseGridsize,
                                                             benchmark_noiseGridsize + 8);
                float x = idx % width_ + deltax;
                float y = idx / width_ + deltay;
                if (x < 0.01) x = 0.01;
                if (y < 0.01) y = 0.01;
                if (x > width_ - 1.01) x = width_ - 1.01;
                if (y > height_ - 1.01) y = height_ - 1.01;

                xx = getInterpolatedElement(remap_x_, x, y, width_);
                yy = getInterpolatedElement(remap_y_, x, y, height_);
            }

            if (xx < 0)
                out_data[idx] = 0;
            else
            {
                // get integer and rational parts
                int xxi = xx;
                int yyi = yy;
                xx -= xxi;
                yy -= yyi;
                float xxyy = xx * yy;

                // get array base pointer
                const float* src = in_data + xxi + yyi * original_width_;

                // interpolate (bilinear)
                out_data[idx] =  xxyy * src[1 + original_width_]
                                 + (yy - xxyy) * src[original_width_]
                                 + (xx - xxyy) * src[1]
                                 + (1 - xx - yy + xxyy) * src[0];
            }
        }

        if (benchmark_varNoise > 0)
        {
            delete[] noiseMapX;
            delete[] noiseMapY;
        }
    }
    else
    {
        memcpy(result->image, input_image->image, sizeof(float) * width_ * height_);
    }

    ApplyBlurNoise(result->image);

    return result;
}

ImageAndExposure* GeometricCalibration::UndistortOpencv(const ImageAndExposure* input_image) const
{
    if (!valid_) return 0;

    if (!opencv_calib_enabled_)
    {
        printf("No opencv calibration info!\n");
        return 0;
    }

    ImageAndExposure* result = new ImageAndExposure(width_, height_, input_image->timestamp, input_image->exposure_time);

    if (!passthrough_)
    {
        float* out_data = result->image;
        float* in_data = input_image->image;

//        uchar* in_data_uchar = new uchar[width_ * height_];
//        for (auto i = 0; i < width_ * height_; i++)
//        {
//            in_data_uchar[i] = (uchar)(in_data[i]);
//        }
//        cv::Mat input_mat = cv::Mat(height_, width_, CV_8U, in_data_uchar);

        cv::Mat input_mat = cv::Mat(height_, width_, CV_32F, in_data);


//        cv::Mat new_camera_matrix = cv::getOptimalNewCameraMatrix(original_K_, original_distort_coeffs_,
//                                                                  cv::Size_<int>(original_width_, original_height_),
//                                                                  0, cv::Size_<int>(width_, height_), 0, true);
//        printf("\nNew Kamera Matrix:\n");
//        std::cout << new_camera_matrix << "\n\n";

        cv::Mat undistorted_image = cv::Mat(height_, width_, CV_32F);
//        cv::undistort(input_mat, undistorted_image, original_K_, original_distort_coeffs_, new_camera_matrix);

        cv::remap(input_mat, undistorted_image, map1_, map2_, CV_INTER_LINEAR, cv::BorderTypes::BORDER_CONSTANT);


//        memcpy(result->image, (float*)undistorted_image.data, sizeof(float) * width_ * height_);
        for (auto i = 0; i < width_ * height_; i++)
        {
            result->image[i] = undistorted_image.at<float>(i);
        }
    }
    else
    {
        memcpy(result->image, input_image->image, sizeof(float) * width_ * height_);
    }

    ApplyBlurNoise(result->image);

    return result;
}


void GeometricCalibration::ReadConfiguration(const std::string& calibration_file,
                                             std::string& intrinsics,
                                             std::string& input_resolution,
                                             std::string& crop_option,
                                             std::string& output_resolution)
{
    printf("Reading Calibration from file %s", calibration_file.c_str());

    std::ifstream f(calibration_file.c_str());
    if (!f.good())
    {
        f.close();
        printf(" ... not found. Cannot operate without calibration, shutting down.\n");
        valid_ = false;
        return;
    }

    printf(" ... found!\n");
    std::getline(f, intrinsics);
    std::getline(f, input_resolution);
    std::getline(f, crop_option);
    std::getline(f, output_resolution);
    f.close();

    return;
}

void GeometricCalibration::ResolveDistortionModel(const std::string& intrinsics)
{
    float ic[8];

    // for backwards-compatibility: Use RadTan model for 8 parameters.
    if (std::sscanf(intrinsics.c_str(), "%f %f %f %f %f %f %f %f",
                   &ic[0], &ic[1], &ic[2], &ic[3],
                   &ic[4], &ic[5], &ic[6], &ic[7]) == 8)
    {
        printf("found RadTan (OpenCV) camera model, building rectifier.\n");
        model_ = GeometricCalibration::DistortionModel::kRadTan;
    }

    // for backwards-compatibility: Use Pinhole / FoV model for 5 parameter.
    else if (std::sscanf(intrinsics.c_str(), "%f %f %f %f %f",
                        &ic[0], &ic[1], &ic[2], &ic[3], &ic[4]) == 5)
    {
        if (ic[4] == 0)
        {
            printf("found PINHOLE camera model, building rectifier.\n");
            model_ = GeometricCalibration::DistortionModel::kPinhole;
        }
        else
        {
            printf("found ATAN/FOV camera model, building rectifier.\n");
            model_ = GeometricCalibration::DistortionModel::kFov;
        }
    }


    // clean model selection implementation.
    else if (std::sscanf(intrinsics.c_str(), "KannalaBrandt %f %f %f %f %f %f %f %f",
                        &ic[0], &ic[1], &ic[2], &ic[3],
                        &ic[4], &ic[5], &ic[6], &ic[7]) == 8)
    {
        printf("found KannalaBrandt camera model, building rectifier.\n");
        model_ = GeometricCalibration::DistortionModel::kKB;
    }

    else if (std::sscanf(intrinsics.c_str(), "RadTan %f %f %f %f %f %f %f %f",
                        &ic[0], &ic[1], &ic[2], &ic[3],
                        &ic[4], &ic[5], &ic[6], &ic[7]) == 8)
    {
        printf("found RadTan (OpenCV) camera model, building rectifier.\n");
        model_ = GeometricCalibration::DistortionModel::kRadTan;
    }

    else if (std::sscanf(intrinsics.c_str(), "Equidistant %f %f %f %f %f %f %f %f",
                        &ic[0], &ic[1], &ic[2], &ic[3],
                        &ic[4], &ic[5], &ic[6], &ic[7]) == 8)
    {
        printf("found Equidistant camera model, building rectifier.\n");
        model_ = GeometricCalibration::DistortionModel::kEquidistant;
    }

    else if (std::sscanf(intrinsics.c_str(), "FOV %f %f %f %f %f",
                        &ic[0], &ic[1], &ic[2], &ic[3],
                        &ic[4]) == 5)
    {
        printf("found ATAN/FOV camera model, building rectifier.\n");
        model_ = GeometricCalibration::DistortionModel::kFov;
    }

    else if (std::sscanf(intrinsics.c_str(), "Pinhole %f %f %f %f %f",
                        &ic[0], &ic[1], &ic[2], &ic[3],
                        &ic[4]) == 5)
    {
        printf("found PINHOLE camera model, building rectifier.\n");
        model_ = GeometricCalibration::DistortionModel::kPinhole;
    }

    else
    {
        printf("UNKNOWN camera model, please check your configuration!\n");
        model_ = GeometricCalibration::kUnknown;
    }

    valid_ = ( model_ != GeometricCalibration::DistortionModel::kUnknown );
    if (!valid_)
    {
        printf("called with invalid number of parameters.... forgot to implement me?\n");
        return;
    }

    int parameters_num = 0;
    switch (model_)
    {
        case GeometricCalibration::DistortionModel::kFov:
        case GeometricCalibration::DistortionModel::kPinhole:
            parameters_num = 5;
            break;

        case GeometricCalibration::DistortionModel::kRadTan:
        case GeometricCalibration::DistortionModel::kEquidistant:
        case GeometricCalibration::DistortionModel::kKB:
            parameters_num = 8;
            break;

        default:
            break;
    }

    original_parameters_ = VecX(parameters_num);
    printf("In: ");
    for (auto n = 0; n < parameters_num; n++)
    {
        original_parameters_[n] = ic[n];
        printf("%f ", original_parameters_[n]);
    }
    printf("\n");

    AdjustIfRelativeCalibrated();
}

void GeometricCalibration::AdjustIfRelativeCalibrated()
{
    if (!valid_) return;

    if (original_parameters_[2] < 1 && original_parameters_[3] < 1)
    {
        printf("\n\nFound fx=%f, fy=%f, cx=%f, cy=%f.\n I'm assuming this is the \"relative\" calibration file format,"
                       "and will rescale this by image width / height to fx=%f, fy=%f, cx=%f, cy=%f.\n\n",
               original_parameters_[0], original_parameters_[1],
               original_parameters_[2], original_parameters_[3],
               original_parameters_[0] * original_width_,
               original_parameters_[1] * original_height_,
               original_parameters_[2] * original_width_ - 0.5,
               original_parameters_[3] * original_height_ - 0.5);

        // rescale and substract 0.5 offset.
        // the 0.5 is because I'm assuming the calibration is given such that the pixel at (0,0)
        // contains the integral over intensity over [0,0]-[1,1], whereas I assume the pixel (0,0)
        // to contain a sample of the intensity ot [0,0], which is best approximated by the integral over
        // [-0.5,-0.5]-[0.5,0.5]. Thus, the shift by -0.5.
        original_parameters_[0] *= original_width_;
        original_parameters_[1] *= original_height_;
        original_parameters_[2] = original_parameters_[2] * original_width_ - 0.5;
        original_parameters_[3] = original_parameters_[3] * original_height_ - 0.5;
    }
}

void GeometricCalibration::ResolveResolution(const std::string& resolution, bool input)
{
    int w, h;
    valid_ = (std::sscanf(resolution.c_str(), "%d %d", &w, &h) == 2);
    if (!valid_)
    {
        printf("Failed to read resolution (invalid format?)\n");
        return;
    }

    printf("%s resolution: %d x %d\n", input ? "Input" : "Output", w, h);
    if (input)
    {
        original_width_ = w;
        original_height_ = h;
    }
    else
    {
        width_ = w;
        height_ = h;
    }
}

void GeometricCalibration::ResolveRectifyOption(const std::string& rectify)
{
    RectifyOption rectification;

    if (rectify == "crop")
    {
        rectification = RectifyOption::CROP;
        printf("Rectify Crop\n");
    }
    else if (rectify == "full")
    {
        rectification = RectifyOption::FULL;
        printf("Rectify Full\n");
    }
    else if (rectify == "none")
    {
        rectification = RectifyOption::NONE;
        printf("No Rectification\n");
    }
    else
    {
        printf("Out: Failed to Read Output pars... not rectifying.\n");
        valid_ = false;
    }

    if (!valid_) { return; }

    if (benchmarkSetting_width != 0)
    {
        width_ = benchmarkSetting_width;
        if (rectification == RectifyOption::NONE) // crop instead of none, since probably resolution changed.
            rectification = RectifyOption::CROP;
    }
    if (benchmarkSetting_height != 0)
    {
        height_ = benchmarkSetting_height;
        if (rectification == RectifyOption::NONE) // crop instead of none, since probably resolution changed.
            rectification = RectifyOption::CROP;
    }

    if (benchmarkSetting_width != 0  || benchmarkSetting_height != 0)
        printf("Apply rectification, final output resolution: %d x %d\n", width_, height_);

    ApplyRectification(rectification);
}

void GeometricCalibration::ApplyRectification(const RectifyOption& rectification)
{
    if (!valid_) { return; }

    switch (rectification)
    {
        case RectifyOption::CROP:
            CalculateOptimalKIfCrop();
            break;

        case RectifyOption::FULL:
            // todo
            assert(false);
            break;

        case RectifyOption::NONE:
            if (width_ != original_width_ || height_ != original_height_)
            {
                printf("ERROR: rectification mode NONE requires input and output dimensions to match!\n\n");
                valid_ = false;
                return;
            }
            optimal_K_.setIdentity();
            optimal_K_(0,0) = original_parameters_[0];
            optimal_K_(1,1) = original_parameters_[1];
            optimal_K_(0,2) = original_parameters_[2];
            optimal_K_(1,2) = original_parameters_[3];
            passthrough_ = true;
            break;

        default:
            printf("ERROR: Unknown rectification mode!\n\n");
            valid_ = false;
            return;
    }

    if (benchmarkSetting_fxfyfac != 0)
    {
        optimal_K_(0,0) = fmax(benchmarkSetting_fxfyfac, (float)optimal_K_(0,0));
        optimal_K_(1,1) = fmax(benchmarkSetting_fxfyfac, (float)optimal_K_(1,1));
        passthrough_ = false; // cannot pass through when fx / fy have been overwritten.
    }

    CalculateRectifyMap();
}

void GeometricCalibration::CalculateOptimalKIfCrop()
{
    if (opencv_calib_enabled_)
    {
        cv::Mat new_camera_matrix = cv::getOptimalNewCameraMatrix(original_K_, original_distort_coeffs_,
                                                                  cv::Size_<int>(original_width_, original_height_),
                                                                  0, cv::Size_<int>(width_, height_), 0, true);
        optimal_K_.setIdentity();
        optimal_K_(0, 0) = new_camera_matrix.at<double>(0, 0);
        optimal_K_(1, 1) = new_camera_matrix.at<double>(1, 1);
        optimal_K_(0, 2) = new_camera_matrix.at<double>(0, 2);
        optimal_K_(1, 2) = new_camera_matrix.at<double>(1, 2);

        printf("\nGet optimal camera matrix by opencv:\n");
        std::cout << optimal_K_ << "\n\n";
        return;
    }


    printf("finding CROP optimal new model!\n");
    optimal_K_.setIdentity();

    float* remapX = new float[width_ * height_];
    float* remapY = new float[width_ * height_];

    // 1. stretch the center lines as far as possible, to get initial coarse quess.
    float* tgX = new float[100000];
    float* tgY = new float[100000];
    float minX = 0;
    float maxX = 0;
    float minY = 0;
    float maxY = 0;

    for (int x = 0; x < 100000; x++)
    { tgX[x] = (x - 50000.0f) / 10000.0f; tgY[x] = 0; }
    DistortCoordinates(tgX, tgY,tgX, tgY,100000);
    for (int x = 0; x < 100000; x++)
    {
        if(tgX[x] > 0 && tgX[x] < original_width_ - 1)
        {
            if (minX == 0) minX = (x - 50000.0f) / 10000.0f;
            maxX = (x - 50000.0f) / 10000.0f;
        }
    }

    for (int y = 0; y < 100000; y++)
    { tgY[y] = (y - 50000.0f) / 10000.0f; tgX[y] = 0; }
    DistortCoordinates(tgX, tgY,tgX, tgY,100000);
    for (int y = 0; y < 100000; y++)
    {
        if(tgY[y] > 0 && tgY[y] < original_height_ -1)
        {
            if (minY == 0) minY = (y-50000.0f) / 10000.0f;
            maxY = (y - 50000.0f) / 10000.0f;
        }
    }
    delete[] tgX;
    delete[] tgY;

    minX *= 1.01;
    maxX *= 1.01;
    minY *= 1.01;
    maxY *= 1.01;

    printf("initial range: x: %.4f - %.4f; y: %.4f - %.4f!\n", minX, maxX, minY, maxY);

    // 2. while there are invalid pixels at the border: shrink square at the side that has invalid pixels,
    // if several to choose from, shrink the wider dimension.
    bool oobLeft = true, oobRight = true, oobTop = true, oobBottom = true;
    int iteration = 0;
    while (oobLeft || oobRight || oobTop || oobBottom)
    {
        oobLeft = oobRight = oobTop = oobBottom = false;

        for (int y = 0; y < height_; y++)
        {
            remapX[y*2] = minX;
            remapX[y*2+1] = maxX;
            remapY[y*2] = remapY[y*2 + 1] = minY + (maxY - minY) * (float)y / ((float)height_ - 1.0f);
        }
        DistortCoordinates(remapX, remapY, remapX, remapY, 2 * height_);
        for(int y = 0; y < height_; y++)
        {
            if(!(remapX[2*y] > 0 && remapX[2*y] < original_width_ - 1))
                oobLeft = true;
            if(!(remapX[2*y+1] > 0 && remapX[2*y+1] < original_width_ - 1))
                oobRight = true;
        }


        for (int x = 0; x < width_; x++)
        {
            remapY[x*2] = minY;
            remapY[x*2+1] = maxY;
            remapX[x*2] = remapX[x*2+1] = minX + (maxX - minX) * (float)x / ((float)width_ - 1.0f);
        }
        DistortCoordinates(remapX, remapY, remapX, remapY, 2 * width_);
        for (int x = 0; x < width_; x++)
        {
            if (!(remapY[2*x] > 0 && remapY[2*x] < original_height_ - 1))
                oobTop = true;
            if (!(remapY[2*x+1] > 0 && remapY[2*x+1] < original_height_ - 1))
                oobBottom = true;
        }


        if((oobLeft || oobRight) && (oobTop || oobBottom))
        {
            if ((maxX - minX) > (maxY - minY))
                oobBottom = oobTop = false;	// only shrink left/right
            else
                oobLeft = oobRight = false; // only shrink top/bottom
        }

        if (oobLeft) minX *= 0.995;
        if (oobRight) maxX *= 0.995;
        if (oobTop) minY *= 0.995;
        if (oobBottom) maxY *= 0.995;

        iteration++;

        printf("iteration %05d: range: x: %.4f - %.4f; y: %.4f - %.4f!\n", iteration, minX, maxX, minY, maxY);
        if (iteration > 500)
        {
            printf("FAILED TO COMPUTE GOOD CAMERA MATRIX - SOMETHING IS SERIOUSLY WRONG. ABORTING \n");
            valid_ = false;
            break;
        }
    }

    delete[] remapX;
    delete[] remapY;

    if (!valid_) { return; }

    optimal_K_(0,0) = ((float)width_ - 1.0f) / (maxX - minX);
    optimal_K_(1,1) = ((float)height_ -1.0f)/ (maxY - minY);
    optimal_K_(0,2) = -minX * optimal_K_(0,0);
    optimal_K_(1,2) = -minY * optimal_K_(1,1);

    printf("\nRectified Kamera Matrix:\n");
    std::cout << optimal_K_ << "\n\n";
}

void GeometricCalibration::CalculateRectifyMap()
{
    if (!valid_) return;

    remap_x_ = new float[width_ * height_];
    remap_y_ = new float[width_ * height_];

    if (opencv_calib_enabled_)
    {
        cv::initUndistortRectifyMap(original_K_, original_distort_coeffs_,
                                    cv::Mat::eye(cv::Size_<int>(3,3), CV_32FC1),
                                    cv::getOptimalNewCameraMatrix(original_K_, original_distort_coeffs_,
                                                                  cv::Size_<int>(original_width_, original_height_),
                                                                  0, cv::Size_<int>(width_, height_), 0, true),
                                    cv::Size_<int>(width_, height_), CV_32FC1, map1_, map2_);

        for(auto p = 0; p < width_ * height_; p++)
        {
            remap_x_[p] = map1_.at<float>(p);
            remap_y_[p] = map2_.at<float>(p);
        }

        return;
    }


    for (int y = 0; y < height_; y++)
        for (int x = 0;x < width_; x++)
        {
            remap_x_[x + y * width_] = x;
            remap_y_[x + y * width_] = y;
        }

    DistortCoordinates(remap_x_, remap_y_, remap_x_, remap_y_, height_ * width_);

    for (int y = 0; y < height_; y++)
        for (int x = 0;x < width_; x++)
        {
            // make rounding resistant.
            float ix = remap_x_[x + y * width_];
            float iy = remap_y_[x + y * width_];

            if (ix == 0) ix = 0.001;
            if (iy == 0) iy = 0.001;
            if (ix == original_width_ - 1) ix = original_width_ - 1.001;
            if (iy == original_height_ - 1) ix = original_height_ - 1.001;

            if (ix > 0 && iy > 0 && ix < original_width_ - 1 &&  iy < original_width_ - 1)
            {
                remap_x_[x + y * width_] = ix;
                remap_y_[x + y * width_] = iy;
            }
            else
            {
                remap_x_[x + y * width_] = -1;
                remap_y_[x + y * width_] = -1;
            }
        }
}


void GeometricCalibration::DistortCoordinates(float* in_x, float* in_y, float* out_x, float* out_y, int n)
{
    if (!valid_) return;

    switch (model_)
    {
        case DistortionModel::kFov:
            DistortCoordinatesFov(in_x, in_y, out_x, out_y, n);
            break;

        case DistortionModel::kPinhole:
            DistortCoordinatesPinhole(in_x, in_y, out_x, out_y, n);
            break;

        case DistortionModel::kRadTan:
            DistortCoordinatesRadTan(in_x, in_y, out_x, out_y, n);
            break;

        case DistortionModel::kEquidistant:
            DistortCoordinatesEquidistant(in_x, in_y, out_x, out_y, n);
            break;

        case DistortionModel::kKB:
            DistortCoordinatesKB(in_x, in_y, out_x, out_y, n);
            break;

        default:
            break;
    }
}

void GeometricCalibration::DistortCoordinatesFov(float* in_x, float* in_y, float* out_x, float* out_y, int n)
{
    float dist = original_parameters_[4];
    float d2t = 2.0f * tan(dist / 2.0f);

    // current camera parameters
    float fx = original_parameters_[0];
    float fy = original_parameters_[1];
    float cx = original_parameters_[2];
    float cy = original_parameters_[3];

    float ofx = optimal_K_(0,0);
    float ofy = optimal_K_(1,1);
    float ocx = optimal_K_(0,2);
    float ocy = optimal_K_(1,2);

    for(int i=0;i<n;i++)
    {
        float x = in_x[i];
        float y = in_y[i];
        float ix = (x - ocx) / ofx;
        float iy = (y - ocy) / ofy;

        float r = sqrtf(ix*ix + iy*iy);
        float fac = (r==0 || dist==0) ? 1 : atanf(r * d2t)/(dist*r);

        ix = fx*fac*ix+cx;
        iy = fy*fac*iy+cy;

        out_x[i] = ix;
        out_y[i] = iy;
    }
}

void GeometricCalibration::DistortCoordinatesPinhole(float* in_x, float* in_y, float* out_x, float* out_y, int n)
{
    // current camera parameters
    float fx = original_parameters_[0];
    float fy = original_parameters_[1];
    float cx = original_parameters_[2];
    float cy = original_parameters_[3];

    float ofx = optimal_K_(0,0);
    float ofy = optimal_K_(1,1);
    float ocx = optimal_K_(0,2);
    float ocy = optimal_K_(1,2);

    for(int i=0;i<n;i++)
    {
        float x = in_x[i];
        float y = in_y[i];
        float ix = (x - ocx) / ofx;
        float iy = (y - ocy) / ofy;
        ix = fx*ix+cx;
        iy = fy*iy+cy;
        out_x[i] = ix;
        out_y[i] = iy;
    }
}

void GeometricCalibration::DistortCoordinatesRadTan(float* in_x, float* in_y, float* out_x, float* out_y, int n)
{
    // RADTAN
    float fx = original_parameters_[0];
    float fy = original_parameters_[1];
    float cx = original_parameters_[2];
    float cy = original_parameters_[3];
    float k1 = original_parameters_[4];
    float k2 = original_parameters_[5];
    float r1 = original_parameters_[6];
    float r2 = original_parameters_[7];

    float ofx = optimal_K_(0,0);
    float ofy = optimal_K_(1,1);
    float ocx = optimal_K_(0,2);
    float ocy = optimal_K_(1,2);

    for (int i = 0; i < n; i++)
    {
        float x = in_x[i];
        float y = in_y[i];

        // RADTAN
        float ix = (x - ocx) / ofx;
        float iy = (y - ocy) / ofy;
        float mx2_u = ix * ix;
        float my2_u = iy * iy;
        float mxy_u = ix * iy;
        float rho2_u = mx2_u+my2_u;
        float rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;
        float x_dist = ix + ix * rad_dist_u + 2.0 * r1 * mxy_u + r2 * (rho2_u + 2.0 * mx2_u);
        float y_dist = iy + iy * rad_dist_u + 2.0 * r2 * mxy_u + r1 * (rho2_u + 2.0 * my2_u);
        float ox = fx*x_dist+cx;
        float oy = fy*y_dist+cy;

        out_x[i] = ox;
        out_y[i] = oy;
    }

}

void GeometricCalibration::DistortCoordinatesEquidistant(float* in_x, float* in_y, float* out_x, float* out_y, int n)
{
    // EQUI
    float fx = original_parameters_[0];
    float fy = original_parameters_[1];
    float cx = original_parameters_[2];
    float cy = original_parameters_[3];
    float k1 = original_parameters_[4];
    float k2 = original_parameters_[5];
    float k3 = original_parameters_[6];
    float k4 = original_parameters_[7];

    float ofx = optimal_K_(0,0);
    float ofy = optimal_K_(1,1);
    float ocx = optimal_K_(0,2);
    float ocy = optimal_K_(1,2);

    for(int i=0;i<n;i++)
    {
        float x = in_x[i];
        float y = in_y[i];

        // EQUI
        float ix = (x - ocx) / ofx;
        float iy = (y - ocy) / ofy;
        float r = sqrt(ix * ix + iy * iy);
        float theta = atan(r);
        float theta2 = theta * theta;
        float theta4 = theta2 * theta2;
        float theta6 = theta4 * theta2;
        float theta8 = theta4 * theta4;
        float thetad = theta * (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);
        float scaling = (r > 1e-8) ? thetad / r : 1.0;
        float ox = fx*ix*scaling + cx;
        float oy = fy*iy*scaling + cy;

        out_x[i] = ox;
        out_y[i] = oy;
    }
}

void GeometricCalibration::DistortCoordinatesKB(float* in_x, float* in_y, float* out_x, float* out_y, int n)
{
    const float fx = original_parameters_[0];
    const float fy = original_parameters_[1];
    const float cx = original_parameters_[2];
    const float cy = original_parameters_[3];
    const float k0 = original_parameters_[4];
    const float k1 = original_parameters_[5];
    const float k2 = original_parameters_[6];
    const float k3 = original_parameters_[7];

    float ofx = optimal_K_(0,0);
    float ofy = optimal_K_(1,1);
    float ocx = optimal_K_(0,2);
    float ocy = optimal_K_(1,2);

    for(int i=0;i<n;i++)
    {
        float x = in_x[i];
        float y = in_y[i];

        // RADTAN
        float ix = (x - ocx) / ofx;
        float iy = (y - ocy) / ofy;

        const float Xsq_plus_Ysq = ix*ix + iy*iy;
        const float sqrt_Xsq_Ysq = sqrtf(Xsq_plus_Ysq);
        const float theta = atan2f( sqrt_Xsq_Ysq, 1 );
        const float theta2 = theta*theta;
        const float theta3 = theta2*theta;
        const float theta5 = theta3*theta2;
        const float theta7 = theta5*theta2;
        const float theta9 = theta7*theta2;
        const float r = theta + k0*theta3 + k1*theta5 + k2*theta7 + k3*theta9;

        if(sqrt_Xsq_Ysq < 1e-6)
        {
            out_x[i] = fx * ix + cx;
            out_y[i] = fy * iy + cy;
        }
        else
        {
            out_x[i] = (r / sqrt_Xsq_Ysq) * fx * ix + cx;
            out_y[i] = (r / sqrt_Xsq_Ysq) * fy * iy + cy;
        }
    }
}

void GeometricCalibration::ApplyBlurNoise(float* img) const
{
    if (benchmark_varBlurNoise == 0) return;

    int numnoise = (benchmark_noiseGridsize + 8) * (benchmark_noiseGridsize + 8);
    float* noiseMapX = new float[numnoise];
    float* noiseMapY = new float[numnoise];
    float* blutTmp = new float[width_ * height_];

    if (benchmark_varBlurNoise > 0)
    {
        for (int i = 0;i < numnoise; i++)
        {
            noiseMapX[i] =  benchmark_varBlurNoise  * (rand() / (float)RAND_MAX);
            noiseMapY[i] =  benchmark_varBlurNoise  * (rand() / (float)RAND_MAX);
        }
    }


    float gaussMap[1000];
    for (int i = 0; i < 1000; i++)
        gaussMap[i] = expf((float)(-i * i / (100.0 * 100.0)));

    // x-blur.
    for (int y = 0;y < height_; y++)
        for (int x = 0;x < width_; x++)
        {
            float xBlur = getInterpolatedElement11BiCub(noiseMapX,
                                                        4 + (x/(float)width_) * benchmark_noiseGridsize,
                                                        4 + (y/(float)height_) * benchmark_noiseGridsize,
                                                        benchmark_noiseGridsize + 8 );

            if (xBlur < 0.01) xBlur = 0.01;


            int kernelSize = 1 + (int)(1.0f + xBlur * 1.5);
            float sumW = 0;
            float sumCW = 0;
            for (int dx = 0; dx <= kernelSize; dx++)
            {
                int gmid = 100.0f * dx / xBlur + 0.5f;
                if (gmid > 900 ) gmid = 900;
                float gw = gaussMap[gmid];

                if (x + dx > 0 && x + dx < width_)
                {
                    sumW += gw;
                    sumCW += gw * img[x + dx + y * this->width_];
                }

                if (x - dx > 0 && x - dx < width_ && dx != 0)
                {
                    sumW += gw;
                    sumCW += gw * img[x - dx + y * this->width_];
                }
            }

            blutTmp[x + y * this->width_] = sumCW / sumW;
        }

    // y-blur.
    for (int x = 0; x < width_; x++)
        for (int y = 0; y < height_; y++)
        {
            float yBlur = getInterpolatedElement11BiCub(noiseMapY,
                                                        4 + (x / (float)width_) * benchmark_noiseGridsize,
                                                        4 + (y / (float)height_) * benchmark_noiseGridsize,
                                                        benchmark_noiseGridsize + 8 );

            if(yBlur < 0.01) yBlur=0.01;

            int kernelSize = 1 + (int)(0.9f+yBlur*2.5);
            float sumW=0;
            float sumCW=0;
            for(int dy=0; dy <= kernelSize; dy++)
            {
                int gmid = 100.0f * dy / yBlur + 0.5f;
                if (gmid > 900 ) gmid = 900;
                float gw = gaussMap[gmid];

                if(y + dy > 0 && y + dy < height_)
                {
                    sumW += gw;
                    sumCW += gw * blutTmp[x + (y + dy) * this->width_];
                }

                if(y-dy>0 && y-dy<height_ && dy!=0)
                {
                    sumW += gw;
                    sumCW += gw * blutTmp[x + (y - dy) * this->width_];
                }
            }
            img[x + y * this->width_] = sumCW / sumW;
        }


    delete[] noiseMapX;
    delete[] noiseMapY;
}


}