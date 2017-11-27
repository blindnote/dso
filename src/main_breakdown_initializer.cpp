//
// Created by Yin Rochelle on 11/21/17.
//

#include <iostream>

#include <opencv2/highgui.hpp>
#include "util/globalCalib.h"
#include "utility/images_reader.h"
#include "utility/cmd_args_parser.h"
#include "calibration/geometric_calibration.h"
#include "calibration/photometric_calibration.h"
#include "IOWrapper/ImageDisplay.h"

using namespace std;
using namespace std;


int main( int argc, char** argv )
{
    for(auto i = 1; i < argc; i++)
        ParseArgument(argv[i]);

    setting_photometricCalibration = 0;
    setting_affineOptModeA = -1;
    setting_affineOptModeB = -1;

    GeometricCalibration geo_calib(calib_file, opencv_yaml);
    if (!geo_calib.Valid()) return 0;


    Eigen::Vector2i input_resolution = geo_calib.GetInputResolution();
    int w = input_resolution[0];
    int h = input_resolution[1];

    PhotometricCalibration photo_calib(w, h, gamma_file, vignette_file);

    Eigen::Vector2i output_resolution = geo_calib.GetInputResolution();
    int output_w = output_resolution[0];
    int output_h = output_resolution[1];
    setGlobalCalib(output_w, output_h, geo_calib.GetOptimalK().cast<float>());


    ImagesReader reader(images_path, w, h);
    for (auto id = start_id; id < end_id; id++)
    {
        MinimalImageB* raw_image = reader.GetRawImage(id);
        IOWrap::displayImage("Raw", raw_image, true);

        ImageAndExposure* corrected_image = new ImageAndExposure(w, h, reader.GetTimestamp(id));
        photo_calib.ApplyCorrection<unsigned char>(corrected_image, raw_image->data, reader.GetExposure(id));
        if (photo_calib.GammaValid())
        {
//            char ph_calib_buf[20];
//            snprintf(ph_calib_buf, 20, "Photometric_%d", id);
            IOWrap::displayImage("Photometric", corrected_image, true);
        }


        ImageAndExposure* undistort_image = geo_calib.Undistort(corrected_image);
        if (undistort_image == 0)
        {
            printf("Fail to undistort %d!", id);
        }
        else
        {
//            char geo_calib_buf[20];
//            snprintf(geo_calib_buf, 20, "Undistort_%d", id);
            IOWrap::displayImage("Undistort", undistort_image, true);
        }


        ImageAndExposure* undistort_opencv_img = geo_calib.UndistortOpencv(corrected_image);
        if (undistort_opencv_img == 0)
        {
            printf("Fail to undistort opencv %d!\n", id);
        }
        else
        {
//            char geo_opecv_calib_buf[20];
//            snprintf(geo_opecv_calib_buf, 20, "UndistortOpencv_%d", id);
            IOWrap::displayImage("UndistortOpencv", undistort_opencv_img, true);
        }


        cv::waitKey(0);


        delete corrected_image;
        delete raw_image;
    }

    cv::destroyAllWindows();

    return 0;
}