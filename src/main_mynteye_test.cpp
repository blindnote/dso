//
// Created by Yin Rochelle on 11/12/17.
//
#include <iostream>
#include <chrono>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camera.h"
#include "utility.h"

#include "FullSystem/FullSystem.h"
#include "util/Undistort.h"


namespace {
    template<typename T>
    std::stringstream &Append(std::stringstream &ss, T text,
                              std::ios *fmt = nullptr, bool reset = false) {
        if (reset) {
            ss.str("");
            ss.clear();
        }
        if (fmt) {
            ss.copyfmt(*fmt);
        }
        ss << std::move(text);
        return ss;
    }

    template<typename T>
    std::stringstream &Append(std::stringstream &ss, T text,
                              const std::unique_ptr<std::ios> &fmt, bool reset = false) {
        return Append(ss, std::move(text), fmt.get(), reset);
    }

}


using namespace std;
using namespace mynteye;
using namespace dso;


Undistort* undistorter = nullptr;

cv::Mat CameraIntrinsics;
cv::Mat DistCoeffs;


bool RecvImageAndShow(Camera& cam, const View &view, const Gravity& gravity, const string& winname, const uint64_t cnt)
{
    cv::Mat image;
    if (!cam.RetrieveImage(image, view) == ErrorCode::SUCCESS)
        return false;

    std::stringstream ss;
    Append(ss, image.cols, nullptr, true) << "x" << image.rows << ", " << cnt;
    DrawInfo(image, ss.str(), gravity);
    cv::imshow((winname+"_input").c_str(), image);

    cv::Mat image_undistorted_opencv;
    cv::undistort(image, image_undistorted_opencv, CameraIntrinsics, DistCoeffs);
    cv::imshow((winname+"_opencv").c_str(), image_undistorted_opencv);


    if (undistorter)
    {
        MinimalImageB minImgB(image.cols, image.rows, (unsigned char*)image.data);
        ImageAndExposure* undistImg = undistorter->undistort<unsigned char>(&minImgB, 1.0);

        MinimalImageF minImgF(undistImg->w, undistImg->h, undistImg->image);
        IOWrap::displayImage((winname+"_undistort").c_str(), &minImgF);
    }

    return true;
}


std::string cam_name = "1";
std::string calib = "/Users/yinr/ComputerVision/SLAM/workspace/dso/config/mynteye_default_left.txt";



void parseArgument(char* arg)
{
    int option;
    char buf[1000];

    if(1==sscanf(arg,"calib=%s",buf))
    {
        calib = buf;
        printf("loading calibration from %s!\n", calib.c_str());
        return;
    }

    if(1==sscanf(arg,"cam=%s",buf))
    {
        cam_name = buf;
        printf("working on camera %s!\n", cam_name.c_str());
        return;
    }

    printf("could not parse argument \"%s\"!!\n", arg);
}


std::string mynt_calib="/Users/yinr/ComputerVision/SLAM/workspace/MYNT-EYE-SDK/1.x/1.6/mynteye-1.6-mac-x64-opencv-3.2.0/settings/SN00D1190E0009062D.conf";

void LoadIntrinsics()
{
    cv::FileStorage fs;
    fs.open(mynt_calib, cv::FileStorage::READ);

    if (!fs.isOpened())
    {
        std::cerr << "Fail to open " << mynt_calib << endl;
        return;
    }

    fs["M1"] >> CameraIntrinsics;
    fs["D1"] >> DistCoeffs;

//    DistCoeffs.at<float>(7) = 0.0;

    cout << "M1:" << endl << setprecision(8) << CameraIntrinsics << endl;
    cout << "D1:" << endl << setprecision(8) << DistCoeffs << endl;


    return;
}


// http://blog.csdn.net/u012494876/article/details/53368164
// 平均采样法
//int Fps_AvgSample(float elapse)  // ms
//{
//    static float avg_duration = 0.f;
//    static float alpha = 1.f / 100.f;  // 采样数设置为100
//    static int frame_count = 0;
//
//    ++frame_count;
//
//    if (1 == frame_count)
//    {
//        avg_duration = elapse;
//    }
//    else
//    {
//        avg_duration = avg_duration * (1.0 - alpha) + elapse * alpha;
//    }
//
//    return static_cast<int>(1.f / avg_duration * 1000);
//}

int Fps_FixedDuration()
{
    static int fps = 0;
    static long long last_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    static int frame_count = 0;


    ++frame_count;

    long long curr_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
//    printf("curr:%u ms, last:%u ms, delta:%u ms\n", curr_time, last_time, curr_time - last_time);
    if (curr_time - last_time > 1000)   // 1000ms = 1sec
    {
        fps = frame_count;
        frame_count = 0;
        last_time = curr_time;
    }

    return fps;
}



int main( int argc, char** argv )
{
    for(int i=1; i<argc;i++) parseArgument(argv[i]);

//    LoadIntrinsics();

    undistorter = Undistort::getUndistorterForFile(calib, "", "");
    setGlobalCalib(
            (int)undistorter->getSize()[0],
            (int)undistorter->getSize()[1],
            undistorter->getK().cast<float>());


//    const char *name = (argc >=2 ) ? argv[1] : "0";
    cout << "Open Camera: " << cam_name << endl;

    CalibrationParameters *calib_params = nullptr;
//    if (argc >= 3) {
//        stringstream ss;
//        ss << argv[2];
        calib_params = new CalibrationParameters;
//        calib_params->Load(ss.str());
        calib_params->Load(mynt_calib.c_str());
//    }
    InitParameters init_params(cam_name, calib_params);

    Camera cam;
    //cam.SetMode(Mode::MODE_CPU);
    cam.Open(init_params);

    if (calib_params)
        delete calib_params;

    if (!cam.IsOpened()) {
        std::cerr << "Error: Open camera failed" << std::endl;
        return 1;
    }
    cout << "\033[1;32mPress ESC/Q on Windows to terminate\033[0m\n";

    cam.ActivateAsyncGrabFeature(true);
    cam.SetAutoExposureEnabled(false);

    CalibrationParameters parameters = cam.GetCalibrationParameters();
    cout << "LEFT EYE:" << endl << setprecision(8) << parameters.M1 << endl << parameters.D1 << endl;
    cout << "RIGHT EYE:" << endl << setprecision(8) << parameters.M2 << endl << parameters.D2 << endl;

    cout << "Auto-Exposure: " << (cam.IsAutoExposureEnabled() ? "true" : "false") << endl;
//    cout << "Brightness: "  << cam.GetBrightness() << endl;
//    cout << "Contrast: " << cam.GetContrast() << endl;
//    cout << "Resolution: "  << cam.GetResolution().width << ", " << cam.GetResolution().height << endl;

    std::uint64_t count_left = 0, count_right = 0;

//    double tick_beg = (double)cv::getTickCount();

//    double time_ms = cam.GetTimestamp();

    for(;;) {
        ErrorCode code = cam.Grab();

        if (code != ErrorCode::SUCCESS) continue;

//        if (RecvImageAndShow(cam, View::VIEW_LEFT_UNRECTIFIED, Gravity::TOP_LEFT, "left", count_left)) {
//            count_left++;
//        }

        cv::Mat rect_image;
        if (cam.RetrieveImage(rect_image, View::VIEW_LEFT) == ErrorCode::SUCCESS) {
//            cv::imshow("left_rect", rect_image);

//            int fps = Fps((cam.GetTimestamp() - time_ms) * 0.1);
            int fps = Fps_FixedDuration();
            printf("IMG FPS: left: %d\n", fps);

//            time_ms = cam.GetTimestamp();
        }


//        if (RecvImageAndShow(cam, View::VIEW_RIGHT_UNRECTIFIED, Gravity::TOP_RIGHT, "right", count_right)) {
//            count_right++;
//        }

//        double elapsed = ((double)cv::getTickCount() - tick_beg) / cv::getTickFrequency();
//        printf("IMG FPS: left: %.2f, right: %.2f\n", (count_left / elapsed), (count_right / elapsed));


//        double elapsed = (cam.GetTimestamp() - time_ms) * 0.00001;
//        printf("IMG FPS: left: %.2f, right: %.2f\n", (count_left / elapsed), (count_right / elapsed));


//        cout << "Brightness: "  << cam.GetBrightness() << endl;
//        cout << "Contrast: " << cam.GetContrast() << endl;
//        cout << "DroppedCount_PROC_GRAB:" << cam.GetDroppedCount(PROC_GRAB) << endl;
//        cout << "DroppedCount_PROC_RECTIFY: " << cam.GetDroppedCount(PROC_RECTIFY) << endl;
//        cout << "DroppedCount_PROC_LAST: " << cam.GetDroppedCount(PROC_LAST) << endl;
//        cout << "Timestamp: " << cam.GetTimestamp() << endl;


        char key = (char) cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') {  // ESC/Q
            break;
        }

    }


    cout << "Close Camera" << endl;
    cam.Close();
    cv::destroyAllWindows();

    return 0;
}