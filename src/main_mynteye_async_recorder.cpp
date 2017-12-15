//
// Created by Yin Rochelle on 12/14/17.
//

#include <iostream>
#include <algorithm>
#include <string>
#include <chrono>
#include <deque>
#include <thread>
#include <iomanip>
#include <set>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camera.h"
#include "utility.h"


namespace
{
    template<typename Duration>
    int64_t count(const std::chrono::system_clock::duration &d)
    {
        return std::chrono::duration_cast<Duration>(d).count();
    }

    template<typename Duration>
    std::chrono::system_clock::time_point mycast(const std::chrono::system_clock::time_point &d)
    {
        return std::chrono::time_point_cast<Duration>(d);
    }
}


using namespace std;
using namespace mynteye;
using namespace std::chrono;


struct VIOData
{
    cv::Mat left;
    cv::Mat right;
//    std::vector<IMUData> imudatas;
//    uint32_t timestamp_imu_grab;
    uint32_t timestamp_img_post;
    int64_t timestamp_sys_img_post;
};


Camera cam;
std::chrono::system_clock::time_point time_begin;

class GrabCallbacks
{
public:
    GrabCallbacks(): total_(0)
    {

    }
    ~GrabCallbacks() = default;

    void OnPre()
    {
        //
    }

    void OnPost(cv::Mat &left, cv::Mat &right)
    {
        VIOData data;
        data.timestamp_sys_img_post = count<chrono::milliseconds>(chrono::system_clock::now() - time_begin);
        data.timestamp_img_post = cam.GetTimestamp();
//        cam.RetrieveIMUData(data.imudatas, data.timestamp_imu_grab);

        left.copyTo(data.left);
        right.copyTo(data.right);

        std::unique_lock<std::mutex> lk(mtx_);

        Q.push_back(data);
        q_cv.notify_one();
        total_ += 1;
    }

    std::int64_t GetCount()
    {
        return total_;
    }

    void Consume(VIOData &viodata)
    {
        std::unique_lock<std::mutex> lk(mtx_);

        while (Q.empty()) {
            q_cv.wait(lk);
        }

        VIOData &data = Q.front();
        data.left.copyTo(viodata.left);
        data.right.copyTo(viodata.right);
//        viodata.timestamp_imu_grab = data.timestamp_imu_grab;
        viodata.timestamp_img_post = data.timestamp_img_post;
        viodata.timestamp_sys_img_post = data.timestamp_sys_img_post;
//        viodata.imudatas.swap(data.imudatas);
        Q.pop_front();
    }

private:
    std::mutex mtx_;
    std::deque<VIOData> Q;
    std::condition_variable q_cv;
    int64_t total_;
};


int main( int argc, char** argv )
{
    const char *cam_name = "1";
    std::string mynt_calib="/Users/yinr/ComputerVision/SLAM/workspace/MYNT-EYE-SDK/1.x/1.6/mynteye-1.6-mac-x64-opencv-3.2.0/settings/SN00D1190E0009062D.conf";

    CalibrationParameters *calib_params = new CalibrationParameters;
    calib_params->Load(mynt_calib.c_str());

    InitParameters init_params(cam_name, calib_params);

    try
    {
        time_begin = chrono::system_clock::now();
        cam.Open(init_params);


        if (calib_params)
            delete calib_params;

        if (!cam.IsOpened()) {
            std::cerr << "Error: Open camera failed" << std::endl;
            return 1;
        }

        cam.ActivateAsyncGrabFeature(true);

        GrabCallbacks grab_callbacks;
        cam.SetGrabProcessCallbacks(std::bind(&GrabCallbacks::OnPre, &grab_callbacks),
                                    std::bind(&GrabCallbacks::OnPost, &grab_callbacks,
                                              std::placeholders::_1, std::placeholders::_2));



        for(;;)
        {
            VIOData viodata;

            grab_callbacks.Consume(viodata);

            vector<IMUData> imudatas;
            uint32_t timestamp_imu_grab;
            cam.RetrieveIMUData(imudatas, timestamp_imu_grab);


            cv::imshow("left", viodata.left);
            cv::imshow("right", viodata.right);

//            cout << "timestamp: " << (viodata.timestamp_imu_grab / 10) << " ms, GetTimestamp(): "
            cout << "timestamp: " << (timestamp_imu_grab / 10) << " ms, GetTimestamp(): "
                 << (viodata.timestamp_img_post / 10)
                 << " ms, system timestamp:" << viodata.timestamp_sys_img_post
                 << endl;

            if (imudatas.empty()) continue;

//            for (auto i = 0; i < viodata.imudatas.size(); ++i)
            for (auto i = 0; i < imudatas.size(); ++i)
            {
//                IMUData &imudata = viodata.imudatas[i];
                IMUData &imudata = imudatas[i];

                cout << "  imu[" << i << "] time: " << (imudata.time / 10) << " ms"
                     << fixed << setprecision(13)
                     << ", accel(" << imudata.accel_x << "," << imudata.accel_y << "," << imudata.accel_z << ")"
                     << ", gyro(" << imudata.gyro_x << "," << imudata.gyro_y << "," << imudata.gyro_z << ")"
                     << endl;
            }

            char key = (char) cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') {  // ESC/Q
                break;
            }
        }

        cout << "Close Camera" << endl;
        cam.Close();
    }
    catch(const std::exception & e)
    {
        if (cam.IsOpened()) {
            cam.Close();
        }
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    cv::destroyAllWindows();

    return 0;
}