//
// Created by Yin Rochelle on 11/17/17.
//
#include <iomanip>
#include <iostream>
#include <deque>
#include <thread>
#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <boost/thread/thread.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camera.h"
#include "utility.h"

#include "util/settings.h"
#include "FullSystem/FullSystem.h"
#include "util/Undistort.h"
#include "IOWrapper/ImageRW.h"
#include "IOWrapper/Pangolin/PangolinDSOViewer.h"
#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"



using namespace std;
using namespace mynteye;
using namespace dso;


int FpsStatus()
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




class GrabCallbacks
{
public:
    GrabCallbacks() {}
    ~GrabCallbacks() = default;

    void OnPost(cv::Mat &left, cv::Mat &right) {
        std::lock_guard<std::mutex> lk(mtx_);

        Q.push_back(left);

//        int fps = FpsStatus();
//        printf("fps:%d\n", fps);
    }

//    std::uint64_t GetCount() {
//        std::lock_guard<std::mutex> lk(mtx_);
//        return Q.size();
//    }

    bool Consume(cv::Mat &image) {
        std::lock_guard<std::mutex> lk(mtx_);

        if (Q.empty()) {
//            printf("false\n");
            return false;
        }

        image = Q.front();
        Q.pop_front();
//        printf("Q:%u\n", Q.size());
        return true;
    }

private:
    std::mutex mtx_;
    std::deque<cv::Mat> Q;
};



Camera* cam = nullptr;
void my_exit_handler(int s)
{
    printf("Caught signal %d\n",s);
    exit(1);
}

void exitThread()
{
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = my_exit_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

    if (cam)
    {
        cam->Close();
        delete cam;
        cam = nullptr;
    }
    while(true) pause();
}



std::string cam_name="1";
std::string calib = "";
std::string vignetteFile = "";
std::string gammaFile = "";
float playbackSpeed=0;	// 0 for linearize (play as fast as possible, while sequentializing tracking & mapping). otherwise, factor on timestamps.
int total_frames = 100;


void settingsDefault(int preset)
{
    printf("\n=============== PRESET Settings: ===============\n");
    if(preset == 0 || preset == 1)
    {
        printf("DEFAULT settings:\n"
                       "- %s real-time enforcing\n"
                       "- 2000 active points\n"
                       "- 5-7 active frames\n"
                       "- 1-6 LM iteration each KF\n"
                       "- original image resolution\n", preset==0 ? "no " : "1x");

        playbackSpeed = (preset==0 ? 0 : 1);
        //playbackSpeed = 1;
        setting_desiredImmatureDensity = 1500;
        setting_desiredPointDensity = 2000;
        setting_minFrames = 5;
        setting_maxFrames = 7;
        setting_maxOptIterations=6;
        setting_minOptIterations=1;

        setting_logStuff = false;
    }

    if(preset == 2 || preset == 3)
    {
        printf("FAST settings:\n"
                       "- %s real-time enforcing\n"
                       "- 800 active points\n"
                       "- 4-6 active frames\n"
                       "- 1-4 LM iteration each KF\n"
                       "- 424 x 320 image resolution\n", preset==0 ? "no " : "5x");

        playbackSpeed = (preset==2 ? 0 : 5);
        setting_desiredImmatureDensity = 600;
        setting_desiredPointDensity = 800;
        setting_minFrames = 4;
        setting_maxFrames = 6;
        setting_maxOptIterations=4;
        setting_minOptIterations=1;

//        benchmarkSetting_width = 424;
//        benchmarkSetting_height = 320;

        setting_logStuff = false;
    }

    printf("==============================================\n");
}


void parse_arguments(char *arg)
{
    int option;
    char buf[1000];

    if(1==sscanf(arg,"cam=%s",buf))
    {
        cam_name = buf;
        printf("working on camera %s!\n", cam_name.c_str());
        return;
    }

    if(1==sscanf(arg,"preset=%d",&option))
    {
        settingsDefault(option);
        return;
    }
    if(1==sscanf(arg,"nogui=%d",&option))
    {
        if(option==1)
        {
            disableAllDisplay = true;
            printf("NO GUI!\n");
        }
        return;
    }
    if(1==sscanf(arg,"calib=%s",buf))
    {
        calib = buf;
        printf("loading calibration from %s!\n", calib.c_str());
        return;
    }
    if(1==sscanf(arg,"vignette=%s",buf))
    {
        vignetteFile = buf;
        printf("loading vignette from %s!\n", vignetteFile.c_str());
        return;
    }

    if(1==sscanf(arg,"gamma=%s",buf))
    {
        gammaFile = buf;
        printf("loading gammaCalib from %s!\n", gammaFile.c_str());
        return;
    }

    if(1==sscanf(arg,"nogui=%d",&option))
    {
        if(option==1)
        {
            disableAllDisplay = true;
            printf("NO GUI!\n");
        }
        return;
    }

    printf("could not parse argument \"%s\"!!\n", arg);
}




FullSystem* fullSystem = 0;
Undistort* undistorter = 0;
uint64_t frameID = 0;

std::string mynt_calib="/Users/yinr/ComputerVision/SLAM/workspace/MYNT-EYE-SDK/1.x/1.6/mynteye-1.6-mac-x64-opencv-3.2.0/settings/SN00D1190E0009062D.conf";

void LoadIntrinsics(Undistort* undistorter_ptr)
{
    cv::FileStorage fs;
    fs.open(mynt_calib, cv::FileStorage::READ);

    if (!fs.isOpened())
    {
        std::cerr << "Fail to open " << mynt_calib << endl;
        return;
    }

    fs["M1"] >> undistorter_ptr->K_OpenCV;
    fs["D1"] >> undistorter_ptr->DistCoeffs_OpenCV;

//    DistCoeffs.at<float>(7) = 0.0;

    cout << "M1:" << endl << setprecision(8) << undistorter_ptr->K_OpenCV << endl;
    cout << "D1:" << endl << setprecision(8) << undistorter_ptr->DistCoeffs_OpenCV << endl;
    return;
}





int main(int argc, char** argv)
{
    for(auto i = 1; i < argc; i++)
        parse_arguments(argv[i]);


    // hook crtl+C.
    boost::thread exThread = boost::thread(exitThread);


    printf("PHOTOMETRIC MODE WITHOUT CALIBRATION!\n");
    setting_photometricCalibration = 0;
    setting_affineOptModeA = 0; //-1: fix. >=0: optimize (with prior, if > 0).
    setting_affineOptModeB = 0; //-1: fix. >=0: optimize (with prior, if > 0).


    undistorter = Undistort::getUndistorterForFile(calib, gammaFile, vignetteFile);

    setGlobalCalib(
            (int)undistorter->getSize()[0],
            (int)undistorter->getSize()[1],
            undistorter->getK().cast<float>());

    LoadIntrinsics(undistorter);


    fullSystem = new FullSystem();
    fullSystem->linearizeOperation = (playbackSpeed == 0);


    IOWrap::PangolinDSOViewer* viewer = 0;
    if(!disableAllDisplay)
    {
        viewer = new IOWrap::PangolinDSOViewer(
                (int)undistorter->getSize()[0],
                (int)undistorter->getSize()[1], false);
        fullSystem->outputWrapper.push_back(viewer);
    }



    try
    {
        const char *name = cam_name.c_str();
        cout << "Open Camera: " << name << endl;
        InitParameters init_params(name);

        cam = new Camera();
        cam->Open(init_params);

        if (!cam->IsOpened()) {
            std::cerr << "Error: Open camera failed" << std::endl;
            return 1;
        }

        cout << "\033[1;32mPress ESC/Q on Windows to terminate\033[0m\n";

        cam->ActivateAsyncGrabFeature(true);

        GrabCallbacks grab_callbacks;
        cam->SetGrabProcessCallbacks(nullptr, std::bind(&GrabCallbacks::OnPost, &grab_callbacks, std::placeholders::_1, std::placeholders::_2));


        std::thread runthread([&]() {

            long long last_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();

            for(;;)
            {
                cv::Mat image;
                if (!grab_callbacks.Consume(image))
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(30));
                    continue;
                }


                frameID++;
//                    cv::imshow("image", image);
                long long curr_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count();
                long long elapse = curr_timestamp - last_timestamp;
                last_timestamp = curr_timestamp;


                if (frameID < 20) continue;


                struct timeval tv_start; gettimeofday(&tv_start, NULL);

                MinimalImageB minImgB(image.cols, image.rows, (unsigned char*)image.data);
//                  ImageAndExposure* undistImg = undistorter->undistort<unsigned char>(&minImgB, 1.0);
                ImageAndExposure* undistImg = undistorter->undistort_opencv<unsigned char>(&minImgB, elapse*0.001);

                fullSystem->addActiveFrame(undistImg, frameID);
                delete undistImg;

                struct timeval tv_end; gettimeofday(&tv_end, NULL);
                printf("%u us\n", (tv_end.tv_usec - tv_start.tv_usec));


                char key = (char) cv::waitKey(1);
                if (key == 27 || key == 'q' || key == 'Q') {  // ESC/Q
                    break;
                }
            }
        });


        if(viewer != 0)
            viewer->run();

        runthread.join();

    }
    catch(const std::exception & e)
    {
        if (cam) {
            cam->Close();
        }
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }


    for(IOWrap::Output3DWrapper* ow : fullSystem->outputWrapper)
    {
        ow->join();
        delete ow;
    }

    delete undistorter;
    delete fullSystem;

    cout << "Close Camera" << endl;
    cam->Close();
//    cv::destroyAllWindows();
    delete cam;
    cam = nullptr;

    return 0;
}