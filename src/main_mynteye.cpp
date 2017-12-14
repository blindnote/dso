//
// Created by Yin Rochelle on 11/17/17.
//

#include <iostream>

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




std::string calib = "";
std::string opencvFile ="";
std::string vignetteFile = "";
std::string gammaFile = "";
float playbackSpeed=0;	// 0 for linearize (play as fast as possible, while sequentializing tracking & mapping). otherwise, factor on timestamps.
std::string cam_name="1";
std::string eye="left";
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


void parseArgument(char* arg)
{
    int option;
    char buf[1000];

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
    if(1==sscanf(arg,"opencv=%s",buf))
    {
        opencvFile = buf;
        printf("loading opencv_calib from %s!\n", opencvFile.c_str());
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

    if(1==sscanf(arg,"nomt=%d",&option))
    {
        if(option==1)
        {
            multiThreading = false;
            printf("NO MultiThreading!\n");
        }
        return;
    }

    if(1==sscanf(arg,"cam=%s",buf))
    {
        cam_name = buf;
        printf("working on camera %s!\n", cam_name.c_str());
        return;
    }
    if(1==sscanf(arg,"eye=%s",buf))
    {
        eye = buf;
        printf("use data from %s eye!\n", eye.c_str());
        return;
    }
    if(1==sscanf(arg,"total=%d",&option))
    {
        total_frames = option;
        return;
    }


    if(1==sscanf(arg,"max=%d",&option))
    {
        total_frames = option;
        printf("Max Frames -> %d !\n", total_frames);
        return;
    }

    printf("could not parse argument \"%s\"!!\n", arg);
}



FullSystem* fullSystem = 0;
Undistort* undistorter = 0;
uint64_t frameID = 0;
int first_track_frame = 0;

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


int main( int argc, char** argv )
{
    for(int i=1; i<argc;i++) parseArgument(argv[i]);

    // hook crtl+C.
    boost::thread exThread = boost::thread(exitThread);


    printf("PHOTOMETRIC MODE WITHOUT CALIBRATION!\n");
    setting_photometricCalibration = 0;
    setting_affineOptModeA = 0; //-1: fix. >=0: optimize (with prior, if > 0).
    setting_affineOptModeB = 0; //-1: fix. >=0: optimize (with prior, if > 0).

//    setting_maxShiftWeightT= 0.04f * (752+480);
//    setting_maxShiftWeightR= 0.0f * (752+480);
//    setting_maxShiftWeightRT= 0.02f * (752+480);


    undistorter = Undistort::getUndistorterForFile(calib, gammaFile, vignetteFile, opencvFile);

    setGlobalCalib(
            (int)undistorter->getSize()[0],
            (int)undistorter->getSize()[1],
            undistorter->getK().cast<float>());


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
        //cam.SetMode(Mode::MODE_CPU);
        cam->Open(init_params);

        if (!cam->IsOpened()) {
            std::cerr << "Error: Open camera failed" << std::endl;
            return 1;
        }

        cout << "\033[1;32mPress ESC/Q on Windows to terminate\033[0m\n";

        cam->ActivateAsyncGrabFeature(true);
//        cam->SetAutoExposureEnabled(false);


        std::thread runthread([&]()
        {

            long long last_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch()).count();

            while (frameID < total_frames)
            {
              ErrorCode code = cam->Grab();
              if (code != ErrorCode::SUCCESS) continue;

              cv::Mat img;
              if ((eye == "left" && cam->RetrieveImage(img, View::VIEW_LEFT_UNRECTIFIED) == ErrorCode::SUCCESS)
            //                  || (eye == "right" && cam->RetrieveImage(img, View::VIEW_RIGHT_UNRECTIFIED) == ErrorCode::SUCCESS)
                      )
              {

                  frameID++;

                  long long curr_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::system_clock::now().time_since_epoch()).count();
                  long long elapse = curr_timestamp - last_timestamp;
                  last_timestamp = curr_timestamp;

                  if (frameID < 10) continue;
                  else if (frameID == 10) {
                      first_track_frame = frameID;
                  }

            //                    cout << "........... elapse:" << elapse << endl;
                  MinimalImageB minImgB(img.cols, img.rows, (unsigned char*)img.data);
            //                  ImageAndExposure* undistImg = undistorter->undistort<unsigned char>(&minImgB, 1.0);
                  ImageAndExposure* undistImg = undistorter->undistort_opencv<unsigned char>(&minImgB, elapse*0.001);

                  fullSystem->addActiveFrame(undistImg, frameID);
                  delete undistImg;


                  if (fullSystem->initFailed)
                  {
                      printf("Init Failed!!\n");

                      if (frameID - first_track_frame >= 60)
                          break;

                      printf("Resetting!!\n");
                      std::vector<IOWrap::Output3DWrapper*> wraps = fullSystem->outputWrapper;
                      delete fullSystem;

                      for(IOWrap::Output3DWrapper* ow : wraps) ow->reset();

                      fullSystem = new FullSystem();
                      fullSystem->linearizeOperation = (playbackSpeed==0);

                      fullSystem->outputWrapper = wraps;
                  }

                  if(fullSystem->isLost)
                  {
                      printf("LOST!!\n");
                      break;
                  }
              }

              printf("....... finish processing frame %d\n", frameID);

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
