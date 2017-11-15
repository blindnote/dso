//
// Created by Yin Rochelle on 23/06/2017.
//

#include <cstdio>
#include <iostream>
#include <vector>
#include <sstream>
#include <thread>
#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <librealsense/rs.hpp>

#include <boost/thread/thread.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

#include "util/settings.h"
#include "FullSystem/FullSystem.h"
#include "util/Undistort.h"
#include "IOWrapper/ImageRW.h"
#include "IOWrapper/Pangolin/PangolinDSOViewer.h"
#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"


rs::device* dev = 0;

std::string calib = "";
std::string vignetteFile = "";
std::string gammaFile = "";
bool useSampleOutput=false;

int ignoreNumOfFrames = 10;
int sfmInitNumOfFrames = 30;
int currentNumberOfFrames = 0;
int totalNumberOfFrames = 1000;


using namespace dso;


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

    if (dev)
    {
        dev->stop();
    }
    while(true) pause();
}


void parseArgument(char* arg)
{
    int option;
    char buf[1000];

    if(1==sscanf(arg,"sampleoutput=%d",&option))
    {
        if(option==1)
        {
            useSampleOutput = true;
            printf("USING SAMPLE OUTPUT WRAPPER!\n");
        }
        return;
    }

    if(1==sscanf(arg,"quiet=%d",&option))
    {
        if(option==1)
        {
            setting_debugout_runquiet = true;
            printf("QUIET MODE, I'll shut up!\n");
        }
        return;
    }


    if(1==sscanf(arg,"nolog=%d",&option))
    {
        if(option==1)
        {
            setting_logStuff = false;
            printf("DISABLE LOGGING!\n");
        }
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
    if(1==sscanf(arg,"nomt=%d",&option))
    {
        if(option==1)
        {
            multiThreading = false;
            printf("NO MultiThreading!\n");
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

    printf("could not parse argument \"%s\"!!\n", arg);
}



FullSystem* fullSystem = 0;
Undistort* undistorter = 0;
int frameID = 0;

//void vidCb(const sensor_msgs::ImageConstPtr img)
//{
//    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
//    assert(cv_ptr->image.type() == CV_8U);
//    assert(cv_ptr->image.channels() == 1);
//
//
//    if(setting_fullResetRequested)
//    {
//        std::vector<IOWrap::Output3DWrapper*> wraps = fullSystem->outputWrapper;
//        delete fullSystem;
//        for(IOWrap::Output3DWrapper* ow : wraps) ow->reset();
//        fullSystem = new FullSystem();
//        fullSystem->linearizeOperation=false;
//        fullSystem->outputWrapper = wraps;
//        if(undistorter->photometricUndist != 0)
//            fullSystem->setGammaFunction(undistorter->photometricUndist->getG());
//        setting_fullResetRequested=false;
//    }
//
//    MinimalImageB minImg((int)cv_ptr->image.cols, (int)cv_ptr->image.rows,(unsigned char*)cv_ptr->image.data);
//    ImageAndExposure* undistImg = undistorter->undistort<unsigned char>(&minImg, 1,0, 1.0f);
//    fullSystem->addActiveFrame(undistImg, frameID);
//    frameID++;
//    delete undistImg;
//
//}


struct RgbCallback
{
public:
    RgbCallback(int64_t& lastRgbTime,
        uint8_t* rgbBuffers)
        : lastTimestamp(lastRgbTime),
          buffers(rgbBuffers)
    {
    }

    void operator()(rs::frame frame)
    {
        if (currentNumberOfFrames < 10)
        {
            currentNumberOfFrames++;
            return;
        }

        lastTimestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();

        memcpy(buffers, frame.get_data(), frame.get_width() * frame.get_height() * 3);

        cv::Mat color = cv::Mat(frame.get_height(), frame.get_width(), CV_8UC3, CV_LOAD_IMAGE_COLOR);

        for(int r = 0; r < frame.get_height(); r++)
            for(int c = 0; c < frame.get_width(); c++)
            {
                int index = r * frame.get_width() + c;
                color.at<cv::Vec3b>(r, c)[0] = (uchar) (buffers[index * 3 + 2]);
                color.at<cv::Vec3b>(r, c)[1] = (uchar) (buffers[index * 3 + 1]);
                color.at<cv::Vec3b>(r, c)[2] = (uchar) (buffers[index * 3]);
            }

        cv::Mat m;
        cv::cvtColor(color, m, CV_BGR2GRAY);

        MinimalImageB minImg(m.cols, m.rows, (unsigned char*)m.data);
        ImageAndExposure* undistImg = undistorter->undistort<unsigned char>(&minImg, 1, 0, 1.0f);
  //      std::string pic_name = "/Users/yinr/ComputerVision/SLAM/workspace/dso/build/r200/" + std::to_string(lastTimestamp) + ".jpg";
  //      std::string pic_gray_name = "/Users/yinr/ComputerVision/SLAM/workspace/dso/build/r200/" + std::to_string(lastTimestamp) + "_g.jpg";

        // IOWrap::writeImage(pic_name, &minImg);
    //    cv::imwrite(pic_name, cv::Mat(color.rows, color.cols, CV_8UC3, color.data));
    //    cv::imwrite(pic_gray_name, cv::Mat(m.rows, m.cols, CV_8U, m.data));

        fullSystem->addActiveFrame(undistImg, frameID);
        frameID++;
        delete undistImg;
    }

private:
    int64_t& lastTimestamp;
    uint8_t* buffers;
};




int main( int argc, char** argv )
{
    for(int i=1; i<argc;i++) parseArgument(argv[i]);

    // hook crtl+C.
    boost::thread exThread = boost::thread(exitThread);


    setting_desiredImmatureDensity = 1000;
    setting_desiredPointDensity = 1200;
    setting_minFrames = 5;
    setting_maxFrames = 7;
    setting_maxOptIterations=4;
    setting_minOptIterations=1;
    setting_logStuff = false;
    setting_kfGlobalWeight = 1.3;


    printf("MODE WITH CALIBRATION, but without exposure times!\n");
    setting_photometricCalibration = 2;
    setting_affineOptModeA = 0;
    setting_affineOptModeB = 0;



    undistorter = Undistort::getUndistorterForFile(calib, gammaFile, vignetteFile);

    setGlobalCalib(
            (int)undistorter->getSize()[0],
            (int)undistorter->getSize()[1],
            undistorter->getK().cast<float>());


    fullSystem = new FullSystem();
    fullSystem->linearizeOperation=false;


    IOWrap::PangolinDSOViewer* viewer = 0;
    if(!disableAllDisplay)
    {
        viewer = new IOWrap::PangolinDSOViewer(
                (int)undistorter->getSize()[0],
                (int)undistorter->getSize()[1], false);
        fullSystem->outputWrapper.push_back(viewer);
    }


    if(useSampleOutput)
        fullSystem->outputWrapper.push_back(new IOWrap::SampleOutputWrapper());


    if(undistorter->photometricUndist != 0)
        fullSystem->setGammaFunction(undistorter->photometricUndist->getG());

    //rs::device* dev = 0;

    try
    {
        int raw_width = 640, raw_height = 480;

        int64_t lastRgbTime;
        uint8_t * newImage = (uint8_t *)calloc(raw_width * raw_height * 3, sizeof(uint8_t));
        RgbCallback* rgbCallback = new RgbCallback(lastRgbTime, newImage);

        rs::context ctx;
        printf("There are %d connected RealSense devices.\n", ctx.get_device_count());
        if(ctx.get_device_count() == 0) return EXIT_FAILURE;

        dev = ctx.get_device(0);

      //  dev->set_option(rs::option::color_enable_auto_exposure, false);
        dev->enable_stream(rs::stream::color, raw_width, raw_height, rs::format::rgb8, 30);
        //dev->enable_stream(rs::stream::color, rs::preset::best_quality);
      //  dev->set_frame_callback(rs::stream::color, *rgbCallback);

        std::thread runthread([&]()
        {
            dev->start();

            const rs::intrinsics color_intrinsics = dev->get_stream_intrinsics(rs::stream::color);

            std::vector<cv::Mat> sfmInitImages;
            std::vector<std::string> sfmInitImageNames;

            while (currentNumberOfFrames < totalNumberOfFrames)
            {
                if (dev->is_streaming())
                    dev->wait_for_frames();

                currentNumberOfFrames++;
                if (currentNumberOfFrames < ignoreNumOfFrames)
                {
                    continue;
                }

                const uint8_t* color_image = (const uint8_t *)dev->get_frame_data(rs::stream::color);
                cv::Mat image = cv::Mat(color_intrinsics.height, color_intrinsics.width, CV_8UC3);
                for(int y = 0; y < color_intrinsics.height; ++y)
                    for (int x = 0; x < color_intrinsics.width; ++x)
                    {
                        int index = y * color_intrinsics.width + x;

                        image.at<cv::Vec3b>( y , x )[0] = color_image[ index * 3 + 2 ];   // b
                        image.at<cv::Vec3b>( y , x )[1] = color_image[ index * 3 + 1 ];   // g
                        image.at<cv::Vec3b>( y , x )[2] = color_image[ index * 3 ];       // r
                    }

                cv::Mat m;
                cv::cvtColor(image, m, CV_BGR2GRAY);

//                cv::imshow("curr", m);
//                cv::waitKey(200);

                MinimalImageB minImg(m.cols, m.rows, (unsigned char*)m.data);
                ImageAndExposure* undistImg = undistorter->undistort<unsigned char>(&minImg, 1, 0, 1.0f);

                long long lastTimestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count();
                //std::string pic_name = "/Users/yinr/ComputerVision/SLAM/workspace/dso/build/r200/" + std::to_string(lastTimestamp) + ".jpg";
                //std::string pic_gray_name = "/Users/yinr/ComputerVision/SLAM/workspace/dso/build/r200/" + std::to_string(lastTimestamp) + "_g.jpg";

                //IOWrap::writeImage(pic_name, &minImg);
                //cv::imwrite(pic_name, cv::Mat(color.rows, color.cols, CV_8UC3, color.data));
                //cv::imwrite(pic_gray_name, cv::Mat(m.rows, m.cols, CV_8U, m.data));

                std::string pic_label = std::to_string(lastTimestamp) + ".jpg";

                if (currentNumberOfFrames < sfmInitNumOfFrames)
                {
                    sfmInitImageNames.push_back(pic_label);
                    sfmInitImages.push_back(m);

                    if (currentNumberOfFrames == sfmInitNumOfFrames - 1 )
                    {
                        fullSystem->setSfmInitializer(undistorter->getK(), sfmInitImageNames, sfmInitImages);

                        if (fullSystem->GetFirstFrameIndex() != -1 &&
                            fullSystem->GetSecondFrameIndex() != -1)
                        {
                            cv::Mat sfmGrayMat1 = fullSystem->GetFirstFrameMat();
                            cv::Mat sfmGrayMat2 = fullSystem->GetSecondFrameMat();

                            MinimalImageB minImg1(sfmGrayMat1.cols, sfmGrayMat1.rows, (unsigned char*)sfmGrayMat1.data);
                            MinimalImageB minImg2(sfmGrayMat2.cols, sfmGrayMat2.rows, (unsigned char*)sfmGrayMat2.data);

                            ImageAndExposure *sfmInitImg1 = undistorter->undistort<unsigned char>(&minImg1, 1, 0, 1.0f);
                            ImageAndExposure *sfmInitImg2 = undistorter->undistort<unsigned char>(&minImg2, 1, 0, 1.0f);

                            fullSystem->addSfmInitFrame(sfmInitImg1, fullSystem->GetFirstFrameIndex(), true);
                            fullSystem->addSfmInitFrame(sfmInitImg2, fullSystem->GetSecondFrameIndex(), false);
                        }
                    }

                    continue;
                }

                fullSystem->addActiveFrame(undistImg, frameID);
                frameID++;
                delete undistImg;
            }

            dev->stop();
        });


        if(viewer != 0)
            viewer->run();

        runthread.join();

        free( newImage );
        delete rgbCallback;

   //     cv::destroyAllWindows();

    }
    catch(const rs::error & e)
    {
        dev->stop();
        std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch(const std::exception & e)
    {
        dev->stop();
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

    dev->stop();

    return 0;
}
