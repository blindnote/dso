//
// Created by Yin Rochelle on 11/17/17.
//
#include <iostream>
#include <vector>
#include <thread>
#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <boost/thread.hpp>
#include <opencv2/opencv.hpp>

#include "util/NumType.h"
#include "util/settings.h"
#include "util/globalFuncs.h"
#include "util/globalCalib.h"
#include "util/DatasetReader.h"

#include "FullSystem/FullSystem.h"
#include "FullSystem/PixelSelector2.h"
#include "OptimizationBackend/MatrixAccumulators.h"

#include "IOWrapper/Output3DWrapper.h"
#include "IOWrapper/ImageDisplay.h"
#include "IOWrapper/Pangolin/PangolinDSOViewer.h"
#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"


using namespace std;
using namespace dso;



std::string calib = "";
std::string source = "";
std::string gamma_calib = "";
std::string vignette = "";
std::string opencv_calib = "";

int start_seq =1;
int end_seq = 10;
//int end=100000;
float playbackSpeed=0;	// 0 for linearize (play as fast as possible, while sequentializing tracking & mapping). otherwise, factor on timestamps.


int mode=0;


std::string MyntPath = "/Users/yinr/ComputerVision/SLAM/workspace/MYNT-EYE-SDK/1.x/1.6/mynteye-1.6-mac-x64-opencv-3.2.0/";
std::string ImagePath = MyntPath + "samples/dataset/image_0/";
std::string TimesPath = MyntPath + "samples/dataset/times.txt";



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

    while(true) pause();
}



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






void parse_argument(char* arg)
{
    int option;
    float foption;
    char buf[1000];


    if(1==sscanf(arg,"preset=%d",&option))
    {
        settingsDefault(option);
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

    if(1==sscanf(arg,"start=%d",&option))
    {
        start_seq = option;
        printf("START AT %d!\n", start_seq);
        return;
    }
    if(1==sscanf(arg,"end=%d", &option))
    {
        end_seq = option;
        printf("END AT %d!\n", end_seq);
        return;
    }

    if(1==sscanf(arg,"opencv=%s",buf))
    {
        opencv_calib = buf;
        printf("loading opencv_calib from %s!\n", opencv_calib.c_str());
        return;
    }

    if(1==sscanf(arg,"files=%s",buf))
    {
        source = buf;
        printf("loading data from %s!\n", source.c_str());
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
        vignette = buf;
        printf("loading vignette from %s!\n", vignette.c_str());
        return;
    }

    if(1==sscanf(arg,"gamma=%s",buf))
    {
        gamma_calib = buf;
        printf("loading gammaCalib from %s!\n", gamma_calib.c_str());
        return;
    }

//    if(1==sscanf(arg,"speed=%f",&foption))
//    {
//        playbackSpeed = foption;
//        printf("PLAYBACK SPEED %f!\n", playbackSpeed);
//        return;
//    }

    if(1==sscanf(arg,"mode=%d",&option))
    {

        mode = option;
        if(option==0)
        {
            printf("PHOTOMETRIC MODE WITH CALIBRATION!\n");
        }
        if(option==1)
        {
            printf("PHOTOMETRIC MODE WITHOUT CALIBRATION!\n");
            setting_photometricCalibration = 0;
            setting_affineOptModeA = 0; //-1: fix. >=0: optimize (with prior, if > 0).
            setting_affineOptModeB = 0; //-1: fix. >=0: optimize (with prior, if > 0).
//
            setting_maxShiftWeightT= 0.04f * (752+480);
            setting_maxShiftWeightR= 0.0f * (752+480);
            setting_maxShiftWeightRT= 0.02f * (752+480);
        }
        if(option==2)
        {
            printf("PHOTOMETRIC MODE WITH PERFECT IMAGES!\n");
            setting_photometricCalibration = 0;
            setting_affineOptModeA = -1; //-1: fix. >=0: optimize (with prior, if > 0).
            setting_affineOptModeB = -1; //-1: fix. >=0: optimize (with prior, if > 0).
            setting_minGradHistAdd=3;
        }

//        if(option==3)
//        {
//            setting_photometricCalibration = 1;
//        }
        return;
    }

    printf("could not parse argument \"%s\"!!!!\n", arg);
}


std::string mynt_calib="/Users/yinr/ComputerVision/SLAM/workspace/MYNT-EYE-SDK/1.x/1.6/mynteye-1.6-mac-x64-opencv-3.2.0/settings/SN00D1190E0009062D.conf";



int main( int argc, char** argv )
{
    for(int i=1; i<argc;i++) parse_argument(argv[i]);

    // hook crtl+C.
    boost::thread exThread = boost::thread(exitThread);


    ImageFolderReader* reader = new ImageFolderReader(source, calib, gamma_calib, vignette, opencv_calib);
    reader->setGlobalCalibration();



    if(setting_photometricCalibration > 0 && reader->getPhotometricGamma() == 0)
    {
        printf("ERROR: dont't have photometric calibation. Need to use commandline options mode=1 or mode=2 ");
        exit(1);
    }


    FullSystem* fullSystem = new FullSystem();
    fullSystem->setGammaFunction(reader->getPhotometricGamma());
    fullSystem->linearizeOperation = (playbackSpeed==0);



    IOWrap::PangolinDSOViewer* viewer = 0;
    if(!disableAllDisplay)
    {
        viewer = new IOWrap::PangolinDSOViewer(wG[0],hG[0], false);
        fullSystem->outputWrapper.push_back(viewer);
    }



    std::thread runthread([&]() {

        for(auto i = start_seq; i < end_seq; i++)
//        for(auto i = start_seq; i < reader->getNumImages(); i++)
        {
            ImageAndExposure* img = reader->getImage_OpenCV(i);
//            ImageAndExposure* img = reader->getImage(i);

            fullSystem->addActiveFrame(img, i);

            delete img;

            if (fullSystem->initFailed)
            {
                printf("Init Failed:start_seq:%d, i:%d!!\n", start_seq, i);
                break;
//                if (i - start_seq > 30)
//                    break;
//
//                printf("Resetting!!\n");
//                std::vector<IOWrap::Output3DWrapper*> wraps = fullSystem->outputWrapper;
//                delete fullSystem;
//
//                for(IOWrap::Output3DWrapper* ow : wraps) ow->reset();
//
//                fullSystem = new FullSystem();
//                fullSystem->setGammaFunction(reader->getPhotometricGamma());
//                fullSystem->linearizeOperation = (playbackSpeed==0);
//
//                fullSystem->outputWrapper = wraps;

            }

            if (fullSystem->isLost)
            {
                printf("LOST!!\n");
                break;
            }
        }

        fullSystem->blockUntilMappingIsFinished();
    });


    if(viewer != 0)
        viewer->run();

    runthread.join();

    for(IOWrap::Output3DWrapper* ow : fullSystem->outputWrapper)
    {
        ow->join();
        delete ow;
    }


    printf("DELETE FULLSYSTEM!\n");
    delete fullSystem;

    printf("DELETE READER!\n");
    delete reader;

    printf("EXIT NOW!\n");


    return 0;
}
