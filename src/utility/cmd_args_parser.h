//
// Created by Yin Rochelle on 11/21/17.
//

#ifndef DSO_CMD_ARGS_PARSER_H
#define DSO_CMD_ARGS_PARSER_H

#include <string>

#include "util/settings.h"


using namespace dso;


float playbackSpeed = 0.0; 	// 0 for linearize (play as fast as possible, while sequentializing tracking & mapping). otherwise, factor on timestamps.


void SetDefaultParameters(int preset)
{
    printf("\n=============== PRESET Settings: ===============\n");

    if(preset == 0 || preset == 1)
    {
        printf("DEFAULT settings:\n"
                       "- %s real-time enforcing\n"
                       "- 2000 active points\n"
                       "- 5-7 active frames\n"
                       "- 1-6 LM iteration each KF\n"
                       "- original image resolution\n", preset == 0 ? "no " : "1x");

        playbackSpeed = (preset == 0 ? 0 : 1);
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
                       "- 424 x 320 image resolution\n", preset == 0 ? "no" : "5x");

        playbackSpeed = (preset == 2 ? 0 : 5);
        setting_desiredImmatureDensity = 600;
        setting_desiredPointDensity = 800;
        setting_minFrames = 4;
        setting_maxFrames = 6;
        setting_maxOptIterations=4;
        setting_minOptIterations=1;

//        benchmarkSetting_width = 424;
        benchmarkSetting_width = 500;
        benchmarkSetting_height = 320;

        setting_logStuff = false;
    }

    printf("==================================================\n");
}


std::string images_path = "";
std::string calib_file = "";
std::string vignette_file = "";
std::string gamma_file = "";
std::string opencv_yaml = "";
int start_id = 1;
int end_id = 100;


void ParseArgument(char* arg)
{
    char buffer[250];
    int option;

    if(1==sscanf(arg,"preset=%d",&option))
    {
        SetDefaultParameters(option);
        return;
    }

    if(1==sscanf(arg, "files=%s", buffer))
    {
        images_path = buffer;
        printf("images => %s\n", images_path.c_str());
        return;
    }

    if(1==sscanf(arg, "calib=%s", buffer))
    {
        calib_file = buffer;
        printf("calibration => %s\n", calib_file.c_str());
        return;
    }

    if(1==sscanf(arg, "vignette=%s", buffer))
    {
        vignette_file = buffer;
        printf("vignette => %s\n", vignette_file.c_str());
        return;
    }

    if(1==sscanf(arg, "gamma=%s", buffer))
    {
        gamma_file = buffer;
        printf("gamma => %s\n", gamma_file.c_str());
        return;
    }

    if(1==sscanf(arg, "opencv=%s", buffer))
    {
        opencv_yaml = buffer;
        printf("opencv calib => %s\n", opencv_yaml.c_str());
        return;
    }

    if(1==sscanf(arg,"start=%d",&option))
    {
        start_id = option;
        printf("start => %d\n", start_id);
        return;
    }

    if(1==sscanf(arg,"end=%d",&option))
    {
        end_id = option;
        printf("end => %d\n", end_id);
        return;
    }

    printf("could not parse argument \"%s\"!!!!\n", arg);
}


#endif //DSO_CMD_ARGS_PARSER_H
