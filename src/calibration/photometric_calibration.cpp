//
// Created by Yin Rochelle on 11/21/17.
//


#include <fstream>

#include "util/settings.h"
#include "util/MinimalImage.h"
#include "IOWrapper/ImageRW.h"
#include "photometric_calibration.h"



namespace dso
{

PhotometricCalibration::PhotometricCalibration(int w, int h,
                                               std::string gamma_file,
                                               std::string vignette_image)
        : w_(w), h_(h), gamma_depth_(256), vignette_(0), vignette_inv_(0)
{
    gamma_valid_ = LoadGamma(gamma_file);
    vignette_valid_ = LoadVignette(vignette_image);
}

PhotometricCalibration::~PhotometricCalibration()
{
    if (vignette_ != 0) delete[] vignette_;
    if (vignette_inv_ != 0) delete[] vignette_inv_;
}


template<class T>
void PhotometricCalibration::ApplyCorrection(ImageAndExposure* output_image, T* input_image,
                                             float exposure_time, float factor)
{
    assert(output_image != 0);
    assert(output_image->w == w_ && output_image->h == h_);
    float* data = output_image->image;
    assert(data != 0);

    int wh = w_ * h_;
    if(!gamma_valid_ || !vignette_valid_ || exposure_time <= 0
       || setting_photometricCalibration == 0) // disable full photometric calibration.
    {
        for(int i = 0; i < wh; i++)
        {
            data[i] = factor * input_image[i];
        }
        output_image->exposure_time = exposure_time;
        output_image->timestamp = 0;
    }
    else
    {
        for(int i = 0; i < wh; i++)
        {
            data[i] = gamma_inv_[input_image[i]];
        }

        if(setting_photometricCalibration == 2)
        {
            for(int i = 0; i < wh;i++)
                data[i] *= vignette_inv_[i];
        }

        output_image->exposure_time = exposure_time;
        output_image->timestamp = 0;
    }

    if (!setting_useExposure)
        output_image->exposure_time = 1;
}
template void PhotometricCalibration::ApplyCorrection<unsigned char>(ImageAndExposure *output_image,
                                                                     unsigned char *input_image,
                                                                     float exposure_time, float factor);
template void PhotometricCalibration::ApplyCorrection<unsigned short>(ImageAndExposure *output_image,
                                                                      unsigned short *input_image,
                                                                      float exposure_time, float factor);


bool PhotometricCalibration::LoadGamma(std::string gamma_file)
{
    if (gamma_file == "")
    {
        printf("No Gamma calibration!\n");
        return false;
    }

    std::ifstream f(gamma_file.c_str());
    printf("Reading Gamma Calibration from ... %s\n", gamma_file.c_str());
    if (!f.good())
    {
        printf("PhotometricCalibration: Could not open file!\n");
        f.close();
        return false;
    }

    // read G.
    std::string line;
    std::getline( f, line );
    std::istringstream l1i( line );
    std::vector<float> Gvec = std::vector<float>( std::istream_iterator<float>(l1i), std::istream_iterator<float>() );
    f.close();

    gamma_depth_ = Gvec.size();

    if(gamma_depth_ < 256)
    {
        printf("PhotometricCalibration: invalid format! got %d entries in first line, expected at least 256!\n",(int)Gvec.size());
        return false;
    }

    for(int i = 0;i < gamma_depth_; i++) gamma_inv_[i] = Gvec[i];

    // check
    for(int i = 0;i < gamma_depth_ - 1; i++)
    {
        if(gamma_inv_[i+1] <= gamma_inv_[i])
        {
            printf("PhotometricCalibration: G-1 invalid! it has to be strictly increasing, but it isn't!\n");
            return false;
        }
    }

    float min = gamma_inv_[0];
    float max = gamma_inv_[gamma_depth_ - 1];
    for(int i = 0;i < gamma_depth_; i++)
        gamma_inv_[i] = 255.0 * (gamma_inv_[i] - min) / (max - min);			// make it to 0..255 => 0..255.


    if (setting_photometricCalibration == 0)
    {
        for(int i = 0; i < gamma_depth_; i++)
            gamma_inv_[i] = 255.0f * i / (float)(gamma_depth_ - 1);
    }

    printf("Successfully read gamma calibration!\n");
    return true;
}


bool PhotometricCalibration::LoadVignette(std::string vignette_image)
{
    if (vignette_image == "")
    {
        printf("No vignette calibration!\n");
        return false;
    }

    printf("Reading Vignette Image from ... %s\n", vignette_image.c_str());

    int resolution = w_ * h_;
    vignette_ = new float[resolution];
    vignette_inv_ = new float[resolution];

    MinimalImage<unsigned short>* vm16 = IOWrap::readImageBW_16U(vignette_image.c_str());
    MinimalImageB* vm8 = IOWrap::readImageBW_8U(vignette_image.c_str());

    if (vm16 != 0)
    {
        if (vm16->w != w_ || vm16->h != h_)
        {
            printf("PhotometricCalibration: Invalid vignette image size! got %d x %d, expected %d x %d\n",
                   vm16->w, vm16->h, w_, h_);
            if (vm16 != 0) delete vm16;
            if (vm8 != 0) delete vm8;
            return false;
        }

        float maxV=0;
        for (int i = 0;i< resolution; i++)
            if (vm16->at(i) > maxV) maxV = vm16->at(i);

        for (int i = 0;i < resolution; i++)
            vignette_[i] = vm16->at(i) / maxV;
    }
    else if(vm8 != 0)
    {
        if(vm8->w != w_ || vm8->h != h_)
        {
            printf("PhotometricCalibration: Invalid vignette image size! got %d x %d, expected %d x %d\n",
                   vm8->w, vm8->h, w_, h_);
            if (vm16 != 0) delete vm16;
            if (vm8 != 0) delete vm8;
            return false;
        }

        float maxV=0;
        for (int i = 0; i< resolution; i++)
            if (vm8->at(i) > maxV) maxV = vm8->at(i);

        for (int i = 0; i < resolution;i++)
            vignette_[i] = vm8->at(i) / maxV;
    }
    else
    {
        printf("PhotometricCalibration: Invalid vignette image\n");
        if (vm16 != 0) delete vm16;
        if (vm8 != 0) delete vm8;
        return false;
    }

    for(int i = 0; i < resolution; i++)
        vignette_inv_[i] = 1.0f / vignette_[i];


    printf("Successfully read vignette calibration!\n");
    return true;
}

}