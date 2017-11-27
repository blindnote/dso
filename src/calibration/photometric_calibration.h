//
// Created by Yin Rochelle on 11/21/17.
//

#ifndef DSO_PHOTOMETRIC_CALIBRATION_H
#define DSO_PHOTOMETRIC_CALIBRATION_H


#include "util/MinimalImage.h"
#include "util/ImageAndExposure.h"


namespace dso {

class PhotometricCalibration
{
public:
    PhotometricCalibration(int w, int h,
                           std::string gamma_file,
                           std::string vignette_image);
    ~PhotometricCalibration();

    template<class T>
    void ApplyCorrection(ImageAndExposure* output_image, T* input_image,
                         float exposure_time, float factor = 1.0);

    float* GetGammaInv()
    {
        return !gamma_valid_ ? 0 : gamma_inv_;
    }

    inline bool GammaValid() { return gamma_valid_; }
    inline bool VignetteValid() { return vignette_valid_; }


private:
    bool LoadGamma(std::string gamma_file);
    bool LoadVignette(std::string vignette_image);


    bool gamma_valid_;
    bool vignette_valid_;

    int w_, h_;

    float gamma_inv_[256*256];
    int gamma_depth_;

    float* vignette_;
    float* vignette_inv_;
};

}

#endif //DSO_PHOTOMETRIC_CALIBRATION_H
