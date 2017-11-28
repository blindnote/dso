//
// Created by Yin Rochelle on 11/27/17.
//

#ifndef DSO_SYSTEM_H
#define DSO_SYSTEM_H


#include <memory>
#include <mutex>

#include "util/ImageAndExposure.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/CoarseInitializer.h"



namespace dso
{

class System
{
public:
    System(const float* GInv, bool linear = true);
    ~System();

    void AddFrame(ImageAndExposure* p_img, int id);


protected:
    enum Status
    {
        NOT_STARTED_YET = -1,
        START_INITIALIZATION = 1,
        INITIALIZED = 2,
        ON_TRACK = 3,
        LOST = 4,
    };


    void SetGammaFunction(const float* BInv = nullptr);


    inline void SetStatus(Status status)
    {
        std::unique_lock<std::mutex> lock(mtx_status_);
        status_ = status;
    }


private:
    void Initialize(FrameHessian* p_fh);
    void Initialize2(FrameHessian* p_fh);

    void SetPreCalculateValues();


    uint64_t keyframe_ids_;

    bool linear_;

    std::mutex mtx_status_;
    Status status_;

    CalibHessian hessian_calib_;

    std::vector<FrameHessian*> vec_hessian_frames;

    std::shared_ptr<CoarseInitializer> sp_initializer_;
};

}


#endif //DSO_SYSTEM_H
