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

    bool AddFrame(ImageAndExposure* p_img, int id);

    enum Status
    {
        NOT_STARTED_YET = -1,
        START_INITIALIZATION = 1,
        INITIALIZED = 3,
        ON_TRACK = 4,
        LOST = 5,
    };


protected:

    void SetGammaFunction(const float* BInv = nullptr);


    inline void SetStatus(Status status)
    {
        std::unique_lock<std::mutex> lock(mtx_status_);
        status_ = status;
    }
    inline Status GetStatus()
    {
        std::unique_lock<std::mutex> lock(mtx_status_);
        return status_;
    }


private:
    void Initialize(FrameHessian* p_fh);
    void InitializeWithPointsSorted(FrameHessian* p_fh);

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
