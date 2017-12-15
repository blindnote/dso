//
// Created by Yin Rochelle on 12/14/17.
//
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <set>

#include <opencv2/highgui/highgui.hpp>

#include "camera.h"
#include "utility.h"

#ifdef USE_OPENCV2
#include <opencv2/highgui/highgui.hpp>
#else
#include <opencv2/imgcodecs/imgcodecs.hpp>
#endif

#if defined(_WIN32)
#  define OS_WIN
#  if defined(__MINGW32__) || defined(__MINGW64__)
#    define OS_MINGW
#  endif
#else
#  define OS_UNIX
#endif

#if defined(OS_WIN) && !defined(OS_MINGW)
#  include <direct.h>
#  define PATH_SEPARATOR "\\"
#else
#  include <sys/stat.h>
#  define PATH_SEPARATOR "/"
#endif


#ifdef OS_WIN

#include <conio.h>

#else

#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <termios.h>
// #include <stropts.h>

int _kbhit() {
    static const int STDIN = 0;
    static bool initialized = false;

    if (!initialized) {
        // Use termios to turn off line buffering
        termios term;
        tcgetattr(STDIN, &term);
        term.c_lflag &= ~ICANON;
        tcsetattr(STDIN, TCSANOW, &term);
        setbuf(stdin, NULL);
        initialized = true;
    }

    int bytesWaiting;
    ioctl(STDIN, FIONREAD, &bytesWaiting);
    return bytesWaiting;
}

#endif


namespace
{

    template<typename Duration>
    int64_t count(const std::chrono::system_clock::duration &d) {
        return std::chrono::duration_cast<Duration>(d).count();
    }

    template<typename Duration>
    std::chrono::system_clock::time_point cast(const std::chrono::system_clock::time_point &d) {
        return std::chrono::time_point_cast<Duration>(d);
    }

    std::string to_string(const std::chrono::system_clock::time_point &t,
                          const char *fmt = "%F %T",
                          int32_t precision = 6) {
        auto t_c = std::chrono::system_clock::to_time_t(t);
        std::tm *tm = std::localtime(&t_c);

        std::stringstream ss;
#if defined(OS_ANDROID) || defined(OS_LINUX)
        char foo[20];
    strftime(foo, sizeof(foo), fmt, tm);
    ss << foo;
#else
        ss << std::put_time(tm, fmt);
#endif
        if (precision > 0) {
            if (precision > 6) precision = 6;
            int64_t ms = count<std::chrono::milliseconds>(t - cast<std::chrono::seconds>(t));
            ss << '.' << std::setfill('0') << std::setw(precision)
               << static_cast<int32_t>(ms / std::pow(10, 6-precision));
        }
        return ss.str();
    }

    /////////////////////////////////////////////////////////////////////
    void MakeDir(const std::string &dir)
    {
    #if defined(OS_MINGW)
        const int status = mkdir(dir.c_str());
    #elif defined(OS_WIN)
        const int status = _mkdir(dir.c_str());
    #else // OS_UNIX
        const int status = mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    #endif
        if (status != 0 && errno != EEXIST) {
            std::cerr << "Error: Create directory failed." << std::endl;
            std::cerr << "  Status: " << status << std::endl
                      << "  Path: " << dir << std::endl;
            exit(2);
        }
        if (errno != EEXIST) {
            std::cout << "Create directory: " << dir << std::endl;
        }
    }


    class VIODataset
    {
    public:
        VIODataset(const std::string &outdir)
                : outdir_(outdir), count_(0),
                  dir_img_left_(outdir + PATH_SEPARATOR"cam0"),
                  dir_img_right_(outdir + PATH_SEPARATOR"cam1"),
                  file_timestamps_(outdir + PATH_SEPARATOR"timestamps.txt"),
                  file_imudatum_(outdir + PATH_SEPARATOR"imu.txt")
        {
            MakeDir(outdir);
            MakeDir(dir_img_left_);
            MakeDir(dir_img_right_);

            ofs_timestamps_.open(file_timestamps_, std::ofstream::out);
            ofs_timestamps_ << std::scientific;
            ofs_imudatum_.open(file_imudatum_, std::ofstream::out);
            ofs_imudatum_ << std::scientific;

            ofs_timestamps_ << "#timestamp[ms]" << std::endl;
            ofs_imudatum_ << "#timestamp[ms],gyro_x,gyro_y,gyro_z,acc_x,acc_y,acc_z" << std::endl;
        }

        virtual ~VIODataset()
        {
            ofs_imudatum_.flush();
            ofs_timestamps_.flush();

            ofs_imudatum_.close();
            ofs_timestamps_.close();
        }

        void Save(const cv::Mat &left, const cv::Mat &right,
                  uint32_t timestamp, std::vector<mynteye::IMUData> &imudatum)
        {
            std::stringstream ss;
            ss << std::dec << std::setw(6) << std::setfill('0') << count_++ << ".png";
            std::string filename = ss.str();

            std::string path_img_left = dir_img_left_ + PATH_SEPARATOR + filename;
            std::string path_img_right = dir_img_right_ + PATH_SEPARATOR + filename;

            cv::imwrite(path_img_left, left);
            cv::imwrite(path_img_right, right);

            ofs_timestamps_ << timestamp << std::endl;

            if (imudatum.empty()) return;

            for(auto i = 0; i < imudatum.size(); i++)
            {
                auto &imudata = imudatum[i];
                ofs_imudatum_ << (imudata.time / 10) << ","
                              << std::fixed << std::setprecision(13)
                              << imudata.gyro_x << "," << imudata.gyro_y << "," << imudata.gyro_z << ","
                              << imudata.accel_x << "," << imudata.accel_y << "," << imudata.accel_z
                              << std::endl;
            }
        }

        std::uint64_t GetCount() { return count_; }

    private:
        std::string outdir_;
        uint64_t count_;

        std::string dir_img_left_;
        std::string dir_img_right_;
        std::string file_timestamps_;
        std::string file_imudatum_;

        std::ofstream ofs_timestamps_;
        std::ofstream ofs_imudatum_;
    };



}  // namesapce


//#define GUI_DRAWINFO
#define GUI_VISIBLE
#define LOG_VERBOSE
#define LOG_TIMESTAMP_DETECTION

using namespace std;
using namespace mynteye;

/**
 * ./samples/build/output/bin/camera1 [name] &> log.txt
 */
int main(int argc, char const *argv[]) {
    const char *name;
    if (argc >= 2) {
        name = argv[1];
    } else {
#ifdef _WIN32
        name = "MYNTEYE";
#else
        name = "0";
#endif
    }
    cout << "Open Camera: " << name << endl;

    Camera cam;
    InitParameters params(name);
    cam.Open(params);

    if (!cam.IsOpened()) {
        std::cerr << "Error: Open camera failed" << std::endl;
        return 1;
    }
#ifdef GUI_VISIBLE
    std::cout << "\033[1;32mPress ESC/Q on Windows to terminate\033[0m\n";
#else
    std::cout << "\033[1;32mPress any key to terminate\033[0m\n";
#endif

    cam.ActivateAsyncGrabFeature(true);


    const char *outdir;
    if (argc >= 3) {
        outdir = argv[2];
    } else {
        outdir = "./dataset";
    }
    VIODataset *dataset = new VIODataset(outdir);
    cout << "Capture dataset to directory:\n  " << outdir << endl;



    ErrorCode code;
    cv::Mat img_left, img_right;

#ifdef GUI_DRAWINFO
    double t, fps = 0;
#endif

#ifdef LOG_VERBOSE
    std::uint64_t img_count = 0;
    std::uint64_t imu_count = 0;

    std::vector<IMUData> imudatas;
    std::uint32_t timestamp;

#ifdef LOG_TIMESTAMP_DETECTION
    std::uint64_t imu_time_prev = 0;
    std::set<std::uint64_t> imu_err_img_set;
#endif

    auto time_beg = chrono::system_clock::now();;
#endif

    for (;;) {
#ifdef GUI_DRAWINFO
        t = (double)cv::getTickCount();
#endif

        code = cam.Grab();

        if (code != ErrorCode::SUCCESS) continue;

        if (cam.RetrieveImage(img_left, View::VIEW_LEFT_UNRECTIFIED) == ErrorCode::SUCCESS &&
            cam.RetrieveImage(img_right, View::VIEW_RIGHT_UNRECTIFIED) == ErrorCode::SUCCESS) {

            std::uint32_t timestamp_grab_image = cam.GetTimestamp();

            auto elapsed_ms = count<chrono::milliseconds>(chrono::system_clock::now() - time_beg);

#ifdef GUI_DRAWINFO
            // top left: width x height
            stringstream ss;
            ss << img_left.cols << "x" << img_left.rows;
            DrawInfo(img_left, ss.str(), Gravity::TOP_LEFT);
            DrawInfo(img_right, ss.str(), Gravity::TOP_LEFT);
            // top right: fps
            ss.str(""); ss.clear();
            ss << fixed << setw(7) << setprecision(2) << setfill(' ') << fps;
            DrawInfo(img_left, ss.str(), Gravity::TOP_RIGHT);
            DrawInfo(img_right, ss.str(), Gravity::TOP_RIGHT);
#endif

#ifdef GUI_VISIBLE
            cv::imshow("left", img_left);
            cv::imshow("right", img_right);
#endif

#ifdef LOG_VERBOSE
            ++img_count;

            cam.RetrieveIMUData(imudatas, timestamp);
            imu_count += imudatas.size();

            dataset->Save(img_left, img_right, timestamp_grab_image/10, imudatas);

//            auto elapsed_ms = count<chrono::milliseconds>(chrono::system_clock::now() - time_beg);
/*
            cout << endl;
            cout << "img count: " << img_count
                 << ", avg fps: " << fixed << setprecision(2)
                 << (img_count * 1000.f / elapsed_ms) << endl
                 << "  timestamp: " << (timestamp / 10) << " ms"
                 << "  sys img timestamp ms: " << elapsed_ms
                 << endl;
            cout << "imu count: " << imu_count
                 << ", avg fps: " << fixed << setprecision(2)
                 << (imu_count * 1000.f / elapsed_ms)
                 << endl;

            if (!imudatas.empty()) {
                size_t size = imudatas.size();
                cout << "imu size: " << size << endl;
                for (size_t i = 0; i < size; ++i) {
                    auto &imudata = imudatas[i];
                    cout << "  imu[" << i << "] time: " << (imudata.time / 10) << " ms"
                         << ", accel(" << imudata.accel_x << "," << imudata.accel_y << "," << imudata.accel_z << ")"
                         << ", gyro(" << imudata.gyro_x << "," << imudata.gyro_y << "," << imudata.gyro_z << ")"
                         << endl;
#ifdef LOG_TIMESTAMP_DETECTION
                    if (imu_time_prev != 0) {
                        if (imu_time_prev >= imudata.time) {  // incorrect
                            cerr << "imu timestamp error, img: " << img_count
                                 << ", imu: " << (imu_count - size + i) << endl;
                            imu_err_img_set.insert(img_count);
                        }
                    }
                    imu_time_prev = imudata.time;
#endif
                }
            }
 */
#endif
        }

#ifdef GUI_VISIBLE
        char key = (char) cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') {  // ESC/Q
            break;
        }
#else
        if (_kbhit()) break;
#endif

#ifdef GUI_DRAWINFO
        t = (double)cv::getTickCount() - t;
        fps = cv::getTickFrequency() / t;
#endif
    }

#ifdef LOG_VERBOSE
    auto time_end = chrono::system_clock::now();
    auto time_cost = time_end - time_beg;
    auto time_cost_ms = count<chrono::milliseconds>(time_cost);
    cout << endl << "----------" << endl << endl
         << "time beg: " << to_string(time_beg) << endl
         << "time end: " << to_string(time_end) << endl
         << "time cost: " << time_cost_ms << " ms" << endl << endl
         << "img count: " << img_count << endl
         << "img avg fps: " << fixed << setprecision(2)
         << (img_count * 1000.f / time_cost_ms) << endl << endl
         << "imu count: " << imu_count << endl
         << "img avg fps: " << fixed << setprecision(2)
         << (imu_count * 1000.f / time_cost_ms) << endl << endl
         << "img : imu = " << img_count << " : " << imu_count
         << " = 1 : " << (1.f * imu_count / img_count) << endl;
#ifdef LOG_TIMESTAMP_DETECTION
    cout << endl << "imu timestamp detection: ";
    if (imu_err_img_set.empty()) {
        cout << "ok" << endl;
    } else {
        cout << imu_err_img_set.size() << " frames are incorrect" << endl;
        for (auto &&img_count : imu_err_img_set) {
            cout << "  img " << img_count << endl;
        }
    }
#endif
#endif

    cam.Close();
    cv::destroyAllWindows();
    return 0;
}
