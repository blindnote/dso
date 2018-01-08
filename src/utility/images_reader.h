//
// Created by Yin Rochelle on 11/21/17.
//

#ifndef DSO_IMAGES_READER_H
#define DSO_IMAGES_READER_H


#include <dirent.h>
#include <vector>
#include <fstream>
#include <string>

#if HAS_ZIPLIB
#include "zip.h"
#endif

#include "util/MinimalImage.h"
#include "IOWrapper/ImageRW.h"


using namespace dso;


class ImagesReader
{
public:
    ImagesReader(std::string path, int w, int h)
    : zipped_(false), original_width_(w), original_height_(h)
    {
#if HAS_ZIPLIB
        zip_archive_ = nullptr;
        data_buffer_ = nullptr;

        zipped_ = (path.length()>4 && path.substr(path.length()-4) == ".zip");
#endif

        int count = zipped_ ? GetFileListFromArchive(path, files_)
                    : GetFileListFromDirectory(path, files_);

        printf("got %d images\n", count);

        // load timestamps if possible.
        LoadTimestamps(path);
    }

    ~ImagesReader()
    {
#if HAS_ZIPLIB
        if(zip_archive_ != nullptr)
        {
            zip_close(zip_archive_);
        }
        if(data_buffer_ != nullptr)
        {
            delete data_buffer_;
        }
#endif
    }

    MinimalImageB* GetRawImage(int id)
    {
        if (!zipped_)
        {
            // CHANGE FOR ZIP FILE
            return IOWrap::readImageBW_8U(files_[id]);
        }

#if HAS_ZIPLIB
        const int Plus = 10000;
        const int SmallSize = original_width_ * original_height_ * 6 + 10000;
        const int LargeSize = original_width_ * original_height_ * 30 + 10000;

        if (data_buffer_ == nullptr)
            data_buffer_ = new char[SmallSize];

        zip_file_t* file = zip_fopen(zip_archive_, files_[id].c_str(), 0);
        long readbytes = zip_fread(file, data_buffer_, (long)SmallSize);

        if(readbytes > (long)(SmallSize - Plus))
        {
            printf("read %ld/%ld bytes for file %s. increase buffer!!\n",
                   readbytes,(long)SmallSize, files_[id].c_str());

            delete[] data_buffer_;
            data_buffer_ = new char[(long)LargeSize];
            file = zip_fopen(zip_archive_, files_[id].c_str(), 0);
            readbytes = zip_fread(file, data_buffer_, (long)LargeSize);

            if (readbytes > (long)(LargeSize - Plus))
            {
                printf("buffer still to small (read %ld/%ld). abort.\n",
                       readbytes,(long)LargeSize);
                exit(1);
            }
        }

        return IOWrap::readStreamBW_8U(data_buffer_, readbytes);
#endif
    }

    double GetTimestamp(int id, bool scale = false)
    {
        if (timestamps_.size() == 0)
            return !scale ? 0.0 : id*0.1f;
        if (id >= (int)timestamps_.size()) return 0;
        if (id < 0) return 0;
        return timestamps_[id];
    }

    float GetExposure(int id)
    {
        if (exposures_.size() == 0) return 1.0;
        if (id >= (int)exposures_.size()) return 0;
        if (id < 0) return 0;
        return exposures_[id];
    }


private:
    inline int GetFileListFromArchive(std::string path, std::vector<std::string> &files)
    {
        files.clear();

#if HAS_ZIPLIB
        int zip_error = 0;
        zip_archive_ = zip_open(path.c_str(), ZIP_RDONLY, &zip_error);
        if(zip_error != 0)
        {
            printf("ERROR %d reading archive %s!\n", zip_error, path.c_str());
            exit(1);
        }

        int entries_count = zip_get_num_entries(zip_archive_, 0);
        for(auto k = 0; k < entries_count; k++)
        {
            const char* name = zip_get_name(zip_archive_, k, ZIP_FL_ENC_STRICT);
            std::string nstr = std::string(name);
            if(nstr == "." || nstr == "..") continue;
            files.push_back(name);
        }

        printf("got %d entries and %d files!\n", entries_count, (int)files_.size());
        std::sort(files_.begin(), files_.end());
#endif

        return files.size();
    }

    inline int GetFileListFromDirectory(std::string directory, std::vector<std::string> &files)
    {
        files.clear();

        DIR *dp;
        struct dirent *dirp;
        if((dp  = opendir(directory.c_str())) == NULL)
        {
            return -1;
        }

        while ((dirp = readdir(dp)) != NULL) {
            std::string name = std::string(dirp->d_name);

            if(name != "." && name != "..")
                files.push_back(name);
        }
        closedir(dp);


        std::sort(files.begin(), files.end());

        if(directory.at( directory.length() - 1 ) != '/')
        {
            directory = directory + "/";
        }
        for(unsigned int i=0;i<files.size();i++)
        {
            if(files[i].at(0) != '/')
                files[i] = directory + files[i];
        }

        return files.size();
    }

    inline void LoadTimestamps(std::string path)
    {
        std::ifstream tr;
        std::string timestamps_file = path.substr(0,path.find_last_of('/')) + "/times.txt";
        tr.open(timestamps_file.c_str());
        while(!tr.eof() && tr.good())
        {
            std::string line;
            char buf[200];
            tr.getline(buf, 200);

            int id;
            double stamp;
            float exposure = 0;

//            if(3 == sscanf(buf, "%d %lf %f", &id, &stamp, &exposure))
//            {
//                timestamps_.push_back(stamp);
//                exposures_.push_back(exposure);
//            }

//            else if(2 == sscanf(buf, "%lf %f", &stamp, &exposure))
            if (2 == sscanf(buf, "%le %f", &stamp, &exposure))
            {
                timestamps_.push_back(stamp);
                exposures_.push_back(exposure);
            }
        }
        tr.close();

        // check if exposures are correct, (possibly skip)
        bool exposures_good = ((int)exposures_.size() == (int)files_.size()) ;
        for (int i = 0; i < (int)exposures_.size(); i++)
        {
            if (exposures_[i] == 0)
            {
                // fix!
                float sum=0,num=0;
                if(i > 0 && exposures_[i-1] > 0) { sum += exposures_[i-1]; num++; }
                if(i+1 < (int)exposures_.size() && exposures_[i+1] > 0) { sum += exposures_[i+1]; num++; }

                if(num > 0)
                    exposures_[i] = sum / num;
            }

            if(exposures_[i] == 0) exposures_good = false;
        }


        if((int)files_.size() != (int)timestamps_.size())
        {
            printf("set timestamps and exposures to zero!\n");
            exposures_.clear();
            timestamps_.clear();
        }

        if((int)files_.size() != (int)exposures_.size() || !exposures_good)
        {
            printf("set EXPOSURES to zero!\n");
            exposures_.clear();
        }

        printf("got %d timestamps and %d exposures.\n",
               (int)timestamps_.size(), (int)exposures_.size());
    }


    std::vector<std::string> files_;
    std::vector<double> timestamps_;
    std::vector<float> exposures_;

//    int width_, height_;
    int original_width_, original_height_;


    bool zipped_;
#if HAS_ZIPLIB
    zip_t* zip_archive_;
    char* data_buffer_;
#endif
};



#endif //DSO_IMAGES_READER_H
