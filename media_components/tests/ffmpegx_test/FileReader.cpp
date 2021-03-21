#include "FileReader.h"

#include "console_log.h"

#include <functional>

File_H264::File_H264()
    : f_()
    , file_name_()
    , buf_(FILE_BUFFER_SZIE, '\0')
    , demuxer_(std::make_shared<xffmpeg::MemoryDemuxer>())
{
}

File_H264::~File_H264()
{
    if (f_)
        fclose(f_);
}

int File_H264::Init(const char* p)
{
    if (p)
        file_name_= p;
    f_ = fopen(file_name_.c_str(), "rb");
    if (!f_)
        return -1;
    int ret = demuxer_->Init();
    if (ret != 0)
        std::cout << "demuxer init failed\n";
    return ret;
}

void File_H264::Reset()
{
    fseek(f_, 0, SEEK_SET);
    demuxer_ = nullptr;
    demuxer_ = std::make_shared<xffmpeg::MemoryDemuxer>();
    demuxer_->Init();
}

int File_H264::ReadPkt(uint8_t** pdata, int* len)
{
    int ret = 0;
    size_t readn = 0;
    while (ret == 0) {
        ret = demuxer_->Demux(pdata, len);
        if (ret < 0) {
            return -1;
        } else if (ret == 0) {
            readn = ::fread(buf_.data(), 1, buf_.size(), f_);
            if (readn == 0) {
                std::cout << "read file EOF!";
                return 0;
            }
            demuxer_->data_ = buf_.data();
            demuxer_->data_size_ = readn;
        } else {
            return 1;
        }
    } 
}

/************************************************************************/
/*                                                                      */
/************************************************************************/

FileReader_H264::FileReader_H264()
    : files_()
    , work_thd_()
    , watch_thd_()
    , running_()
    , current_frame_index()
    , watch_frame_index()
    , proc_()
{
}

FileReader_H264::~FileReader_H264()
{
    running_ = false;
    if (watch_thd_.joinable())
        watch_thd_.join();
    if (work_thd_.joinable())
        work_thd_.join();
}

int FileReader_H264::CreateReader(const std::string& fname)
{
    auto ff = std::make_shared<File_H264>();
    int ret = ff->Init(fname.c_str());
    if (ret != 0)
        return -1;
    files_.push_back(ff);
    return (int)files_.size()-1;
}

int FileReader_H264::Init(PktProc proc)
{
    proc_ = proc;
    running_ = true;
    {
        std::thread tmp(std::bind(&FileReader_H264::StartWorkThread, this));
        std::swap(tmp, work_thd_);
    }
    {
        std::thread tmp(std::bind(&FileReader_H264::StartWatchThread, this));
        std::swap(tmp, watch_thd_);
    }
    return 0;
}

void FileReader_H264::StartWorkThread()
{
    int ret = 0;
    while (running_) {
        for (auto& f : files_) {
            uint8_t* p = nullptr;
            int len = 0;

            ret = f->ReadPkt(&p, &len);
            if (ret == 0) {
                f->Reset();
                ret = f->ReadPkt(&p, &len);
                std::cout << "reset file";
            }
            if (len > 0) {
                proc_(f->file_name_, p, len);
            }
        }

        current_frame_index++;
        while (current_frame_index >= watch_frame_index) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        //CONSOLE_LOG_INFO << "next frame: " << current_frame_index << " " << watch_frame_index << "\n";
    }
}

void FileReader_H264::StartWatchThread()
{
    while (running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        ++watch_frame_index;
    }
}


