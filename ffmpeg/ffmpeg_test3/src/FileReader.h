#pragma once

#include <iostream>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <thread>
#include <chrono>
#include <memory>

#include "xffmpeg/xffmpeg.h"

#define  FILE_BUFFER_SZIE (1024)
class File_H264
{
public:
    File_H264();
    ~File_H264();

    int Init(const char* p = nullptr);

    int Read(uint8_t** p, int* len)
    {
        if (!f_)
            return 0;
        size_t n = fread(buf_.data(), 1, buf_.size(), f_);
        *p = buf_.data();
        *len = (int)n;
        return *len;
    }
    void Reset();
    int ReadPkt(uint8_t** pdata, int* len);

    FILE* f_;
    std::string file_name_;
    std::vector<uint8_t> buf_;
    std::shared_ptr<xffmpeg::MemoryDemuxer> demuxer_;
};

typedef void (*PktProc)(const std::string& fname, uint8_t* pdata, int len);

class FileReader_H264
{
public:
    FileReader_H264();
    ~FileReader_H264();

    int CreateReader(const std::string& fname);
    int Init(PktProc proc);

    std::vector<std::shared_ptr<File_H264>> files_;
    std::thread work_thd_;
    std::thread watch_thd_;
    bool running_;

    int64_t current_frame_index;
    int64_t watch_frame_index;
    PktProc proc_;

private:
    void StartWorkThread();
    void StartWatchThread();

};

