#ifndef __SRSRTMP_H
#define __SRSRTMP_H

#include <string>

class SrsRtmp
{
public:
    SrsRtmp();
    explicit SrsRtmp(std::string url);
    ~SrsRtmp();

    int Init();
    int SendH264(void* frameData, size_t frameLen, uint64_t pts);
    int Reset(std::string url);
    int Reset();
    const std::string& URL() const;
private:
    static int Create(const char* url, void** hdl);
    static void Destroy(void* hdl);

    void* hdl_;
    std::string url_;
    int total_num_;
};

#endif

