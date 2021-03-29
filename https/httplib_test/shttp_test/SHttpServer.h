#ifndef _CTRHTTP_SHTTPSERVER_H
#define _CTRHTTP_SHTTPSERVER_H

#include <memory>
#include <string>
#include <thread>

#include "httplib.h"

class SHttpServer
{
public:
    SHttpServer();
    ~SHttpServer();

    int Init(const std::string& ip, int port);
    void Loop();
    void LoopInBackground();
    void Stop();

private:
    std::shared_ptr<httplib::Server> server_;
    std::thread thd_;
    std::string ip_;
    int port_;
};

#endif

