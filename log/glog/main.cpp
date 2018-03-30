#include <glog/logging.h>  

#include <chrono>
#include <iostream>

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;

    /*
    google::InitGoogleLogging("haha");

    FLAGS_log_dir = "./log";

    LOG(INFO) << "hello world";
    LOG(WARNING) << "fsn " << 123 << "312.4";
    VLOG(2) << 111111111111;
    */

    auto tm = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(tm);
    std::cout << std::ctime(&t) << std::endl;
    return 0;
}
