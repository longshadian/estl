
#include <thread>
#include <chrono>
#include "Logger.h"

void fun()
{
    while (true) {
        LOG(DEBUG) << "log_type_debug severity message";
        LOG(INFO) <<  "log_type_info severity message";
        LOG(WARN) <<  "log_type_warn severity message";
        LOG(ERROR) << "log_type_error severity message";
        LOG_FMT(INFO, "int:%d str:%s float:%f", 12, "xxxx", 123.4);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
}

void testLongString()
{
    std::string s;
    s.push_back('[');
    for (int i = 0; i != 1300; ++i) {
        int val = i%10;
        auto temp = std::to_string(val);
        s.push_back(temp[0]);
    }
    s.push_back(']');
    LOG(DEBUG) << s ;
    LOG_FMT(DEBUG, "%s", s.c_str());
}

int main()
{
    logger::LogOptional opt{};
    opt.m_rotation_size = 1024 * 100;
    opt.m_file_name_pattern = "./log/sign_%Y-%m-%d_%H.%2N.log";
    logger::init(opt);

    testLongString();
    logger::stop();
    return 0;
}
