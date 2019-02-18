
#include <thread>
#include <chrono>
#include "Logger.h"

int main()
{
    logger::init("./log/sign_%Y-%m-%d_%H.%3N.log");

    while (true) {
        LOGGER_TRACE(logger::ERROR) << __LINE__ << "\t" << __FILE__ << "\t" << __FUNCTION__ << " log_type_error severity message";
        LOGGER_TRACE(logger::NORMAL) << __LINE__ << "\t" << __FILE__ << "\t" << __FUNCTION__ << " log_type_normal severity message";
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    return 0;
}