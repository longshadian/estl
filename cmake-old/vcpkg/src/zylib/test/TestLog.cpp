#include <chrono>
#include <iostream>
#include <thread>

#include "zylib/Logger.h"

void testAsyncTextFile()
{
    zylib::logger::FileOptional opt{};
    opt.m_file_name_pattern = "./server_%Y-%m-%d.log";
    zylib::logger::initAsyncFile(opt, zylib::logger::WARNING);
}

void testSyncTextFile()
{
    zylib::logger::FileOptional opt{};
    opt.m_file_name_pattern = "./server_%Y-%m-%d.log";
    zylib::logger::initSyncFile(opt);
}

void testSyncConsole()
{
    zylib::logger::initSyncConsole();
}

#define MY_LOG(s) LOG(s)

int main()
{
    testAsyncTextFile();
    //testSyncTextFile();
    //testSyncConsole();
    std::srand((unsigned int)std::time(nullptr));
    int n = 0;
    while (n < 3) {
        ++n;
        auto val = int( std::rand() % 1000);
        std::this_thread::sleep_for(std::chrono::seconds(2));
        LOG(DEBUG)  << "DEBUG " << val;
        LOG(INFO)   << "INFO " << val;
        LOG(WARNING)<< "WARNING " << val;
        LOG(ERROR)  << "ERROR " << val;
    }

    {
        zylib::logger::SafeExit exit{};
        (void)exit;
    }

    std::cout <<"==========\n";

    while (n>0) {
        --n;
        auto val = int( std::rand() % 1000);
        std::this_thread::sleep_for(std::chrono::seconds(2));
        LOG(DEBUG)  << "DEBUG " << val;
        LOG(INFO)   << "INFO " << val;
        LOG(WARNING)<< "WARNING " << val;
        LOG(ERROR)  << "ERROR " << val;
    }
    return 0;
}
