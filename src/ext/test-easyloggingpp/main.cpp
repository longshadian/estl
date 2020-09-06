
#include <thread>
#include <chrono>
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
//#include <filesystem>



#include "Easylogging.h"


//#define ELPP_THREAD_SAFE
//#define ELPP_FORCE_USE_STD_THREAD
//#define ELPP_DISABLE_FATAL_LOGS
#include "easylogging++.h"

INITIALIZE_EASYLOGGINGPP

void LogRollOut(const char* filename, std::size_t size)
{
    static int index = 0;
    std::ostringstream ostm{};
    ostm << filename << "." << ++index;
    std::string new_filename = ostm.str();

    std::error_code ec{};
    //std::filesystem::rename(filename, new_filename.c_str(), ec);
}

void InitLog()
{
    const char* format_global = "%datetime{%Y-%M-%d %h:%m:%s.%g} %level [%thread] %line %func %msg";
    const char* format_info = "%datetime{%Y-%M-%d %h:%m:%s.%g} %level %thread %file %line %func %msg";

    el::Configurations conf;
    conf.setGlobally(el::ConfigurationType::Format, format_global);
    conf.set(el::Level::Info, el::ConfigurationType::Format, format_info);
    conf.setGlobally(el::ConfigurationType::MaxLogFileSize, "1048576");
    conf.setGlobally(el::ConfigurationType::ToFile, "true");
    conf.setGlobally(el::ConfigurationType::SubsecondPrecision, "6");
    conf.setGlobally(el::ConfigurationType::Filename, "easylog_%datetime{%Y%M%d_%h%m%s}.log");
    conf.setGlobally(el::ConfigurationType::LogFlushThreshold, "1");
    conf.setGlobally(el::ConfigurationType::PerformanceTracking, "false");
   // el::base::consts::kDefaultLoggerId
    el::Loggers::reconfigureLogger("default", conf);
    el::Loggers::addFlag(el::LoggingFlag::ImmediateFlush);
    el::Loggers::addFlag(el::LoggingFlag::StrictLogFileSizeCheck);

    el::Helpers::installPreRollOutCallback(LogRollOut);

/*
    LOG(INFO) << "Log using default file";

    // To set GLOBAL configurations you may use
    conf.setGlobally(
        el::ConfigurationType::Format, "%date %msg");
    el::Loggers::reconfigureLogger("default", conf);
    */
}

void Test1(std::string b, std::vector<int> a)
{
    std::string s = "11111111111";
    s.resize(1000, 'a');
    while (1) {
        //LOG(GLOBAL) << s;
        LOG(TRACE) << s;
        //LOG(DEBUG) << s;
        LOG(FATAL) << s;
        LOG(ERROR) << s;
        LOG(WARNING) << s;
        LOG(INFO) << __FILE__ << " " << __FUNCTION__ << " " << s;
        //LOG(VERBOSE) << s;
        //LOG(Unknown) << s;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void Test2()
{
    std::cout << "Test2: " << std::this_thread::get_id()<< "\n";

    std::string s = "11111111111";
    s.resize(1000, 'b');
    while (1) {
        LOG(TRACE) << s;
        LOG(FATAL) << s;
        LOG(ERROR) << s;
        LOG(WARNING) << s;
        LOG(INFO) << s;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

int main(int argc, char** argv) 
{
    //InitLog();
    Easylogging log;
    //log.Init();
    log.InitFromConf("log.conf");

    std::thread t(Test2);

    std::cout << "Test1" << std::this_thread::get_id()<< "\n";

    Test1(std::string{}, std::vector<int>{});

    t.join();

    el::Helpers::uninstallPreRollOutCallback();
    return 0;
}


