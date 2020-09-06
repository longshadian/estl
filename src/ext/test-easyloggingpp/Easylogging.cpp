
//#include <filesystem>

#include "Easylogging.h"

Easylogging::Easylogging()
{
}

Easylogging::~Easylogging()
{
}

int Easylogging::Init()
{
    const char* format_global = "%datetime{%Y-%M-%d %h:%m:%s.%g} %level [%thread] %line %func %msg";

    el::Configurations conf;
    conf.setGlobally(el::ConfigurationType::Format, format_global);
    conf.setGlobally(el::ConfigurationType::MaxLogFileSize, "1048576");
    conf.setGlobally(el::ConfigurationType::ToFile, "true");
    conf.setGlobally(el::ConfigurationType::SubsecondPrecision, "6");
    conf.setGlobally(el::ConfigurationType::ToStandardOutput, "false");
    conf.setGlobally(el::ConfigurationType::Filename, "easylog_%datetime{%Y%M%d_%h%m%s}.log");
    conf.setGlobally(el::ConfigurationType::LogFlushThreshold, "1");
    conf.setGlobally(el::ConfigurationType::PerformanceTracking, "false");
    // el::base::consts::kDefaultLoggerId
    el::Loggers::reconfigureLogger("default", conf);
    el::Loggers::addFlag(el::LoggingFlag::ImmediateFlush);
    el::Loggers::addFlag(el::LoggingFlag::StrictLogFileSizeCheck);
    el::Loggers::addFlag(el::LoggingFlag::DisableApplicationAbortOnFatalLog);

    el::Helpers::installPreRollOutCallback(Easylogging::LogRollOut);
    return 0;
}

int Easylogging::InitFromConf(const std::string& filename)
{
    el::Configurations conf(filename);
    el::Loggers::reconfigureAllLoggers(conf);
    el::Loggers::addFlag(el::LoggingFlag::ImmediateFlush);
    el::Loggers::addFlag(el::LoggingFlag::StrictLogFileSizeCheck);
    el::Loggers::addFlag(el::LoggingFlag::DisableApplicationAbortOnFatalLog);
    el::Helpers::installPreRollOutCallback(Easylogging::LogRollOut);
    return 0;
}

void Easylogging::LogRollOut(const char* filename, std::size_t size)
{
    static int index = 0;
    if (index >= 3) {
        index = 0;
    }

    std::ostringstream ostm{};
    ostm << filename << "." << ++index;
    std::string new_filename = ostm.str();

/*
    std::error_code ec{};
    if (std::filesystem::exists(new_filename, ec)) {
        std::filesystem::remove(new_filename, ec);
    }
    std::filesystem::rename(filename, new_filename.c_str(), ec);
*/
}

