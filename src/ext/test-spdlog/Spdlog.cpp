#include <string>
#include <iostream>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

void ErrMsg(const std::string& msg)
{
    std::cout << "spdlog error: " << msg << std::endl;
}

int main()
{
    auto console = spdlog::stdout_color_mt("console");
    //auto console = spdlog::basic_logger_mt("basic_logger", "logs/basic.txt");
    console->set_level(spdlog::level::trace);
    console->set_error_handler(std::bind(&ErrMsg, std::placeholders::_1));
    console->trace("xxx trace");
    console->debug("xxxx debug");
    console->info("xxxxx");
    console->warn("xxxx warn");
    console->error("xxxx error");
    console->critical("xxxx critical {}", 123, 123);
    try {
        spdlog::set_error_handler(&ErrMsg);
        spdlog::info("xxdf {} {}", 112);
        spdlog::drop_all();
    }
    catch (const spdlog::spdlog_ex& ex) {
        std::cout << "spdlog exception:}" << ex.what() << "\n";
    }

    system("pause");
    return 0;
}

