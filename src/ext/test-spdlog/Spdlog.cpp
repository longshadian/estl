#include <string>
#include <iostream>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

static void ErrMsg(const std::string& msg)
{
    std::cout << "==> spdlog error callback: " << msg << std::endl;
}

static std::string ThreadID()
{
    std::ostringstream ostm{};
    ostm << std::this_thread::get_id();
    return ostm.str();
}

static int ConsoleTest()
{
    std::string fmt_pattern =
        "[%Y-%m-%d %H:%M:%S.%f] [%^%l%$] [%t] [%s:%#] %v";
    auto console = spdlog::stdout_color_mt("console");
    //auto console = spdlog::basic_logger_mt("basic_logger", "logs/basic.txt");
    console->set_pattern(fmt_pattern);
    console->set_level(spdlog::level::trace);
    console->set_error_handler(std::bind(&ErrMsg, std::placeholders::_1));
    console->trace("xxx trace");
    console->debug("xxxx debug");
    console->info("xxxxx");
    console->warn("xxxx warn");
    console->error("xxxx error");
    console->critical("xxxx critical {}", 123, 123);

    spdlog::set_default_logger(std::make_shared<spdlog::logger>("multi_sink",
        spdlog::sinks_init_list({ console })));
    try {
        spdlog::set_error_handler(&ErrMsg);
        spdlog::info("xxdf {} {}", 112);

        int n = 0;
        while (1) {
            ++n;
            std::this_thread::sleep_for(std::chrono::seconds(1));
            console->trace("xxx     trace {}", n);
            console->debug("xxxx    debug {}", n);
            console->info("xxxx     info {} thread_id {}", n, ThreadID());
            console->warn("xxxx     warn {}", n);
            console->error("xxxx    error {}", n);
            console->critical("xxxx critical {}", n);
            SPDLOG_INFO("SPDLOG_INFO {}\n", n);
        }
        spdlog::drop_all();
        return 1;
    } catch (const spdlog::spdlog_ex& ex) {
        std::cout << "spdlog exception: " << ex.what() << "\n";
    }
    return 0;
}

static int FileTest()
{
    //auto console = spdlog::stdout_color_mt("console");
    //auto console = spdlog::basic_logger_mt("basic_logger", "logs/basic.txt");
    auto console = spdlog::rotating_logger_mt("filer_logger", "logs/rotate.txt", 1024 * 5, 3);
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
        return 1;
    } catch (const spdlog::spdlog_ex& ex) {
        std::cout << "spdlog exception: " << ex.what() << "\n";
    }
    return 0;
}

int main()
{
    ConsoleTest();

    system("pause");
    return 0;
}

