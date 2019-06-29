#include <iostream>

//#include <unistd.h>

#include <array>
#include <cstdarg>
#include <ctime>

#include "Log.h"

/*
void snprintfText()
{
    std::array<char, 3> buffer = {0};
    auto n = snprintf(buffer.data(), buffer.size(), "%s", "1234");
    std::cout << n << std::endl;

    std::cout << buffer.data() << std::endl;

    struct tm cur_tm;
    time_t t = time(nullptr);

    localtime_s(&cur_tm, &t);

    printf("%d/%d/%d %d:%d:%d\n", 1900+ cur_tm.tm_year, cur_tm.tm_mon + 1, cur_tm.tm_mday,
        cur_tm.tm_hour, cur_tm.tm_min, cur_tm.tm_sec);
}
*/

int main()
{
    zylib::Logger::init("/home/cgy/work/test/log/logouts");
    //zylib::ServerLoger::printLogFormat("test", zylib::LOG_DEBUG, "123456789012345678901");

    int n = 0;
    while (true) {
        std::string s;
        for (int i = 0; i != n; ++i) {
            auto c = std::to_string(i%10);
            s.append(c);
        }
        zylib::Logger::printLogFormat("test", zylib::LOG_DEBUG, "%s", s.c_str());
        ++n;
        if (n > zylib::Logger::MAX_LOG_MSG_LENGTH) {
            n = 0;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    system("pause");
    return 0;
}
