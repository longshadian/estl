#include <iostream>
#include <chrono>

struct ServerRunTime
{
    int64_t m_day{ 0 };
    int64_t m_hour{ 0 };
    int64_t m_minute{ 0 };
    int64_t m_total_minute{ 0 };
};


int main()
{
    ServerRunTime run_time{};
    auto m_start_time = std::chrono::system_clock::now();

    auto tnow = m_start_time 
        + std::chrono::hours(49) 
        + std::chrono::minutes(120)
        + std::chrono::seconds(321)
        ;

    run_time.m_total_minute = std::chrono::duration_cast<std::chrono::minutes>(tnow - m_start_time).count();
    auto remain_minute = run_time.m_total_minute;
    run_time.m_day = remain_minute/(24*60);
    remain_minute = remain_minute%(24*60);
    run_time.m_hour = remain_minute/60;
    run_time.m_minute = remain_minute%60;

    printf("%d %d %d %d\n", (int)run_time.m_day, (int)run_time.m_hour, (int)run_time.m_minute, (int)run_time.m_total_minute);

    std::cout << (run_time.m_day*24*60 + run_time.m_hour*60 + run_time.m_minute == run_time.m_total_minute) << "\n";

    return 0;
}
