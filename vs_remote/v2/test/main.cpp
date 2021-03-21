#include <cstring>

#include <string>
#include <iostream>
#include <chrono>
#include <thread>
#include <ctime>

#ifndef WIN32
#include <sys/time.h>
#endif // !WIN32

#include <boost/date_time/date.hpp>
#include <boost/date_time.hpp>
#include <boost/date_time/gregorian/gregorian.hpp> 
#include <boost/date_time/microsec_time_clock.hpp>
#include <boost/date_time/c_local_time_adjustor.hpp>

void fun()
{
    auto pt = boost::posix_time::microsec_clock::universal_time();
    std::cout << pt << "\n";
    std::cout << boost::date_time::c_local_adjustor<boost::posix_time::ptime>::utc_to_local(pt) << "\n";
}

inline
boost::posix_time::ptime utcToLocal(time_t t)
{
    return boost::date_time::c_local_adjustor<boost::posix_time::ptime>::utc_to_local(
        boost::posix_time::from_time_t(t));
}

std::string formatPTime(boost::posix_time::ptime t, const char* fmt)
{
    try {
        std::ostringstream ostm{};
        if (fmt) {
            // need new time_facet
            boost::posix_time::time_facet* facet = new boost::posix_time::time_facet(fmt);
            ostm.imbue(std::locale(std::locale(), facet));
        }
        ostm << t;
        return ostm.str();
    }
    catch (...) {
        return {};
    }
}

boost::posix_time::time_duration get_utc_offset() {
    using namespace boost::posix_time;

    // boost::date_time::c_local_adjustor uses the C-API to adjust a
    // moment given in utc to the same moment in the local time zone.
    typedef boost::date_time::c_local_adjustor<ptime> local_adj;

    const ptime utc_now = second_clock::universal_time();
    const ptime now = local_adj::utc_to_local(utc_now);

    return now - utc_now;
}

std::string get_utc_offset_string() {
    std::stringstream out;

    using namespace boost::posix_time;
    time_facet* tf = new time_facet();
    tf->time_duration_format("%+%H:%M");
    out.imbue(std::locale(out.getloc(), tf));

    out << get_utc_offset();

    return out.str();
}

void printD(boost::gregorian::date d)
{
    std::cout << "data: " << d << "\n";
    std::cout << d.year() << "-" << d.month() << "-" << d.day() << "\n";
}

void printTimeDuration(boost::posix_time::time_duration td)
{
    std::cout << "time_duration: " << td << "\n";
    std::cout << td.hours() << ":" << td.minutes() << ":" << td.seconds() << "\n";
    std::cout << td.total_seconds() << "\n";
    std::cout << td.total_milliseconds() << "\n";
    std::cout << td.total_microseconds() << "\n";
    std::cout << td.fractional_seconds() << "\n";
}

void ios8061(std::time_t t)
{
    boost::posix_time::ptime ptime = boost::posix_time::from_time_t(t);
    printD(ptime.date());
    printTimeDuration(ptime.time_of_day());

    std::cout << "-----------\n";
    std::cout << get_utc_offset_string() << "\n";
}

int fun1()
{
    std::cout << "universal_time:\n";
    {
        auto ptime = boost::posix_time::microsec_clock::universal_time();
        printD(ptime.date());
        printTimeDuration(ptime.time_of_day());
    }

    std::cout << "\n local_time:\n";
    {
        auto ptime = boost::posix_time::microsec_clock::local_time();
        printD(ptime.date());
        printTimeDuration(ptime.time_of_day());
    }

    std::cout << "\n time_t:\n";
    {
        auto ptime = boost::posix_time::from_time_t(std::time(nullptr));
        printD(ptime.date());
        printTimeDuration(ptime.time_of_day());
    }

#ifndef WIN32
    std::cout << "\n gettimeofday:\n";
    {
        struct timeval tv {};
        gettimeofday(&tv, nullptr);
        boost::posix_time::ptime ptime = boost::posix_time::from_time_t(tv.tv_sec);
        ptime += boost::posix_time::time_duration{ 0,0,0, tv.tv_usec };
        printD(ptime.date());
        printTimeDuration(ptime.time_of_day());
    }

#endif // !WIN32

    std::cout << "\n utcToLocal:\n";
    {
        auto tnow = std::time(nullptr);
        auto ptime = utcToLocal(tnow);
        printD(ptime.date());
        printTimeDuration(ptime.time_of_day());

        auto tend = boost::posix_time::to_time_t(ptime);
        std::cout << "now: " << tnow
            << " now_ex: " << tend
            << " delta: " << tend - tnow
            << "\n";
    }

    {
        auto tnow = boost::posix_time::microsec_clock::universal_time();
        std::this_thread::sleep_for(std::chrono::seconds{ 100 });
        auto tnow2 = boost::posix_time::microsec_clock::universal_time();
        boost::posix_time::time_duration td = tnow2 - tnow;
        std::cout << td.total_seconds()
            << " " << td.total_milliseconds()
            << " " << td.total_microseconds()
            << " " << td.seconds()
            << "\n";
    }

    return 0;
}

int main()
{
    std::time_t t = std::time(nullptr);
    try {
        auto pt = boost::date_time::c_local_adjustor<boost::posix_time::ptime>::utc_to_local(boost::posix_time::from_time_t(t));
        std::ostringstream ostm{};
        // need new time_facet
        const char* fmt = "%Y-%m-%d %H:%M:%S";
        boost::posix_time::time_facet* facet = new boost::posix_time::time_facet(fmt);
        ostm.imbue(std::locale(std::locale(), facet));
        ostm << pt;
        std::cout << ostm.str() << "\n";
    }
    catch (...) {
        return {};
    }
}
