#include <boost/date_time.hpp>
#include <boost/date_time/c_local_time_adjustor.hpp>
#include <boost/filesystem.hpp>

#include <iostream>
#include <fstream>

boost::posix_time::ptime utcToLocal(time_t t)
{
    return boost::date_time::c_local_adjustor<boost::posix_time::ptime>::utc_to_local(
        boost::posix_time::from_time_t(t));
}

std::string formatPTime(boost::posix_time::ptime t, const char* fmt = nullptr)
{
    try {
        std::ostringstream ostm{};
        if (fmt) {
            boost::posix_time::time_facet* facet = new boost::posix_time::time_facet(fmt);
            ostm.imbue(std::locale(std::locale(), facet));
        }
        ostm << t;
        return ostm.str();
    }
    catch (...) {
        return{};
    }
}


void saveZJRound_Disk(time_t t, std::string file_name, std::string content)
{
    auto local_t = utcToLocal(t);
    auto dir_name = formatPTime(local_t, "%Y%m%d-%H");
    if (dir_name.empty()) {
        std::cout << "formatPTime faile. " << t;
        return;
    }

    try {
        std::string server_path = "/home/cgy/work/zylib/test/boost";
        boost::filesystem::path full_path{ server_path + '/' + dir_name };
        if (boost::filesystem::exists(full_path)) {
            if (!boost::filesystem::is_directory(full_path)) {
                std::cout << "path isn't directory: " << full_path.string();
                return;
            }
        } else {
            boost::filesystem::create_directory(full_path);
            std::string full_path_file = full_path.string() + "/" + file_name;
            std::ofstream ofsm{ full_path_file.c_str(), std::ios::binary };
            if (!ofsm) {
                std::cout << "can't open " << full_path_file;
                return;
            }
            ofsm << "12345";
        }
    } catch (const boost::filesystem::filesystem_error& e) {
        std::cout<< "boost filesystem exception " << e.what();
        return;
    }
}

void testSaveDisk()
{
    auto t = std::time(nullptr);
    auto local_t = utcToLocal(t);
    std::cout << "local_t " << formatPTime(local_t) << "\n";
    std::cout << local_t << "\n";

    saveZJRound_Disk(t, "xxx", "123456");
}

void testDay()
{
    auto p1 = boost::posix_time::from_time_t(1498623492);
    auto p2 = boost::posix_time::from_time_t(1498537092);

    std::cout << p1.date() << "\n";

    std::cout << (p1.date() == p2.date()) << "\n";
}

int main()
{
    testDay();
    return 0;
}
