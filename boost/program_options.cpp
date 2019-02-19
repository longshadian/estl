#include <thread>
#include <chrono>
#include <iostream>

#include <boost/program_options.hpp>

namespace bpo = boost::program_options;

int cmdFun(int argc, char** argv)
{
    bpo::options_description desc("Allowed options");
    try {
        desc.add_options()
            ("help",        "help message")
            ("update1",     "callc invite_id from user id. save them to database")
            ("update12",    "select invite_id from database. save them to redis")
            ;

        bpo::variables_map vm;
        bpo::store(bpo::parse_command_line(argc, argv, desc), vm);
        bpo::notify(vm);
        if (vm.count("update")) {
            std::cout << "match update\n";
            return 0;
        }
        if (vm.count("update12")) {
            std::cout << "match save\n";
            return 0;
        }
        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 0;
        }
        std::cout << desc << "\n";
    } catch (const std::exception& e) {
        std::cout << "unknown options: " << e.what() << "\n";
        std::cout << desc << "\n";
    }
    return 0;
}

int main(int argc, char** argv)
{
    auto ret = cmdFun(argc, argv);
    //std::cout << "xxxx: " << ret << "\n";

    std::map<std::string, std::string> xxx{};
    xxx.insert({"aaa", "123"});
    xxx.insert({"abc", "1234"});

    std::cout << xxx.count("aaa") << "\n";

    return ret;
}
