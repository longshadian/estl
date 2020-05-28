#include <regex>
#include <string>
#include <array>
#include <iostream>

#include "Common.h"

#include "../doctest/doctest.h"

namespace test_regex 
{

void fun()
{
    std::string s = "a2017-9-2 11:11:11";
    std::string pattern = "(\\d+)-(\\d+)-(\\d+)";
    std::regex email_regex(pattern);

    auto words_begin =
        std::sregex_iterator(s.begin(), s.end(), email_regex);
    auto words_end = std::sregex_iterator();

    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::smatch match = *i;
        std::string match_str = match.str();
        std::cout << "xxxx:  " << match_str << '\n';
        for (size_t i = 0; i != match.size(); ++i) {
            std::cout << "\t\t" << match[i] << "\n";
        }
        std::cout << match.prefix() << "\n";
        std::cout << match.suffix() << "\n";
    }
}

int fun2()
{
    try {
        std::string prf = "access";
        std::string pattern = prf + "_(\\d+).log.(\\d+)";
        std::regex rgx(pattern);
        std::string s = "access_19090901.log.1";

        std::smatch results{};
        if (std::regex_search(s, results, rgx)) {
            for (size_t i = 0; i != results.size(); ++i) {
                if (i == 0)
                    continue;
                LogInfo << "\t" << results[i] << "\t" << results[i].length() << "\n";
            }
        }
        return 0;
    } catch (const std::regex_error & e) {
        PrintWarn("exception: code: %d reason: %s", e.code(),  e.what());
        return -1;
    }
}

int CheckYYYYMMDDD()
{
    try {
        std::string pattern = R"((\d+)-(\d+)-(\d+) (\d+):(\d+):(\d+))";
        std::regex email_regex(pattern);
        std::smatch results{};
        std::string test_email_str = "2017-9-27 11:11:11";
        if (std::regex_search(test_email_str, results, email_regex)) {
            std::cout << results.str() << std::endl;
            std::cout << results.size() << "\n";
            for (size_t i = 0; i != results.size(); ++i) {
                if (i == 0)
                    continue;
                std::cout << "\t\t" << results[i] << "\t" << results[i].length() << "\n";
            }
        }
        return 0;
    } catch (std::regex_error e) {
        PrintWarn("exception: code: %d reason: %s", e.code(),  e.what());
        return -1;
    }
}

int fun3()
{
    try {
        std::array<int, 4> arr;
        std::string pattern = R"(^(\d+)\.(\d+)\.(\d+)\.(\d+)$)";
        std::regex ipv4_regex(pattern);

        std::vector<std::string> str_vec =
        {
            "127.0.0.1",
            "127.0.0.-1",
            "127.0.0.10001",
            "127.0.0001.10001",
            "10000.a.0.1",
            "-10000.a.0.1",
            "10000.0.0.1.1",
            ".a.0.0.1.1",
        };
        for (const auto& s : str_vec) {
            arr.fill(0);
            std::smatch results{};
            if (std::regex_search(s, results, ipv4_regex)) {
                for (size_t i = 0; i != results.size(); ++i) {
                    if (i == 0)
                        continue;
                    arr[i - 1] = atoi(results[i].str().c_str());
                }
                std::cout << "match " << s << " " << arr[0] << " " << arr[1] << " " << arr[2] << " " << arr[3] << "\n";
            }
            else {
                std::cout << "dont match " << s << "\n";
            }
        }
        return 1;
    }
    catch (const std::exception & e) {
        std::cout << "exception: " << e.what() << "\n";
    }
    return -1;
}

int fun4()
{
    try {
        std::array<std::string, 4> arr;
        std::string pattern = R"(^(\w+|/)\.(\w+)\.(\w+)\.(\w+)$)";
        std::regex ipv4_regex(pattern);

        std::vector<std::string> str_vec =
        {
            "127.0.0.1",
            "127.0.0.-1",
            "127.0.0.10001",
            "127.0.0001.10001",
        };
        for (const auto& s : str_vec) {
            arr.fill("");
            std::smatch results{};
            if (std::regex_search(s, results, ipv4_regex)) {
                for (size_t i = 0; i != results.size(); ++i) {
                    if (i == 0)
                        continue;
                    arr[i - 1] = results[i].str();
                }
                std::cout << "match         " << s << "         " << arr[0] << " " << arr[1] << " " << arr[2] << " " << arr[3] << "\n";
            }
            else {
                std::cout << "dont match    " << s << "\n";
            }
        }
        return 1;
    }
    catch (const std::exception & e) {
        std::cout << "exception: " << e.what() << "\n";
    }
    return -1;
}


} // namespace test_regex


//#define USE_TEST

#if defined (USE_TEST)
TEST_CASE("TestRegex NAME")
{
    LogInfo << __FILE__;
    //CHECK(test_regex::CheckYYYYMMDDD() == 0);
    CHECK(test_regex::fun2() == 0);
}

#endif
