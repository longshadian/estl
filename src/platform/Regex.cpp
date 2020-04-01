#include <regex>
#include <string>
#include <iostream>


#include "Common.h"

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
        WPrintf("exception: code: %d reason: %s", e.code(),  e.what());
        return -1;
    }
}

} // namespace test_regex


#include "../doctest/doctest.h"
#define USE_TEST

#if defined (USE_TEST)
TEST_CASE("TestRegex NAME")
{
    CHECK(test_regex::CheckYYYYMMDDD() == 0);
}

#endif
