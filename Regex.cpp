#include <regex>
#include <string>
#include <iostream>

#include <boost/convert.hpp>
#include <boost/convert/stream.hpp>


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


int main()
{
    //fun(); return 0;

    //std::string pattern = R"((\d+)-(\d+)-(\d+) (\d+):(\d+):(\d+)\.(\d+))";

    std::string pattern = R"((\d+)-(\d+)-(\d+) (\d+):(\d+):(\d+)\.(\d+))";

    //std::string pattern = "(\\d+)-(\\d+)-(\\d+)";
    //std::string pattern = R"xx(\w+)xx";
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
    } catch (std::regex_error e) {
        std::cout << e.what() << '\t' << e.code() << std::endl;
    }


    try {
        std::string s = "123";
        std::cout << boost::convert<int>(s, boost::cnv::cstream{}).value() << "\n";
    } catch (...) {
        std::cout << "boost exception\n";
    }

    return 0;
}
