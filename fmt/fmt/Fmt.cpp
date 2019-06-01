#include <iostream>
#include <string>
#include <string_view>

//#include <fmt/core.h>
#include <fmt/ostream.h>

//#define FMT_STRING_ALIAS 1
#include <fmt/format.h>

struct A
{
    int32_t m_int{10086};

    friend std::ostream& operator<<(std::ostream& ostm, const A& a) 
    {
        ostm << a.m_int;
        return ostm;
    }
};

void Test1()
{
    fmt::print("print {}\n", 123);
    try {
        std::string str = "Hello World";
        std::string_view sv{ str };
        const char* p = "abc";
        int32_t value = 123;
        A a{};

        std::string s = fmt::format("format {:d} {:sv} {1} {1}\n", p, 123, str, fmt::arg{"sv", sv} sv, a);
        std::cout << s;
    } catch (const fmt::format_error& e) {
        std::cout << "exception: " << e.what() << "\n";
    }
}

int main()
{
    if (1) {
        Test1();
    }

    system("pause");
    return 0;
}
