#include <iostream>
#include <string>
#include <string_view>
#include <cassert>

#include <fmt/core.h>
#include <fmt/ostream.h>

//#define FMT_STRING_ALIAS 1
#include <fmt/format.h>
#include <fmt/printf.h>

struct A 
{
    int32_t m_int { 10086 };

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
        std::string_view sv { str };
        const char* p = "abc";
        int32_t value = 123;
        A a {};

        std::string s = fmt::format("format {:d} {:sv} {1} {1}\n", p, 123, str);
        std::cout << s;
    } catch (const fmt::format_error& e) {
        std::cout << "exception: " << e.what() << "\n";
    }
}

void TestCompileTime()
{
    A a;
    // std::string s = fmt::format(FMT_STRING("format {} {}"), 1, "abc");
    fmt::print(FMT_STRING("format {1} {0} {1} {2}\n"), 1, "abc", a);
    //std::cout << s << "\n";
}

void Test2()
{
    fmt::memory_buffer buf;
    fmt::format_to(buf, "{}", 42); // replaces itoa(42, buffer, 10)

    fmt::memory_buffer buf1;
    fmt::format_to(buf1, "0x{:X}", 42); // replaces itoa(42, buffer, 16)

    std::cout << fmt::to_string(buf) << " " << fmt::to_string(buf1) << "\n";
}

int main()
{
    if (1) {
        // Test1();
        // TestCompileTime();
        Test2();
    }

    system("pause");
    return 0;
}
