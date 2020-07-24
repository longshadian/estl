#include <iostream>
#include <string_view>
#include <vector>
#include <string>
#include <utility>

#include "Common.h"

#include "../doctest/doctest.h"

namespace test_string_view
{

std::string_view substr_by_range(std::string_view sv, std::size_t b, std::size_t e)
{
    return sv.substr(b, e-b);
}

static
void TestFind()
{
std::string_view s = R"(
[client 192.168.106.1] 
ModSecurity: Access denied with code 403 (phase 2).
Matched "Operator `Rx' with parameter `^[\d.:]+$' against variable `REQUEST_HEADERS:Host' (Value: `192.168.106.200:11101' )
[file "conf / rules / REQUEST - 920 - PROTOCOL - ENFORCEMENT.conf"]
[line "722"] [id "920350"] [rev ""] 
[msg "Host header is a numeric IP address"] 
[data "192.168.106.200 : 11101"] [severity "4"] [ver "OWASP_CRS / 3.2.0"] [maturity "0"] [accuracy "0"] 
[tag "application - multi"] [tag "language - multi"] [tag "platform - multi"] 
[tag "attack - protocol"] [tag "paranoia - level / 1"] [tag "OWASP_CRS"] 
[tag "OWASP_CRS / PROTOCOL_VIOLATION / IP_HOST"] [tag "WASCTC / WASC - 21"] 
[tag "OWASP_TOP_10 / A7"] [tag "PCI / 6.5.10"] 
[hostname "0.0.0.0"] [uri " / "] [unique_id "158901782897.422198"] [ref "o0, 21v21, 21"]
)";

    {
        std::string_view b = "[file \"";
        std::string_view e = "\"]";
        auto pb = s.find(b);
        auto pe = s.find(e);
        std::string sv = std::string(substr_by_range(s, pb +b.length(), pe));
        PrintInfo("sv: ==>%s<==", sv.c_str());
    }
}

static
void Test1()
{
#if 0
    // 以下代码会导致编译期出错，string_view不提供变动接口
    std::string s = "abc";
    std::string_view sv1 = s;
    *sv1.begin() = '1';
#endif
}

} // namespace test_string_view

//#define USE_TEST

#if defined (USE_TEST)
TEST_CASE("TestStringView")
{
    LogInfo << "test string view";
    try {
        test_string_view::TestFind();
    } catch (const std::exception& e) {
        LogWarn << e.what();
        CHECK(false);
    }
}
#endif


