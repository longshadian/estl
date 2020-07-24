#include <iostream>
#include <string_view>
#include <vector>
#include <string>
#include <utility>

#include "Common.h"

#include "../doctest/doctest.h"

namespace test_move
{

struct XString
{
public:
    XString(): m_() {}
    XString(const char* m) : m_(m) { }
    XString(const std::string& m) : m_(m) { }

    ~XString()
    {
    }

    XString(const XString& rhs) = default;
    XString& operator=(const XString& rhs) = default;

    XString(XString&& rhs)
        : m_(std::move(rhs.m_))
    {
        LogInfo << "XString move copy constructor";
    }

    XString& operator=(XString&& rhs)
    {
        LogInfo << "XString move assigment constructor";
        if (this != &rhs) {
            std::swap(m_, rhs.m_);
        }
        return *this;
    }

    std::ostream& operator<<(std::ostream& os) const
    {
        os << this->m_;
        return os;
    }

    bool operator==(const std::string& s) const
    {
        return m_ == s;
    }

    bool operator!=(const std::string& s) const
    {
        return (*this) != s;
    }

    std::string m_;
};

class X
{
public:
    X()
    {
    }

    ~X()
    {
    }

    std::string str_;
    XString xstr_;

    const std::string& get_str() const & { return str_; } 
    std::string& get_str() & { return str_; } 
    std::string&& get_str() && { return std::move(str_); }

    const XString& get_xstr() const & { return xstr_; } 
    XString& get_xstr() & { return xstr_; } 
    //XString&& get_xstr() && { return std::move(xstr_); }
    XString get_xstr() && { return std::move(xstr_); }
};

int TestMove()
{
    const std::string s = "std::string";
    const std::string xs = "xstring";
    X x;
    x.str_ = s;
    x.xstr_ = xs;

    auto s0 = x.get_str();
    CHECK(s0 == s);
    CHECK(x.get_str() == s);

    auto s1 = std::move(x).get_str();
    CHECK(s1 == s);
    CHECK(x.get_str() == "");

    auto xs0 = x.get_xstr();
    CHECK(xs0 == xs);
    CHECK(x.get_xstr() == xs);
    auto xs1 = std::move(x).get_xstr();
    CHECK(xs1 == xs);
    CHECK(x.get_xstr() == "");

    return 0;
}

void Process(const std::string& str)
{
    LogInfo << "left value: " << str;
}

void Process(std::string&& str)
{
    LogInfo << "right value: " << str;
}

template <typename T>
void DispatchProcess(T&& args)
{
    Process(std::forward<T>(args));
}

void TestMove2()
{
    std::string str = "abc";
    DispatchProcess(str);
    DispatchProcess(std::move(str));
}

void Test3()
{
    LogInfo << std::is_same_v<std::string, std::decay_t<std::string>>;
    LogInfo << std::is_same_v<std::string, std::decay_t<std::string&>>;
    LogInfo << std::is_same_v<std::string, std::decay_t<std::string&&>>;
    LogInfo << std::is_same_v<std::string, std::decay_t<const std::string>>;
    LogInfo << std::is_same_v<std::string, std::decay_t<const std::string&>>;
    LogInfo << std::is_same_v<std::string, std::decay_t<const std::string&&>>;
}

} // namespace test_move

#if 1
TEST_CASE("TestMove")
{
    using namespace test_move;
    PrintInfo("TestMove %d", 123);
    CHECK(TestMove() == 0);
    // TestMove2();
    // Test3();
}
#endif

