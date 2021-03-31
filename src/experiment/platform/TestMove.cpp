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

struct A
{
    A(const A&) = delete;
    A& operator=(const A&) = delete;

    A(A&&) = default;
    A& operator=(A&&) = default;

    int a{};
};

struct A1
{
    int a;
};

struct A2
{
    A2() = default;
    int a;
};

struct A3
{
    A3() = default;
    int a{};
};

struct B
{
    B() = default;

    B(const B&) { LogInfo << "B copy constructor"; }
    B& operator=(const B&) { LogInfo << "B copy constructor"; return *this; }

    B(B&&) { LogInfo << "B move constructor"; }
    B& operator=(B&&) { LogInfo << "B move assignment constructor"; return *this; }
};

struct B1
{
    B1(const B1&) = default;
    B1& operator=(const B1&) = default;

    B1(B1&&) = delete;
    B1& operator=(B1&&) = delete;
};

struct C
{
    A a{};
    A1 a1;
    A2 a2;
    A3 a3;
    B b{};
    //B1 b1{};
    int i{};
};

void Test4()
{
    C c1{};
    C c2{};
    c2 = std::move(c1);
    C c3{std::move(c2)};

    LogInfo << "A   is trivial " << std::is_trivial<A>::value;
    LogInfo << "A1  is trivial " << std::is_trivial<A1>::value;
    LogInfo << "A2  is trivial " << std::is_trivial<A2>::value;
    LogInfo << "A3  is trivial " << std::is_trivial<A3>::value;
    LogInfo << "B   is trivial " << std::is_trivial<B>::value;
    LogInfo << "B1  is trivial " << std::is_trivial<B1>::value;
    LogInfo << "C   is trivial " << std::is_trivial<C>::value;

    LogInfo << "A   is is_standard_layout " << std::is_standard_layout<A>::value;
    LogInfo << "A1  is is_standard_layout " << std::is_standard_layout<A1>::value;
    LogInfo << "A2  is is_standard_layout " << std::is_standard_layout<A2>::value;
    LogInfo << "A3  is is_standard_layout " << std::is_standard_layout<A3>::value;
    LogInfo << "B   is is_standard_layout " << std::is_standard_layout<B>::value;
    LogInfo << "B1  is is_standard_layout " << std::is_standard_layout<B1>::value;
    LogInfo << "C   is is_standard_layout " << std::is_standard_layout<C>::value;

    LogInfo << "A   is pod " << std::is_pod<A>::value;
    LogInfo << "A1  is pod " << std::is_pod<A1>::value;
    LogInfo << "A2  is pod " << std::is_pod<A2>::value;
    LogInfo << "A3  is pod " << std::is_pod<A3>::value;
    LogInfo << "B   is pod " << std::is_pod<B>::value;
    LogInfo << "B1  is pod " << std::is_pod<B1>::value;
    LogInfo << "C   is pod " << std::is_pod<C>::value;

}

} // namespace test_move

#if 1
TEST_CASE("TestMove")
{
    using namespace test_move;
    PrintInfo("TestMove %d", 123);
    //CHECK(TestMove() == 0);
    // TestMove2();
    // Test3();
    Test4();
}
#endif

