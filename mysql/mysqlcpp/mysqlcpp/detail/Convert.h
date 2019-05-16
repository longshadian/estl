#pragma once

#include <type_traits>
#include <iostream>
#include <string_view>

#include <boost/convert.hpp>
#include <boost/convert/stream.hpp>

namespace mysqlcpp {
namespace detail {

template <typename T>
struct Convert
{
    static_assert(std::is_arithmetic<T>::value && !std::is_same<bool, T>::value, "T must be arithmetic and can't be bool!");

    static T cvt_noexcept(const char* p)
    {
        try {
            return boost::convert<T>(p, boost::cnv::cstream{}).value();
        } catch (...) {
            return T{};
        }
    }

    static T cvt_noexcept(const std::string& p)
    {
        try {
            return boost::convert<T>(p, boost::cnv::cstream{}).value();
        } catch (...) {
            return T{};
        }
    }

    static T cvt_noexcept(std::string_view sv)
    {
        try {
            return boost::convert<T>(sv, boost::cnv::cstream{}).value();
        } catch (...) {
            return T{};
        }
    }

    static T cvt(const char* p)
    {
        return boost::convert<T>(p, boost::cnv::cstream{}).value();
    }

    static T cvt(const std::string& p)
    {
        return boost::convert<T>(p, boost::cnv::cstream{}).value();
    }

    static T cvt(std::string_view sv)
    {
        return boost::convert<T>(sv, boost::cnv::cstream{}).value();
    }
};

} // detail
} // mysqlcpp
