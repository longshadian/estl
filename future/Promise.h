#pragma once


#include <future>
#include <type_traits>

#include "Future.h"

namespace zylib {

template<typename T>
class Promise
{
public:
    Promise()
        : m_promise(new std::promise<T>())
    {
    }

    ~Promise() = default;

    Promise(const Promise& rhs) = delete;
    Promise& operator=(const Promise& rhs) = delete;

    Promise(Promise&& rhs)
        : m_promise(std::move(rhs.m_promise))
    {
    }
    Promise& operator=(Promise&& rhs)
    {
        if (this != &rhs) {
            std::swap(m_promise, rhs.m_promise);
        }
        return *this;
    }

    void setValue(T&& val)
    {
        static_assert(!std::is_same<T, void>::value, "T is void!");
        m_promise->set_value(std::move<T>(val));
    }

    Future<T> getFuture() 
    { 
        static_assert(!std::is_same<T, void>::value, "T is void!");
        return zylib::Future<T>(m_promise.get_future()); 
    }
private:
    std::unique_ptr<std::promise<T>> m_promise;
};


}

