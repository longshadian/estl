#pragma once

//#include <cassert>
#include <type_traits>
#include "ByteBuffer.h"

#define META(...) \
auto Meta()->decltype(std::tie(__VA_ARGS__)){return std::tie(__VA_ARGS__);} \
auto Meta() const ->decltype(std::tie(__VA_ARGS__)){return std::tie(__VA_ARGS__);}

template <typename T, std::size_t N>
struct ByteBufferSerializer
{
    static void T2BB(ByteBuffer& b, const T & t)
    {
        ByteBufferSerializer<T, N-1>::T2BB(b, t);
        b << std::get<N-1>(t);
        //std::cout << std::get<N-1>(t) << std::endl;
    }

    static void BB2T(ByteBuffer& b, const T& t) 
    {
        ByteBufferSerializer<T, N-1>::BB2T(b, t);
        b >> std::get<N-1>(t);
        //std::cout << std::get<N-1>(t) << std::endl;
    }
};

template <typename T>
struct ByteBufferSerializer<T, 1> 
{
    static void T2BB(ByteBuffer& b, const T& t)
    {
        b << std::get<0>(t);
        //std::cout << std::get<0>(t) << std::endl;
    }

    static void BB2T(ByteBuffer& b, const T& t)
    {
        b >> std::get<0>(t);
        //std::cout << std::get<0>(t) << std::endl;
    }
};

template <typename... Args>
void BBSerializeTo(ByteBuffer& b, const std::tuple<Args...>& t)
{
    ByteBufferSerializer<decltype(t), sizeof...(Args)>::BB2T(b, t);
}

template <typename... Args>
void BBSerializeFrom(ByteBuffer& b, const std::tuple<Args...>& t)
{
    ByteBufferSerializer<decltype(t), sizeof...(Args)>::T2BB(b, t);
}

template<typename T>
inline
ByteBuffer& operator>>(ByteBuffer& b, T& a)
{                                            
    BBSerializeTo(b, a.Meta());                 
    return b;                                  
}                                         

template<typename T>
inline
ByteBuffer& operator<<(ByteBuffer& b, const T& a)
{                                               
    BBSerializeFrom(b, a.Meta());                  
    return b;                                     
}
