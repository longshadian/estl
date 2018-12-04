#pragma once

//#include <cassert>
#include <type_traits>
#include "ByteStream.h"

#define META(...) \
auto Meta()->decltype(std::tie(__VA_ARGS__)){return std::tie(__VA_ARGS__);} \
auto Meta() const ->decltype(std::tie(__VA_ARGS__)){return std::tie(__VA_ARGS__);}

template <typename T, std::size_t N>
struct ByteStreamSerializer
{
    static void T2BB(ByteStream& bs, const T & t)
    {
        ByteStreamSerializer<T, N-1>::T2BB(bs, t);
        bs.Write(std::get<N-1>(t));
        //std::cout << std::get<N-1>(t) << std::endl;
    }

    static void BB2T(ByteStream&bs, const T& t) 
    {
        ByteStreamSerializer<T, N-1>::BB2T(bs, t);
        bs.Read(&std::get<N-1>(t));
        //std::cout << std::get<N-1>(t) << std::endl;
    }
};

template <typename T>
struct ByteStreamSerializer<T, 1> {
    static void T2BB(ByteStream& bs, const T& t)
    {
        bs.Write(std::get<0>(t));
        //std::cout << std::get<0>(t) << std::endl;
    }

    static void BB2T(ByteStream& bs, const T& t)
    {
        bs.Read(&std::get<0>(t));
        //std::cout << std::get<0>(t) << std::endl;
    }
};

template <typename... Args>
void BBSerializeTo(ByteStream& bs, const std::tuple<Args...>& t)
{
    ByteStreamSerializer<decltype(t), sizeof...(Args)>::BB2T(bs, t);
}

template <typename... Args>
void BBSerializeFrom(ByteStream& bs, const std::tuple<Args...>& t)
{
    ByteStreamSerializer<decltype(t), sizeof...(Args)>::T2BB(bs, t);
}

template<typename T>
inline
ByteStream& operator>>(ByteStream& bs, T& a)
{
    BBSerializeTo(bs, a.Meta());                 
    return bs;                                  
}                                         

template<typename T>
inline
ByteStream& operator<<(ByteStream& b, const T& a)
{                                               
    BBSerializeFrom(b, a.Meta());                  
    return b;                                     
}
