#pragma once

#include <cstring>
#include <string>
#include <map>
#include <vector>
#include <memory>
#include <algorithm>
#include <iterator>
#include <sstream>

#include "zylib/ZylibDefine.h"

namespace zylib {

std::vector<std::string> stringSplit(const std::string& s, char c);

//×Ö·û´®Ìæ»»,×Ö·û´®strÖÐµÄsrc×Ö·ûÌæ»»³Édest,·µ»ØÌæ»»¸öÊý
size_t stringReplace(std::string* str, char src, char dest);

template <class RandomAccessIterator>
void linear_random_shuffle(RandomAccessIterator first, RandomAccessIterator last)
{
    typename std::iterator_traits<RandomAccessIterator>::difference_type n = (last - first);
    if (n <= 0)
        return;
    while (--n) {
        std::swap(first[n], first[rand() % (n + 1)]);
    }
}

template <typename T>
void bzero(T* t)
{
    static_assert(std::is_pod<T>::value, "T must be pod!");
    std::memset(t, 0, sizeof(T));
}

template<typename T>
const uint8_t* ReadInteger(const uint8_t* src, size_t len, T* out)
{
    static_assert(std::is_integral<T>::value, "T must be integral");
    const size_t sizeof_out = sizeof(T);
    if (sizeof_out > len)
        return nullptr;
    std::memcpy(out, src, sizeof_out);
    return src += sizeof_out;
}

ZYLIB_EXPORT const uint8_t* ReadCString(const uint8_t* src, size_t len, std::string* out);

//////////////////////////////////////////////////////////////////////////
} // zylib
