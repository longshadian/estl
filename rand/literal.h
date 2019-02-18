#pragma once


unsigned long long operator"" _w(unsigned long long v)
{
    return v * 10000;
};

unsigned long long operator"" _k(unsigned long long v)
{
    return v * 1000;
};
