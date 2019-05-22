#pragma once

#include <string>

class Service
{
public:
    Service() = default;
    virtual ~Service() = default;

    virtual std::string GetName() = 0;
    virtual int Add(int a, int b) = 0;
};


struct Import
{
    int m_version;
};

struct Export
{
    int         m_version;
    Service*    m_s;
};
