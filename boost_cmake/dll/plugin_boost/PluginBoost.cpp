#include "plugin_exe/Service.h"

#include <iostream>

#include <boost/dll.hpp>

class MyService : public Service
{
public:
    MyService() = default;
    virtual ~MyService() = default;

    virtual std::string GetName() override
    {
        return "plugin_boost";
    }

    virtual int Add(int a, int b) override
    {
        return a + b;
    }
};

MyService g_s;
Export g_exp;

extern "C"
{
    BOOST_SYMBOL_EXPORT 
    Export* ExportAPI(Import* imp)
    {
        std::cout << "api import: " << imp->m_version << "\n";
        g_exp.m_version = 100;
        g_exp.m_s = &g_s;
        return &g_exp;
    }
}
