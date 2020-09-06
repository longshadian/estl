#pragma once

#include <thread>

#include "easylogging++.h"

class Easylogging
{
public:
    Easylogging();
    ~Easylogging();

    int Init();
    int InitFromConf(const std::string& filename);

    static void LogRollOut(const char* filename, std::size_t size);

private:
};

