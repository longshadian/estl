#pragma once

#include <string>

class Utilities
{
public:
    static std::string ToUpperCase(std::string s);

    static bool WriteFile(const std::string& fpath, const void* data, std::size_t length);
};
