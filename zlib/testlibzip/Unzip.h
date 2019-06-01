#pragma once

#include <string>

struct zip;

class Unzip
{
public:
    Unzip();
    ~Unzip();

    bool Uncompress(std::string zip_path, std::string uncompress_path);
private:
    struct zip* m_zipfile;
};
