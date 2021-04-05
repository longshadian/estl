#include <cassert>
#include <vector>
#include <iostream>
#include <string>

#include "Version.h"

static void TestVersion()
{
    std::cout << MAIN_VERSION_MAJOR   << "\n";
    std::cout << MAIN_VERSION_MINOR   <<"\n";
    std::cout << MAIN_VERSION_PATCH  <<"\n";
    std::cout << MAIN_VERSION_TWEAK  << "\n";
    std::cout << MAIN_VERSION << "\n";
}

int main(int argc, char** argv)
{
    TestVersion();
    return 0;
}
