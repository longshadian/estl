#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <any>

namespace plugin
{
namespace a
{

class GlobalInstance
{

public:
    GlobalInstance();
    ~GlobalInstance();

    static GlobalInstance* get();
    static void cleanup();

    std::string execute();

};

} // namespace a
} // namespace plugin


