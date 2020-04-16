#pragma once

#include <cstdint>
#include <string_view>
#include <string>
#include <vector>

namespace plugin
{
namespace a
{

bool read_file(const std::string& file_name, std::string& content, std::string& emsg);

} // namespace a
} // namespace plugin

