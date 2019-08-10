#pragma once

#include <string>

namespace base64 {

std::string Encode(const std::string& s);
std::string Encode(unsigned char const*, unsigned int len);
std::string Decode(std::string const& s);

} // namespace base64 
