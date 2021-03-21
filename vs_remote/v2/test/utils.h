#pragma once

#include <string>
#include <random>

bool DumpToFile(const std::string& fname, std::string_view content);
int RandInt(int b, int e);

