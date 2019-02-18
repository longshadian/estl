#pragma once

#include <cassert>
#include <vector>

#include "RedisCpp.h"

#define TEST assert

void pout(const std::vector<rediscpp::Buffer>& v);
void pout(const rediscpp::Buffer& v);

void pout(const rediscpp::BufferArray& v);
void poutArrayCell(const rediscpp::BufferArray& v);
