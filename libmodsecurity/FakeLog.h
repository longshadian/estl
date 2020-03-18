#pragma once

#include <cstdio>
#include <cstdarg>

#define FAKELOG_DEBUG(fmt, ...) printf("[%4d] [DEBUG]  " fmt "\n", __LINE__, __VA_ARGS__)
#define FAKELOG_WARN(fmt, ...)  printf("[%4d] [WARN ]  " fmt "\n", __LINE__, __VA_ARGS__)

