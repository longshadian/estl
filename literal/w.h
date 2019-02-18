#pragma once

constexpr
int64_t operator "" _w(unsigned long long v)
{
	return v * 10000;
}
