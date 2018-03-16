#pragma once

#include <chrono>
#include <functional>

namespace zylib {

using namespace std::chrono;

template <typename T>
struct BasicTimer
{
	using Duration = T;

	template <typename D>
	BasicTimer(D d)
		: m_start()
		, m_expire(d)
	{
	}

	void update(Duration delta)
	{
		m_start += delta;
	}

	bool passed() const
	{
		return m_expire <= m_start;
	}

	void reset()
	{
		m_start = Duration{};
	}

	Duration remain() const
	{
		return m_expire - m_start;
	}

	Duration m_start;
	const Duration m_expire;
};

using TimingWheel = BasicTimer<milliseconds>;
using Delta = milliseconds;

using TimePoint = std::chrono::steady_clock::time_point;

TimePoint getSteadyTimePoint()
{
	return std::chrono::steady_clock::now();
}

Delta getDelta(TimePoint b, TimePoint e)
{
	return std::chrono::duration_cast<Delta>(e - b);
}


//////////////////////////////////////////////////////////////////////////
}

