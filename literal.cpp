#include <iostream>

#include <chrono>
#include <thread>

#include "Timer.h"

#include "msg_gbmj.pb.h"

void fun()
{
    pt::zj::obj_test test{};
    test.set_val_byte();
    test.set_val_string();

}

constexpr
long long operator "" _W(unsigned long long v)
{
	return v * 10000;
}

constexpr
std::chrono::seconds operator "" _s(unsigned long long v)
{
    return std::chrono::seconds{v};
}

int main()
{
    auto v = 20_s;

	zylib::TimingWheel t{ std::chrono::milliseconds{1450} };

	auto t1 = std::chrono::system_clock::now();
	std::cout << "time1:" << t1.time_since_epoch().count() << "\n";
	auto now1 = zylib::getSteadyTimePoint();

	int n = 0;
	while (true) {
		auto now2 = zylib::getSteadyTimePoint();
		auto delta = zylib::getDelta(now1, now2);
		t.update(delta);
		std::cout << "sleep:" << delta.count() << " remain:" << t.remain().count() << "\n";
		if (t.passed()) {
			if (n == 0) {
				t.reset();
				++n;
			} else {
				break;
			}
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		now1 = now2;
	}

	auto t2 = std::chrono::system_clock::now();
	std::cout << "time2:" << t2.time_since_epoch().count() << "\n";
	std::cout << "cost milliseconds:" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "\n";
	return 0;
}
