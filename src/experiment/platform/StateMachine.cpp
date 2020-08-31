#include <cstddef>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <chrono>
#include <thread>
#include <type_traits>

#include "../doctest/doctest.h"
#include "Common.h"

namespace test_state
{

typedef int (*update_proc)(void*);

int state_p1(void*)
{
    LogInfo << "state_p1";
    return 2;
}

int state_p2(void*)
{
    static int n = 0;
    if (++n < 3) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        LogInfo << "state_p2";
        return 2;
    }
    LogInfo << "state_p3";
    return 3;
}

int state_p3(void*)
{
    return -1;
}

update_proc get_proc(int s)
{
    switch (s) {
    case 1 : return state_p1;
    case 2 : return state_p2;
    case 3 : return state_p3;
    default: return nullptr;
    }
}


struct state_machine
{
    int state_;
    update_proc proc_;
};



void Test1()
{
    state_machine sm;
    sm.state_ = 1;
    sm.proc_ = get_proc(sm.state_);

    while (1) {
        int s = sm.proc_(nullptr);
        if (s != sm.state_) {
            sm.state_ = s;
            sm.proc_ = get_proc(s);
        }
        if (!sm.proc_)
            break;
    }
}
    
} // namespace test_state

#define USE_TEST
#if defined (USE_TEST)
TEST_CASE("TestStateMachine")
{
    LogInfo << "TestStateMachine";
    test_state::Test1();
}

#endif

