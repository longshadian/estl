#include <ctime>

#include <chrono>
#include <iostream>
#include <list>
#include <queue>
#include <random>
#include <thread>

namespace test_timer
{

class Function
{
    struct Base
    {
        virtual ~Base() {}
        virtual void call() = 0;
    };

    template <typename F>
    struct Impl : public Base
    {
        Impl(F&& f) : m_fun(std::move(f)) {}
        virtual ~Impl() = default;
        virtual void call() { m_fun(); }

        F m_fun;
    };
public:
    Function() = default;

    template<typename F>
    Function(F&& f) : m_impl(new Impl<F>(std::move(f)))
    {}

    Function(Function&& rhs) : m_impl(std::move(rhs.m_impl))
    {}

    Function& operator=(Function&& rhs)
    {
        if (this != &rhs) {
            m_impl = std::move(rhs.m_impl);
        }
        return *this;
    }
    Function(const Function&) = delete;
    Function& operator=(const Function&) = delete;

    operator bool() const
    {
        return m_impl != nullptr;
    }

    void operator()() { m_impl->call(); }
private:
    std::unique_ptr<Base>   m_impl;
};


using SteadyTimePoint = std::chrono::steady_clock::time_point;

SteadyTimePoint getStreadyTimePoint()
{
    return std::chrono::steady_clock::now();
}

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
        m_start = Duration::zero();
    }

    Duration remain() const
    {
        return m_expire - m_start;
    }

    Duration m_start;
    Duration m_expire;
};

using Timer = BasicTimer<std::chrono::milliseconds>;

struct TimerManager
{
    struct Slot
    {
        Function    m_f;
        Timer       m_t;
    };

    struct PreSortSlot
    {
        bool operator()(const Slot& s1, const Slot& s2) const
        {
            return s1.m_t.remain() < s2.m_t.remain();
        }
    };

    void sortSlot()
    {
        m_timer.sort(PreSortSlot{});
    }

    std::list<Slot> m_timer;
};

std::default_random_engine g_engine{(unsigned long int)std::time(nullptr)};

template <typename T>
T rand()
{
    return std::uniform_int_distribution<T>()(g_engine);
}

template <typename T>
T rand(T closed_begin, T closed_end)
{
    return std::uniform_int_distribution<T>(closed_begin, closed_end)(g_engine);
}

} // namespace test_timer

int TestTimer()
{
    auto tbegin = std::chrono::system_clock::now();
    //std::chrono::milliseconds x{std::chrono::seconds(4)};
    test_timer::Timer t{std::chrono::seconds(-2) + std::chrono::milliseconds(600)};
    test_timer::SteadyTimePoint previous_tp = test_timer::getStreadyTimePoint();
    test_timer::SteadyTimePoint current_tp{};
    while (true) {
        current_tp = test_timer::getStreadyTimePoint();
        auto duration = current_tp - previous_tp;

        previous_tp = current_tp;
        auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
        std::cout << "delta:" << delta.count() << "\n";
        t.update(delta);
        if (t.passed()) {
            auto tend = std::chrono::system_clock::now();
            std::cout << "hehe:" << std::chrono::duration_cast<std::chrono::milliseconds>(tend - tbegin).count() << "\n";
            t.reset();
            break;
        } else {
            std::cout << (t.m_expire - t.m_start).count() << "\n";
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(test_timer::rand(100, 200)));
    }

    test_timer::TimerManager tm;
    tm.m_timer.push_back({test_timer::Function{}, test_timer::Timer{std::chrono::seconds(10)}});
    tm.m_timer.push_back({test_timer::Function{}, test_timer::Timer{std::chrono::seconds(2)}});
    tm.m_timer.push_back({test_timer::Function{}, test_timer::Timer{std::chrono::seconds(31)}});
    tm.m_timer.push_back({test_timer::Function{}, test_timer::Timer{std::chrono::seconds(2)}});
    tm.m_timer.push_back({test_timer::Function{}, test_timer::Timer{std::chrono::seconds(5)}});
    tm.m_timer.push_back({test_timer::Function{}, test_timer::Timer{std::chrono::seconds(6)}});
    tm.m_timer.push_back({test_timer::Function{}, test_timer::Timer{std::chrono::seconds(1)}});

    tm.sortSlot();

    for (auto& it : tm.m_timer) {
        std::cout << std::chrono::duration_cast<std::chrono::seconds>(it.m_t.remain()).count() << "\n";
    }

    return 0;
}

