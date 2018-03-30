//#pragma once

#include <ctime>
#include <sys/time.h>

#include <cmath>
#include <chrono>
#include <thread>
#include <type_traits>

#include <iostream>

template <int64_t T, uint32_t R>
struct Power
{
    enum { VALUE = T * Power<T, R - 1>::VALUE};
};

template <int64_t T>
struct Power<T, 0>
{
    enum {VALUE = 1};
};

template <uint32_t SIGN, uint32_t TM, uint32_t ID, uint32_t SEQUENCE>
class Snowflake
{
public:
    Snowflake(uint32_t id = 0)
        : m_epoch()
        , m_id(id)
        , m_seq()
    {
    }

    const uint64_t m_epoch;
    const uint64_t m_id;
    uint64_t m_seq;

    enum {ID_MAX = Power<2, ID>::VALUE};
    enum {ID_MASK = ID_MAX - 1 };

    enum {SEQUENCE_MAX = Power<2, SEQUENCE>::VALUE};
    enum {SEQUENCE_MASK = SEQUENCE_MAX - 1 };

    enum {TM_MAX = Power<2, TM>::VALUE};

    static uint64_t get_time()
    {
        struct timeval tv;
        gettimeofday(&tv, nullptr);
        uint64_t time = tv.tv_usec;
        time /= 1000;
        time += (static_cast<uint64_t>(tv.tv_sec) * 1000);
        return time;
    }

    int64_t generate()
    {
        int64_t value = 0;
        uint64_t time = get_time() - m_epoch;

        // 保留后41位时间
        value = time << (ID + SEQUENCE);

        // 中间10位是机器ID
        value |= (m_id & ID_MASK) << SEQUENCE;

        // 最后12位是sequenceID
        value |= m_seq++ & SEQUENCE_MASK;
        if (m_seq == SEQUENCE_MAX) {
            m_seq = 0;
        }
        return value;
    }
};

int main()
{
    //enum { V = std::pow<uint64_t>(2, 2); };
    std::cout << "13:" << Power<2, 13>::VALUE << " " << std::pow<uint64_t>(2, 13) << "\n";

    using SF = Snowflake<1, 41, 10, 12>;
    SF sf{};
    std::cout << "tm_max: " << SF::TM_MAX << "\n";
    std::cout << "id_max: " << SF::ID_MAX << "\n";
    std::cout << "seq_max: " << SF::SEQUENCE_MAX << "\n";

    std::cout << sf.generate() << "\n";
    std::cout << sf.generate() << "\n";
    std::cout << sf.generate() << "\n";
    std::cout << sf.generate() << "\n";
    std::cout << sf.generate() << "\n";
    return 0;
}
