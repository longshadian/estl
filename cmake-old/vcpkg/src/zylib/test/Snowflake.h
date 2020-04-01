#pragma once

#include <ctime>
#include <chrono>
#include <array>

#include <cmath>
#include <type_traits>

template <uint64_t TIMESTAMP_BIS, uint64_t WORKER_ID_BITS, uint64_t SEQUENCE_BITS>
class Snowflake
{
    static_assert(1 + TIMESTAMP_BIS + WORKER_ID_BITS + SEQUENCE_BITS == 64
        , "TIMESTAMP_BIS + WORKER_ID_BITS + SEQUENCE_BITS != 63");
public:
    Snowflake(uint32_t worker_id, uint64_t start_epoch = 0)
        : m_epoch(start_epoch)
        , m_worker_id(worker_id)
        , m_last_timestamp()
        , m_sequence()
    {
    }

    ~Snowflake() = default;
    Snowflake(const Snowflake&) = delete;
    Snowflake& operator=(const Snowflake&) = delete;
    Snowflake(Snowflake&&) = delete;
    Snowflake& operator=(Snowflake&&) = delete;
    
    constexpr uint64_t MAX_TIMESTAMP() const { return uint64_t((uint64_t(1) << TIMESTAMP_BIS) - 1); }
    constexpr uint64_t MAX_WORKER_ID() const { return uint64_t((uint64_t(1) << WORKER_ID_BITS) -1); }
    constexpr uint64_t MAX_SEQUENCE() const { return uint64_t((uint64_t(1) << SEQUENCE_BITS) - 1); }
    constexpr uint64_t WORKER_ID_SHIFT() const { return SEQUENCE_BITS; }
    constexpr uint64_t TIMESTAMP_SHIFT() const { return SEQUENCE_BITS + WORKER_ID_BITS; }

    int64_t NewID()
    {
        auto timestamp = GenTimestamp();
        if (timestamp < m_last_timestamp) {
            // 如果出现时间戳回溯了，放弃生成id.例如系统时间被调整
            return 0;
        }

        if (m_last_timestamp == timestamp) {
            m_sequence++;
            if (m_sequence > MAX_SEQUENCE()) {
                // 同一时间戳，序列号超过限制,返回失败
                //timestamp = waitNextMilliseconds(m_last_timestamp);
                return 0;
            }
        } else {
            // 时间戳不同，序列号从0开始
            m_sequence = 0;
        }
        m_last_timestamp = timestamp;

        int64_t id = 0;
        id = timestamp << TIMESTAMP_SHIFT();    // 时间
        id |= m_worker_id << WORKER_ID_SHIFT(); // 中间机器ID
        id |= m_sequence;                       // 最后sequenceID
        return id;
    }

    std::array<uint64_t, 3> Parse(uint64_t value) const
    {
        std::array<uint64_t, 3> arr{};
        arr[2] = value & (MAX_SEQUENCE());
        arr[1] = (value >> WORKER_ID_SHIFT()) & (MAX_WORKER_ID());
        arr[0] = (value >> TIMESTAMP_SHIFT()) & (MAX_TIMESTAMP());
        return arr;
    }

private:
    static uint64_t GenTimestamp() 
    {
        auto tnow = std::chrono::system_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(tnow.time_since_epoch()).count();
    }

    static uint64_t waitNextMilliseconds(uint64_t lastTimestamp)
    {
        auto timestamp = GenTimestamp();
        while (timestamp <= lastTimestamp) {
            // 新的时间戳要大于旧的时间戳，才算作有效时间戳
            timestamp = GenTimestamp();
            //std::cout << "wait\n";
        }
        return timestamp;
    }

private:
    const uint64_t          m_epoch;
    const uint64_t          m_worker_id;
    uint64_t                m_last_timestamp;
    uint64_t                m_sequence;
};
