#include <array>
#include <iostream>
#include <cstring>
#include <ios>
#include <sstream>
#include <boost/convert.hpp>
#include <boost/convert/stream.hpp>

struct boost::cnv::by_default : public boost::cnv::cstream {};

const uint64_t MAGIC_NUM = 100000000;

static uint8_t getHeight_3(uint8_t val)
{
    return uint8_t(((val) & 0xE0) >> 5);
}

static uint8_t getLower_5(uint8_t val)
{
    return uint8_t((val) & 0x1F);
}

static uint8_t	getHeight_5(uint8_t val)
{
    return uint8_t(((val) & 0xF8) >> 3);
}

static uint8_t	getLower_3(uint8_t val)
{
    return uint8_t((val) & 0x07);
}

static void encryptShift(uint8_t* val)
{
    uint8_t height = getHeight_3(*val);
    uint8_t lower = getLower_5(*val);
    *val = static_cast<uint8_t>(lower << 3) | height;
}

static void decryptShift(uint8_t* val)
{
    uint8_t height = getHeight_5(*val);
    uint8_t lower = getLower_3(*val);
    *val = static_cast<uint8_t>(lower << 5) | height;
}

static uint64_t decrypt(uint64_t uid, uint64_t key)
{
    std::array<uint8_t, sizeof(uid)> src{};
    std::array<uint8_t, sizeof(uid)> dest{};
    std::memcpy(src.data(), &uid, src.size());
    for (size_t i = 0; i != src.size(); ++i) {
        uint8_t val = src[i];
        decryptShift(&val);
        //dest[i] = static_cast<uint8_t>(~val);
        dest[i] = val;
    }

    uint64_t value = 0;
    std::memcpy(&value, dest.data(), dest.size());
    return value ^ key;
}

static uint64_t encrypt(uint64_t value, uint64_t key)
{
    value ^= key;
    std::array<uint8_t, sizeof(value)> src{};
    std::array<uint8_t, sizeof(value)> dest{};
    std::memcpy(src.data(), &value, src.size());
    for (size_t i = 0; i != src.size(); ++i) {
        uint8_t val = src[i];
        //val = static_cast<uint8_t>(~val);
        encryptShift(&val);
        dest[i] = val;
    }

    uint64_t uid = 0;
    std::memcpy(&uid, dest.data(), dest.size());
    return uid;
}

static uint64_t shuffleUserID(uint64_t uid)
{
    // 最高一个数值表示奇数长度
    // 排列顺序为 奇数长度 奇数位 偶数位
    std::string odd_num{};
    std::string even_num{};
    auto s = std::to_string(uid);
    for (size_t i = 0; i != s.size(); ++i) {
        if (i % 2 == 0) {
            even_num.push_back(s[i]);
        } else {
            odd_num.push_back(s[i]);
        }
    }
 
    return boost::convert<uint64_t>(std::to_string(odd_num.size()) + odd_num + even_num).value();
}

static uint64_t recoverUserID(uint64_t uid)
{
    auto s = std::to_string(uid);
    std::string s_ex{ s.begin() + 1, s.end() };
    auto odd_len = boost::convert<int>(std::string{ s[0] }).value();
    s = s_ex;
    std::string odd_num{ s.begin(), s.begin() + odd_len };
    std::string even_num{ s.begin() + odd_len, s.end() };
    std::string uid_str{};
    for (size_t i = 0; i != odd_num.size(); ++i) {
        uid_str.push_back(even_num[i]);
        uid_str.push_back(odd_num[i]);
    }
    // 有剩下的偶数位
    if (even_num.size() > odd_num.size()) {
        uid_str.push_back(even_num[odd_num.size()]);
    }
    return boost::convert<uint64_t>(uid_str).value();
}

int main()
{
                 //91118000
    uint64_t uid = 88112189;
    //uint64_t uid = 131160544;
    
    for (uint64_t i = 0; i != 20; ++i) {
        auto uid_s = shuffleUserID(uid + i);
        auto uid_e = encrypt(uid_s, MAGIC_NUM);
        
        uint64_t uid_d = decrypt(uid_e, MAGIC_NUM);
        uint64_t uid_ex = recoverUserID(uid_d);
        printf("%20lu %20lu %20lu %20lu %20lu\n", uid + i, uid_s, uid_e, uid_d, uid_ex);
    }

    /*
    for (uint64_t i = 0; i != 20; ++i) {
        std::ostringstream ostm{};
        ostm << std::oct << ((uid + i) ^ 10000000);
        std::cout << ostm.str() << "\n";
    }
    */

    return 0;
}
