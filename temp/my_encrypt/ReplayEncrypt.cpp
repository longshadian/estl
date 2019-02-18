#include "ReplayEncrypt.h"

#include <cstring>

namespace replaylib {

static uint8_t getHeight_3(uint8_t val)
{
    return uint8_t(((val)&0xE0) >> 5);
}

static uint8_t getLower_5(uint8_t val)
{
    return uint8_t((val) & 0x1F);
}

static uint8_t	getHeight_5(uint8_t val)
{
    return uint8_t(((val)&0xF8) >> 3);
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

void encrypt(const uint8_t* src, size_t len, const uint8_t* key, size_t key_len, uint8_t* out)
{
    if (!key || key_len == 0) {
        std::memcpy(out, src, len);
        return;
    }

    for (size_t i = 0; i != len; ++i) {
        uint8_t val = src[i];
        val = static_cast<uint8_t>(~val) ^ key[i%key_len];
        encryptShift(&val);
        *out = val;
        ++out;
    }
}

void decrypt(const uint8_t* src, size_t len, const uint8_t* key, size_t key_len, uint8_t* out)
{
    if (!key || key_len == 0) {
        std::memcpy(out, src, len);
        return;
    }
    for (size_t i = 0; i != len; ++i) {
        uint8_t val = src[i];
        decryptShift(&val);
        val ^= key[i%key_len];
        *out = static_cast<uint8_t>(~val);
        ++out;
    }
}

} // replaylib
