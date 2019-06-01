#include "Crypto.h"

#include <cassert>
#include <cstring>
#include <vector>
#include <sstream>
#include <string>
#include <memory>
#include <iostream>

static uint8_t GetHeight_3(uint8_t val)
{
    return uint8_t((val & 0xE0) >> 5);
}

static uint8_t GetLower_5(uint8_t val)
{
    return uint8_t(val & 0x1F);
}

static uint8_t GetHeight_5(uint8_t val)
{
    return uint8_t((val & 0xF8) >> 3);
}

static uint8_t GetLower_3(uint8_t val)
{
    return uint8_t(val & 0x07);
}

static void EncryptShift(uint8_t* val)
{
    uint8_t height = GetHeight_3(*val);
    uint8_t lower = GetLower_5(*val);
    *val = static_cast<uint8_t>(lower << 3) | height;
}

static void DecryptShift(uint8_t* val)
{
    uint8_t height = GetHeight_5(*val);
    uint8_t lower = GetLower_3(*val);
    *val = static_cast<uint8_t>(lower << 5) | height;
}

void SimpleEncrypt(const uint8_t* src, size_t len, const uint8_t* key, size_t key_len, uint8_t* out)
{
    if (!key || key_len == 0) {
        std::memcpy(out, src, len);
        return;
    }

    for (size_t i = 0; i != len; ++i) {
        uint8_t val = src[i];
        val = static_cast<uint8_t>(~val) ^ key[i%key_len];
        EncryptShift(&val);
        *out = val;
        ++out;
    }
}

void SimpleDecrypt(const uint8_t* src, size_t len, const uint8_t* key, size_t key_len, uint8_t* out)
{
    if (!key || key_len == 0) {
        std::memcpy(out, src, len);
        return;
    }
    for (size_t i = 0; i != len; ++i) {
        uint8_t val = src[i];
        DecryptShift(&val);
        val ^= key[i%key_len];
        *out = static_cast<uint8_t>(~val);
        ++out;
    }
}

size_t PKCS7Padding_Length(size_t i, size_t k)
{
    return static_cast<size_t>(k - (i % k));
}

int PKCS7Padding_Check(const void* p, size_t len, size_t k) 
{
    const uint8_t* p_src = reinterpret_cast<const uint8_t*>(p);

    if (len == 0 || k == 0 || (len % k) != 0)
        return -1;

    uint8_t padding_val = p_src[len - 1];
    size_t padding_len = static_cast<size_t>(padding_val);
    if (len <= padding_len)
        return -1;

    for (size_t i = 0; i != padding_len; ++i) {
        if (p_src[len - padding_len + i] != padding_val)
            return -1;
    }
    return static_cast<int>(padding_len);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

AesCrypto::AesCrypto()
    : m_key()
    , m_key_length()
    , m_encrypt_key()
    , m_decrypt_key()
{
}

AesCrypto::~AesCrypto()
{
}

int AesCrypto::SetKey(const void* p, size_t len)
{
    m_key.fill(0);
    std::memcpy(m_key.data(), p, len);
    m_key_length = len;

    int ret = ::AES_set_encrypt_key(m_key.data(), m_key_length * 8, &m_encrypt_key);
    if (ret != 0)
        return ret;
    return ::AES_set_decrypt_key(m_key.data(), m_key_length * 8, &m_decrypt_key);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
AesEbc::AesEbc()
{
}

AesEbc::~AesEbc()
{
}

int AesEbc::SetKey(const void* p, size_t len)
{
    return AesCrypto::SetKey(p, len);
}

void AesEbc::Encrypt(const void* p, size_t len, void* out) const
{
    assert((len%AES_BLOCK_SIZE) == 0);
    size_t n = 0;
    const unsigned char* pos = reinterpret_cast<const unsigned char*>(p);
    unsigned char* out_pos = reinterpret_cast<unsigned char*>(out);
    while (n < len) {
        ::AES_ecb_encrypt(pos, out_pos, &m_encrypt_key, AES_ENCRYPT);
        pos += AES_BLOCK_SIZE;
        out_pos += AES_BLOCK_SIZE;
        n += AES_BLOCK_SIZE;
    }
}

void AesEbc::Decrypt(const void* p, size_t len, void* out) const
{
    assert((len%AES_BLOCK_SIZE) == 0);
    size_t n = 0;
    const unsigned char* pos = reinterpret_cast<const unsigned char*>(p);
    unsigned char* out_pos = reinterpret_cast<unsigned char*>(out);
    while (n < len) {
        ::AES_ecb_encrypt(pos, out_pos, &m_decrypt_key, AES_DECRYPT);
        pos += AES_BLOCK_SIZE;
        out_pos += AES_BLOCK_SIZE;
        n += AES_BLOCK_SIZE;
    }
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
AesCbc::AesCbc()
    : m_iv()
{
}

AesCbc::~AesCbc()
{
}

int AesCbc::SetKey(const void* p, size_t len)
{
    return AesCrypto::SetKey(p, len);
}

void AesCbc::SetIV(const void* p)
{
    std::memcpy(m_iv.data(), p, m_iv.size());
}

void AesCbc::Encrypt(const void* p, size_t len, void* out) const
{
    assert((len%AES_BLOCK_SIZE) == 0);
    std::array<uint8_t, AES_BLOCK_SIZE> iv = m_iv;
    ::AES_cbc_encrypt(reinterpret_cast<const unsigned char*>(p), reinterpret_cast<unsigned char*>(out)
        , len, &m_encrypt_key, iv.data(), AES_ENCRYPT);
}

void AesCbc::Decrypt(const void* p, size_t len, void* out) const
{
    assert((len%AES_BLOCK_SIZE) == 0);
    std::array<uint8_t, AES_BLOCK_SIZE> iv = m_iv;
    ::AES_cbc_encrypt(reinterpret_cast<const unsigned char*>(p), reinterpret_cast<unsigned char*>(out)
        , len, &m_decrypt_key, iv.data(), AES_DECRYPT);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
/*
int main()
{
    auto ToHexString = [](const std::vector<uint8_t>& data) -> std::string
    {
        std::ostringstream ostm{};
        std::array<char, 10> buff{};
        for (auto c : data) {
            snprintf(buff.data(), buff.size(), "0x%02X ", c);
            ostm << buff.data();
        }
        return ostm.str();
    };

    std::vector<uint8_t> data{};
    data.resize(16, 'a');
    std::cout << "data: " << data.size() << "\n" << ToHexString(data) << "\n";

    auto padding_len = PKCS7Padding_Length(data.size(), AES_BLOCK_SIZE);
    std::vector<uint8_t> data_padding = data;
    data_padding.resize(data.size() + padding_len, static_cast<uint8_t>(padding_len));
    std::cout << "data_padding: " << data_padding.size() << "\n" << ToHexString(data_padding) << "\n";

    std::array<char, 16> key{};
    key.fill('B');
    auto aes = std::make_unique<AesEbc>();
    aes->SetKey(key.data(), key.size());

    std::vector<uint8_t> data_encrypt{};
    data_encrypt.resize(data_padding.size());
    aes->Encrypt(data_padding.data(), data_padding.size(), data_encrypt.data());
    std::cout << "data_encrypt: " << data_encrypt.size() << "\n" << ToHexString(data_encrypt) << "\n";

    std::vector<uint8_t> data_decrypt{};
    data_decrypt.resize(data_encrypt.size());
    aes->Decrypt(data_encrypt.data(), data_encrypt.size(), data_decrypt.data());
    std::cout << "data_decrypt: " << data_decrypt.size() << "\n" << ToHexString(data_decrypt) << "\n";

    auto n = PKCS7Padding_Check(data_decrypt.data(), data_decrypt.size(), AES_BLOCK_SIZE);
    if (n < 0) {
        std::cout << "check error!\n";
        return 0;
    }

    data_decrypt.resize(data_decrypt.size() - static_cast<size_t>(n));
    std::cout << "data: " << data_decrypt.size() << "\n"  << ToHexString(data_decrypt) << "\n";

    std::cout << "success: " << (ToHexString(data) == ToHexString(data_decrypt)) << "\n";
    return 0;
}

g++ -g -Wall -Wextra -std=c++14 ./Crypto.cpp -lcrypto
*/
