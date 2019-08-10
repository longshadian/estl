#include "stdafx.h"

#include <openssl/rsa.h>
#include <openssl/ssl.h>
#include <openssl/pem.h>
#include <openssl/ssl.h>
#include <openssl/rsa.h>
#include <openssl/evp.h>
#include <openssl/bio.h>
#include <openssl/err.h>
#include <openssl/md5.h>

#include <cstdio>
#include <algorithm>
#include <array>

#include "Encryption.h"
#include "Base64.h"
#include "Log.h"


static const std::string PUBLIC_KEY = 
"-----BEGIN PUBLIC KEY-----\r\n"
"MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQDZoW3BUe7QsNY2E/mU9B1NcVt1\r\n"
"r6sPO5EgkI3KCZ/7vbJ1PrZ9vAdL5ftf2g38wG2z7Do8rXWnutrLl1Nia9R987kt\r\n"
"wOGrH1+gvUNjCeS0zxIAePwLMfvflOJlsDwfuL69EbLwnkztEPC1y68wV+4h/bln\r\n"
"TPjy0WnRnG7iwdDBOwIDAQAB\r\n"
"-----END PUBLIC KEY-----\r\n";

static std::string RsaPublicDecrypt(const void* p, std::size_t p_len, RSA* rsa)
{
    int len = ::RSA_size(rsa);
    std::vector<char> buffer;
    buffer.resize(len);
    int ret = ::RSA_public_decrypt(p_len, (const unsigned char*)p, (unsigned char*)buffer.data(), rsa, RSA_PKCS1_PADDING);
    if (ret < 0) {
        YZHL_LOG_WARNING("RsaPublicDecrypt failed.");
        return "";
    }
    std::string strRet = std::string(buffer.data(), buffer.data() + ret);
    return strRet;
}

static std::string RsaPublicEncrypt(const void* p, std::size_t p_len, RSA* rsa)
{
    int len = ::RSA_size(rsa);
    std::vector<char> buffer;
    buffer.resize(len);
    int ret = ::RSA_public_encrypt(p_len, (const unsigned char*)p, (unsigned char*)buffer.data(), rsa, RSA_PKCS1_PADDING);
    if (ret < 0) {
        YZHL_LOG_WARNING("RsaPublicEncrypt failed.");
        return "";
    }
    std::string strRet = std::string(buffer.data(), buffer.data() + ret);
    return strRet;
}


static std::string ToHex(const void* data, std::size_t len)
{
    const unsigned char* p = reinterpret_cast<const unsigned char*>(data);
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (std::size_t i = 0; i != len; ++i) {
        unsigned int c = p[i];
        out << std::setw(2) << c;
    }
    return out.str();
}

static std::string HashValueMD5(const void* src, std::size_t len)
{
    unsigned char result[MD5_DIGEST_LENGTH];
    ::MD5(reinterpret_cast<const unsigned char*>(src), len, result);
    return ::ToHex(result, MD5_DIGEST_LENGTH);
    /*
    std::ostringstream sout;
    sout << std::hex << std::setfill('0');
    for (auto c : result) {
        sout << std::setw(2) << (long long)c;
    }
    return sout.str();
    */
}

bool Encryption::VerifySign(const std::string& json, const std::string& sign)
{
    if (json.empty() || sign.empty())
        return false;
    std::string md5 = HashValueMD5(json.data(), json.length());
    if (!md5.size()) {
        return false;
    }

    std::string input_data = base64::Decode(sign);
    std::string plain = DecryptSign(input_data);
    if (plain.empty()) {
        return false;
    }

    return ToUpperCase(md5) == ToUpperCase(base64::Decode(plain));
}

bool Encryption::StartRsaEncrypt(const std::string& input_data, std::string* output)
{
    const std::string& pubkey = PUBLIC_KEY;
    BIO* keybio = ::BIO_new_mem_buf(pubkey.c_str(), -1);
    RSA* rsa = ::RSA_new();
    rsa = ::PEM_read_bio_RSA_PUBKEY(keybio, &rsa, NULL, NULL);
    if (!rsa) {
        YZHL_LOG_WARNING("pem read rsa pubkey failed.");
        ::BIO_free_all(keybio);
        return "";
    }

    // 分段加密，长度不能太长
    const int segment_max_len = ::RSA_size(rsa) - 11;
    const std::string& src = input_data;
    int TOTAL_SIZE = src.length();
    const char* p = src.data();

    int count = TOTAL_SIZE / segment_max_len;
    if (TOTAL_SIZE % segment_max_len != 0) {
        count += 1;
    }

    bool succ = true;
    std::string result;
    for (int i = 0; i != count; ++i) {
        std::size_t len = segment_max_len;
        if (segment_max_len * (i + 1) > TOTAL_SIZE) {
            len = TOTAL_SIZE - (segment_max_len * i);
        }
        std::string segment = RsaPublicEncrypt(p, len, rsa);
        if (segment.empty()) {
            succ = false;
            break;
        }
        result += segment;
    }
    ::BIO_free_all(keybio);
    ::RSA_free(rsa);
    if (succ) {
        std::swap(result, *output);
    }
    return succ;
}

std::string Encryption::DecryptSign(const std::string& sign)
{
    std::string plain = StartRsaDecrypt(sign, PUBLIC_KEY);
    return plain;
}

std::string Encryption::StartRsaDecrypt(const std::string& input, const std::string& pub_key)
{
    BIO* keybio = ::BIO_new_mem_buf(pub_key.c_str(), -1);
    RSA* rsa = ::RSA_new();
    rsa = ::PEM_read_bio_RSA_PUBKEY(keybio, &rsa, NULL, NULL);
    if (!rsa) {
        YZHL_LOG_WARNING("pem read rsa pubkey failed.");
        ::BIO_free_all(keybio);
        return "";
    }

    std::size_t TOTAL_SIZE = input.length();
    const char* p = input.data();
    std::size_t count = TOTAL_SIZE / 128;
    if (TOTAL_SIZE % 128 != 0) {
        count += 1;
    }
    std::string result;
    for (std::size_t i = 0; i != count; ++i) {
        std::size_t len = 128;
        if (128 * (i + 1) > TOTAL_SIZE) {
            len = TOTAL_SIZE - (128 * i);
        }
        std::string segment = RsaPublicDecrypt(p, len, rsa);
        result += segment;
    }
    ::BIO_free_all(keybio);
    ::RSA_free(rsa);
    return result;
}

std::string Encryption::ToUpperCase(const std::string& src)
{
    if (src.empty())
        return "";
    std::string dst;
    dst.resize(src.size());
    std::transform(src.begin(), src.end(), dst.begin(), [](int c) { return static_cast<char>(::toupper(c)); });
    return dst;
}

std::string Encryption::ToLowerCase(const std::string& src)
{
    if (src.empty())
        return "";
    std::string dst;
    dst.resize(src.size());
    std::transform(src.begin(), src.end(), dst.begin(), [](int c) { return static_cast<char>(::tolower(c)); });
    return dst;
}

std::string Encryption::CatFile(const char* f)
{
    std::FILE* fp = std::fopen(f, "rb");
    if (!fp)
        return "";

    std::string content;
    std::array<char, 1024> buffer{};
    while (1) {
        int readn = std::fread(buffer.data(), 1, buffer.size(), fp);
        if (readn == 0)
            break;
        content.append(buffer.data(), buffer.data() + readn);
    }
    std::fclose(fp);
    return content;
}

bool Encryption::CatFile(const std::string& path, std::string* out)
{
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }

    bool succ = true;
    std::string content;
    std::array<char, 1024> buffer;
    std::uint64_t total = 0;
    while (true) {
        std::size_t readn = std::fread(buffer.data(), 1, buffer.size(), f);
        if (readn == 0) {
            break;
        }
        if (readn > buffer.size()) {
            succ = false;
            break;
        }
        total += readn;
        content.append(buffer.data(), buffer.data() + readn);
    }
    std::fclose(f);
    if (!succ)
        return succ;
    std::swap(*out, content);
    return true;
}
