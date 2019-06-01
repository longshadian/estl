#include "Openssl.h"

#include <openssl/md5.h>
#include <sstream>
#include <ios>
#include <array>
#include <iomanip>

namespace openssl 
{

std::size_t Base64EncodeSize(std::size_t t)
{
    /*
    EVP_EncodeBlock() encodes a full block of input data in f and of length dlen and stores it in t.For
        every 3 bytes of input provided 4 bytes of output data will be produced.If dlen is not divisible by 3
        then the block is encoded as a final block of data and the output is padded such that it is always
        divisible by 4. Additionally a NUL terminator character will be added.For example if 16 bytes of input
        data is provided then 24 bytes of encoded data is created plus 1 byte for a NUL terminator(i.e. 25
            bytes in total).The length of the data generated without the NUL terminator is returned from the
        function.
    */
    return ((t / 3) +1)*4 + 1;
}

int Base64Encode(const unsigned char* src, std::size_t src_len, unsigned char* out)
{
    return ::EVP_EncodeBlock(out, src, (int)src_len);
}

// base解码  
int Base64Decode(const unsigned char* src, std::size_t src_len, unsigned char* out)
{
    if (src_len == 0)
        return -1;

    //获取"="填充个数
    int padding_len = 0;
    auto pos = src_len;
    while (true) {
        if (src[--pos] == '=')
            ++padding_len;
        else 
            break;
        if (pos == 0)
            return -1;
    }
    int len = ::EVP_DecodeBlock(out, src, (int)src_len);
    if (len == -1)
        return -1;
    return len -= padding_len;
}

std::string DigitSign_RSA_SHA1_Base64(const void* src, std::size_t src_len, RSA* private_key)
{
    return detail::DigitSign_Base64(src, src_len, private_key, ::EVP_sha1());
}

std::string DigitSign_RSA_SHA256_Base64(const void* src, std::size_t src_len, RSA* private_key)
{
    return detail::DigitSign_Base64(src, src_len, private_key, ::EVP_sha256());
}

bool DigitVerify_RSA_SHA1_Base64(const void* src, std::size_t src_len, const std::string& b64_sign, RSA* public_key)
{
    return detail::DigitVerify_Base64(src, src_len, b64_sign, public_key, ::EVP_sha1());
}

bool DigitVerify_RSA_SHA256_Base64(const void* src, std::size_t src_len, const std::string& b64_sign, RSA* public_key)
{
    return detail::DigitVerify_Base64(src, src_len, b64_sign, public_key, ::EVP_sha256());
}


std::string MD5(const unsigned char* src, std::size_t len)
{
    unsigned char result[MD5_DIGEST_LENGTH];
    ::MD5(src, len, result);
    std::ostringstream sout;
    sout << std::hex << std::setfill('0');
    for (auto c : result) {
        sout << std::setw(2) << (long long)c;
    }
    return sout.str();
}

std::string MD5(const std::string& path)
{
    auto* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        return {};
    }

    MD5state_st ctx{};
    std::memset(&ctx, 0, sizeof(ctx));
    ::MD5_Init(&ctx);
    std::array<char, 1024 * 16> read_buffer{};
    std::uint64_t total = 0;
    while (true) {
        std::size_t readn = std::fread(read_buffer.data(), 1, read_buffer.size(), f);
        if (readn = 0) {
            break;
        }
        if (readn > read_buffer.size()) {
            break;
        }
        total += readn;
        ::MD5_Update(&ctx, read_buffer.data(), readn);
    }
    std::fclose(f);

    std::array<unsigned char, MD5_DIGEST_LENGTH> md5_buffer{};
    ::MD5_Final(md5_buffer.data(), &ctx);
    return detail::ToHex(md5_buffer.data(), md5_buffer.size());
}

} // namespace openssl

namespace openssl 
{ 

namespace detail 
{

std::string DigitSign_Base64(const void* src, std::size_t src_len, RSA* private_key, const EVP_MD* evp_md)
{
    auto sign_data = DigitSign(src, src_len, private_key, evp_md);
    if (sign_data.empty())
        return {};

    auto b64_buff_len = Base64EncodeSize(sign_data.size());
    std::vector<unsigned char> b64_buff{};
    b64_buff.resize(b64_buff_len);
    int b64_len = Base64Encode(sign_data.data(), sign_data.size(), b64_buff.data());
    if (b64_len < 0)
        return {};
    const char* pos = (const char*)b64_buff.data();
    return std::string(pos, pos + b64_len);
}

std::vector<unsigned char> DigitSign(const void* src, std::size_t src_len, RSA* private_key, const EVP_MD* evp_md) 
{
    if (src_len == 0)
        return {};

    EVP_PKEY_Uptr evp_pkey{::EVP_PKEY_new()};
    int ret = ::EVP_PKEY_set1_RSA(&*evp_pkey, private_key);
    if (ret != 1) {
#ifdef OPENSSL_DEBUG
        std::cout << "EVP_PKEY_set1_RSA error";
#endif
        return {};
    }

    EVP_MD_CTX_Uptr evp_md_ctx{::EVP_MD_CTX_create()};
    ret = ::EVP_DigestSignInit(&*evp_md_ctx, nullptr, evp_md, nullptr, &*evp_pkey);
    if (ret != 1) {
#ifdef OPENSSL_DEBUG
        std::cout << "EVP_DigestSignInit error";
#endif
        return {};
    }

    ret = ::EVP_DigestSignUpdate(&*evp_md_ctx, src, src_len);
    if (ret != 1) {
#ifdef OPENSSL_DEBUG
        std::cout << "EVP_DigestSignUpdate error";
#endif
        return {};
    }

    std::size_t sign_len = 0;
    ret = ::EVP_DigestSignFinal(&*evp_md_ctx, nullptr, &sign_len);
    if (ret != 1) {
#ifdef OPENSSL_DEBUG
        std::cout << "EVP_DigestSignFinal get len error";
#endif
        return {};
    }

    std::vector<unsigned char> sign_data{};
    sign_data.resize(sign_len);
    ret = ::EVP_DigestSignFinal(&*evp_md_ctx, sign_data.data(), &sign_len);
    if (ret != 1) {
#ifdef OPENSSL_DEBUG
        std::cout << "EVP_DigestSignFinal get sign data error";
#endif
        return {};
    }
    sign_data.resize(sign_len);
    return sign_data;
}

bool DigitVerify_Base64(const void* src, std::size_t src_len, const std::string& sign_str, RSA* public_key, const EVP_MD* evp_md)
{
    //签名base64 decode
    std::vector<unsigned char> sign{};
    sign.reserve(sign_str.size());
    for (auto& c : sign_str) {
        sign.push_back(static_cast<unsigned char>(c));
    }
    std::vector<unsigned char> sign_buffer{};
    sign_buffer.resize(sign.size());
    int sign_buffer_len = Base64Decode(sign.data(), sign.size(), sign_buffer.data());
    if (sign_buffer_len == -1)
        return false;
    sign_buffer.resize(sign_buffer_len);

    return DigitVerify(src, src_len, sign_buffer.data(), sign_buffer_len, public_key, evp_md);
}

bool DigitVerify(const void* src, std::size_t src_len, unsigned char* sign, std::size_t sign_len, RSA* public_key, const EVP_MD* evp_md)
{
    if (src_len == 0)
        return {};

    EVP_PKEY_Uptr evp_pkey{ ::EVP_PKEY_new() };
    int ret = ::EVP_PKEY_set1_RSA(&*evp_pkey, public_key);
    if (ret != 1) {
#ifdef DEBUG
        std::cout << "EVP_PKEY_set1_RSA error";
#endif
        return false;
    }

    EVP_MD_CTX_Uptr evp_md_ctx{ ::EVP_MD_CTX_create() };
    ret = ::EVP_DigestVerifyInit(&*evp_md_ctx, nullptr, evp_md, nullptr, &*evp_pkey);
    if (ret != 1) {
#ifdef DEBUG
    std::cout << "EVP_DigestVerifyInit error";
#endif
        return false;
    }

    ret = ::EVP_DigestVerifyUpdate(&*evp_md_ctx, src, src_len);
    if (ret != 1) {
#ifdef DEBUG
        std::cout << "EVP_DigestVerifyUpdate error";
#endif
        return false;
    }

    ret = ::EVP_DigestVerifyFinal(&*evp_md_ctx, sign, sign_len);
    if (ret == 0) {
#ifdef DEBUG
        std::cout << "EVP_DigestVerifyFinal get sign data error\n";
        //std::cout << ERR_error_string(ERR_get_error(), nullptr) << "\n";
#endif
        return false;
    }
    return true;
}

std::string ToHex(const void* data, std::size_t len)
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

} // namespace detail
} // namespace openssl

