#ifndef _OPENSSL_H_
#define _OPENSSL_H_

#include <string>
#include <memory>
#include <vector>

#include <openssl/md5.h>
#include <openssl/sha.h>
#include <openssl/rsa.h>
#include <openssl/rand.h>
#include <openssl/objects.h>
#include <openssl/bio.h>
#include <openssl/pem.h>
#include <openssl/pem2.h>
#include <openssl/err.h>
#include <openssl/evp.h>

namespace openssl {

struct EVP_PKEY_guard
{
    void operator()(EVP_PKEY* p)
    {
        if (p)
            ::EVP_PKEY_free(p);
    }
};
using EVP_PKEY_uptr = std::unique_ptr<EVP_PKEY, EVP_PKEY_guard>;

struct EVP_MD_CTX_guard
{
    void operator()(EVP_MD_CTX* p)
    {
        if (p)
            ::EVP_MD_CTX_destroy(p);
    }
};
using EVP_MD_CTX_uptr = std::unique_ptr<EVP_MD_CTX, EVP_MD_CTX_guard>;


//计算base64编码需要的字节
size_t base64EncodeSize(size_t t);

// base64编码,返回值:-1失败, 0或1成功.0表示没有填充数据
int base64Encode(const unsigned char* src, size_t src_len, unsigned char* out);

// base解码  返回-1失败, 其他值表示长度
int base64Decode(const unsigned char* src, size_t src_len, unsigned char* out);

//签名算法,返回base64签名字符串，如果返回长度为0，签名出错
std::string digitSign_RSA_SHA1_Base64(const void* src, size_t src_len, RSA* public_key);
std::string digitSign_RSA_SHA256_Base64(const void* src, size_t src_len, RSA* public_key);

//签名校验算法
bool digitVerify_RSA_SHA1_Base64(const void* src, size_t src_len, const std::string& b64_sign, RSA* public_key);
bool digitVerify_RSA_SHA256_Base64(const void* src, size_t src_len, const std::string& b64_sign, RSA* public_key);

//生成MD5
std::string md5(const unsigned char* src, size_t len);

}

namespace openssl { namespace detail {

std::string digitSign_Base64(const void* src, size_t src_len, RSA* private_key, const EVP_MD* evp_md);
std::vector<unsigned char> digitSign(const void* src, size_t src_len, RSA* private_key, const EVP_MD* evp_md);

bool digitVerify_Base64(const void* src, size_t src_len, const std::string& sign_str, RSA* public_key, const EVP_MD* evp_md);
bool digitVerify(const void* src, size_t src_len, unsigned char* sign, size_t sign_len, RSA* public_key, const EVP_MD* evp_md);

}
}

#endif
