
#include <openssl/sha.h>
#include <openssl/rsa.h>
#include <openssl/rand.h>
#include <openssl/objects.h>
#include <openssl/bio.h>
#include <openssl/pem.h>
#include <openssl/pem2.h>
#include <openssl/err.h>
#include <openssl/evp.h>

#include <iostream>
#include <vector>
#include <cstring>
#include <map>
#include <sstream>

#include "Openssl.h"
#include "CURL_Guard.h"

// base64编码  
int base64Encode(const unsigned char *encoded, int encodedLength, char *decoded)
{
    return ::EVP_EncodeBlock((unsigned char*)decoded, (const unsigned char*)encoded, encodedLength);
}

// base解码  
int base64Decode(const unsigned char *encoded, int encodedLength, char *decoded)
{
    return ::EVP_DecodeBlock((unsigned char*)decoded, (const unsigned char*)encoded, encodedLength);
}

std::string catFile(const char* f)
{
    FILE* pf = ::fopen(f, "r");
    if (!pf)
        return {};

    std::string s{};
    std::array<char, 1024> buff{};
    while (true) {
        auto len = ::fread(buff.data(), 1, buff.size(), pf);
        if (len == 0)
            break;
        s.append(buff.data(), buff.data() + len);
    }
    return s;
}

std::vector<uint8_t> catBinary(const char* f)
{
    FILE* pf = ::fopen(f, "r");
    if (!pf)
        return{};

    std::vector<uint8_t> s;
    std::array<uint8_t, 1024> buff{};
    while (true) {
        auto len = ::fread(buff.data(), 1, buff.size(), pf);
        if (len == 0)
            break;
        std::copy(buff.data(), buff.data() + len , std::back_inserter(s));
    }
    return s;
}

size_t stringReplace(const std::string& src, char c, const std::string& str, std::string* out)
{
    size_t count = 0;
    for (auto& it : src) {
        if (it == c) {
            out->append(str);
            ++count;
        } else {
            out->push_back(it);
        }
    }
    return count;
};


bool fun()
{
    const char* public_key = "MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQDI/SLEG7sO+K0ScHA9hhjjCHQ5U9YP7KvZ5zAZs59+7nJs63B7qhm+Vm1yH9gf6j592rne+N0h0hVUHfDeLokK5J2Dk3bORnsLcPt9FcpCQFmaloKZUO1brpYARkDsV9lvwB2mhdj7QTujAzY7dXWR7FDkHe3MHzRuX4BWHQHr1QIDAQAB";
    (void)public_key;
    const char* private_key = "-----BEGIN RSA PRIVATE KEY-----\nMIICXgIBAAKBgQDI/SLEG7sO+K0ScHA9hhjjCHQ5U9YP7KvZ5zAZs59+7nJs63B7qhm+Vm1yH9gf6j592rne+N0h0hVUHfDeLokK5J2Dk3bORnsLcPt9FcpCQFmaloKZUO1brpYARkDsV9lvwB2mhdj7QTujAzY7dXWR7FDkHe3MHzRuX4BWHQHr1QIDAQABAoGBAKkKvkLST/G3lVj82GnmbugEJDxTFqcuFxueJgZ33J2VTwfsvR5FYoGDj2l8/vlYulZ/r/SoutPHLckhpYQ4/2h7FGTeKfh0ZeFLZJHt936hqAIL236onEgFGUjlN5y97rLhawxs0aFVjn+gKpkQ8rilWDurVqIQBY/wSXUXkbABAkEA9sY6xPT0Xar6B4co+7Agg0rxQSYeBTCui0fbNlH1BgciYLlc35MCEQd1C7bQ2lYILQ3E/RX2TdmJOzRwTna0wQJBANCAtnori74EgT5Tt/avvB0JJTD14lh3tpGmgmc+/VAiLq6KymSqhjJJCWmEWjQ4+2elWwGMMvjtXIggU6ITGBUCQDXcTjduv2cySiAaf/dvGamOUdnGWAcJ+Q6tQEs62B/YrsgtrPefPnQ5DHLiE/OTk3VB7BzRDlEviaRCbcCyaAECQQC8qptz9Q9gA+eHJG4kdGQ9ydazdOs5rimvpgH9tsu0xRmEquX1LTB9NAWmSzLsaltjMAcnYVuGUBIYw1eqIqj5AkEA09+1I4EnMtuGLPf5hQ6OSSnrxWJAnWrJx+K2QWRrDdnmINCSTO6kYh0dRZRP9jPlaVfHhNUoqvXZrAaV+8YZqQ==\n-----END RSA PRIVATE KEY-----";
    (void)private_key;

    std::string app_id = "2015052600090779";
    std::string biz_content="{\"total_amount\":\"0.01\",\"subject\":\"1\",\"out_trade_no\":\"IQJZSRC1YMQB5HU\"}";
    std::string charset = "utf-8";
    std::string format="json";
    std::string method="alipay.trade.app.pay";
    std::string notify_url="http://domain.merchant.com/payment_notify";
    std::string sign_type="RSA2";
    std::string timestamp="2016-08-25 20:26:31";
    std::string version="1.0";
    std::map<std::string, std::string> data_sort{};
    data_sort["app_id"] = app_id;
    //data_sort["biz_content"] = biz_content;
    data_sort["charset"] = charset;
    data_sort["format"] = format;
    data_sort["method"] = method;
    //data_sort["notify_url"] = notify_url;
    data_sort["sign_type"] = sign_type;
    //data_sort["timestamp"] = timestamp;
    data_sort["version"] = version;

    std::ostringstream ostm;
    for (auto item : data_sort) {
        ostm << item.first << "=" << item.second;
        ostm << "&";
    }

    auto data_str = ostm.str();
    data_str.pop_back();

    /*
    BIO* key_bio = ::BIO_new_mem_buf((void*)private_key, -1);
    if (!key_bio) {
        std::cout << "key_bio null\n";
        //::BIO_free_all(key_bio);
        return false;
    }

    RSA* rsa = ::PEM_read_bio_RSAPrivateKey(key_bio, nullptr, nullptr, nullptr);
    if (!rsa) {
        std::cout << "rsq null\n";
        return false;
    }
    */
    FILE* f = ::fopen("bk_prv.key", "r");
    RSA* rsa = ::PEM_read_RSAPrivateKey(f, nullptr, nullptr, nullptr);
    if (!rsa) {
        std::cout << "rsq null\n";
         ::ERR_print_errors_fp(stdout);
        return false;
    }

    int sign_need_len = RSA_size(rsa);

    //const char* req_data = R"(app_id=2015052600090779&biz_content={"timeout_express":"30m","seller_id":"","product_code":"QUICK_MSECURITY_PAY","total_amount":"0.01","subject":"1","out_trade_no":"IQJZSRC1YMQB5HU"}&charset=utf-8&format=json&method=alipay.trade.app.pay&notify_url=http://domain.merchant.com/payment_notify&sign_type=RSA2&timestamp=2016-08-25 20:26:31&version=1.0")";
    //const char* req_data = "app_id=2015052600090779&sign_type=RSA2";
    //const char* req_data = "app_id=2015052600090779&version=1.0";
    const char* req_data = "sign_type=RSA2";
    req_data = data_str.c_str();

    std::cout << req_data << "\n";


    std::vector<unsigned char> encrypt_sign{};
    encrypt_sign.resize(sign_need_len);
    int encrypt_ret = RSA_private_encrypt((int)data_str.size(),(const unsigned char*)req_data, encrypt_sign.data(), rsa, RSA_PKCS1_PADDING);
    if (encrypt_ret == -1) {
        std::cout << "RSA_private_encrypt error\n";
         ::ERR_print_errors_fp(stdout);
         return false;
    }

    std::vector<unsigned char> sign{};
    sign.resize(sign_need_len);
    unsigned int sign_actually_len = 0;
    int r = RSA_sign(NID_sha256WithRSAEncryption, (const unsigned char*)encrypt_sign.data(), encrypt_ret,
        sign.data(), &sign_actually_len, rsa);
    if (r != 1) {
        std::cout << "rsa_sign error\n";
         ::ERR_print_errors_fp(stdout);
        return false;
    }

    /*
    std::vector<unsigned char> sign{};
    sign.resize(sign_need_len);
    unsigned int sign_actually_len = 0;
    int r = RSA_sign(NID_sha256WithRSAEncryption, (const unsigned char*)req_data, std::strlen(req_data),
        sign.data(), &sign_actually_len, rsa);
    if (r != 1) {
        std::cout << "rsa_sign error\n";
         ::ERR_print_errors_fp(stdout);
        return false;
    }
    */

    std::vector<unsigned char> b64{};
    for (unsigned int i = 0; i != sign_actually_len; ++i) {
        printf("%x", sign[i]);
        b64.push_back(sign[i]);
    }
    printf("\n");

    std::vector<char> b64_out{};
    b64_out.resize(((b64.size()/3) + 1) * 4 + 1);
    int b64_out_len = base64Encode(b64.data(), (int)b64.size(), b64_out.data());
    std::cout << b64_out_len << "\n";
    std::cout << b64_out.data();
    std::cout << "(===============)\n";
    return true;
}

bool fun2()
{
    auto s = catFile("bk_prv.key");
    //const char* private_key ="MIICXAIBAAKBgQDfNLRsq7J7KBUlXGUIzJkakGhQCjCu3TDUKdpYuo4fcsBBpQkM7q72GGsNQFAxnVvKvrFPrtChEnIFib4o6TyEyTezw66wh2/hab9UGPxljXIqwNv6DuHGxLNRgEaOX4uPFMoSDoEgpcVXxpIO3R65rv0dbGT4saSUNwJ85TycoQIDAQABAoGBAI0tXOFPSDi1hYp4Aj+qiTxQEmptx5USuou3XS+576LchdX/eNYBMhDeKPfcsdxv11tJegUYYUU/0XbHRMvDmk6EQGWve99PC7DmPWrtJ97Qw9cbBONgdn1rdgm0tyrXgfgIKJPYQQVf8mRFJdj0sNPykTBxK43PycaOJqK/dVWxAkEA8QMixRbz19hznUuqrRAERtRcpRCu9nUdSk9/K3GelU2XDrCl0+SC3cZhIfYXMoPKBqj/9Sp4ZyoUH7KcHotEpwJBAO0WGGTUdxQqKqr9QHdX/NQTXmhM2OJDa67jQjILT7JAC/iO4wWX3P2td0zsnFKgN/4RDUVA+sxrlv3PlqDyFXcCQD5YOlFTe6Z1NosU/MSh5QlRe9mzNB9K8lW7tMDPNl+W36GMLolejj/CRnQbjaqijsskQnnwD49YQjZk5J++FPcCQCUmKipadFIvjUH/rsNEgTRF2KwlJnLFt7DOoUewKAu5J2cKFJ6CvjjtnlzqUlBMFJn12At69BFl5mHczjBn3l8CQGeWztRa0APVe4DirwXSnl93+HwflpRykEZ6ZzrFa58PX+kHj80xY5PdTsNJ/zqMCTZd8JtGiftRp4YmJLysx8A=";
    //const char* private_key = "-----BEGIN RSA PRIVATE KEY----- MIICXAIBAAKBgQDfNLRsq7J7KBUlXGUIzJkakGhQCjCu3TDUKdpYuo4fcsBBpQkM 7q72GGsNQFAxnVvKvrFPrtChEnIFib4o6TyEyTezw66wh2/hab9UGPxljXIqwNv6 DuHGxLNRgEaOX4uPFMoSDoEgpcVXxpIO3R65rv0dbGT4saSUNwJ85TycoQIDAQAB AoGBAI0tXOFPSDi1hYp4Aj+qiTxQEmptx5USuou3XS+576LchdX/eNYBMhDeKPfc sdxv11tJegUYYUU/0XbHRMvDmk6EQGWve99PC7DmPWrtJ97Qw9cbBONgdn1rdgm0 tyrXgfgIKJPYQQVf8mRFJdj0sNPykTBxK43PycaOJqK/dVWxAkEA8QMixRbz19hz nUuqrRAERtRcpRCu9nUdSk9/K3GelU2XDrCl0+SC3cZhIfYXMoPKBqj/9Sp4ZyoU H7KcHotEpwJBAO0WGGTUdxQqKqr9QHdX/NQTXmhM2OJDa67jQjILT7JAC/iO4wWX 3P2td0zsnFKgN/4RDUVA+sxrlv3PlqDyFXcCQD5YOlFTe6Z1NosU/MSh5QlRe9mz NB9K8lW7tMDPNl+W36GMLolejj/CRnQbjaqijsskQnnwD49YQjZk5J++FPcCQCUm KipadFIvjUH/rsNEgTRF2KwlJnLFt7DOoUewKAu5J2cKFJ6CvjjtnlzqUlBMFJn1 2At69BFl5mHczjBn3l8CQGeWztRa0APVe4DirwXSnl93+HwflpRykEZ6ZzrFa58P X+kHj80xY5PdTsNJ/zqMCTZd8JtGiftRp4YmJLysx8A= -----END RSA PRIVATE KEY-----";
    std::string const_key = 
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIICXAIBAAKBgQDfNLRsq7J7KBUlXGUIzJkakGhQCjCu3TDUKdpYuo4fcsBBpQkM\n"
        "7q72GGsNQFAxnVvKvrFPrtChEnIFib4o6TyEyTezw66wh2/hab9UGPxljXIqwNv6\n"
        "DuHGxLNRgEaOX4uPFMoSDoEgpcVXxpIO3R65rv0dbGT4saSUNwJ85TycoQIDAQAB\n"
        "AoGBAI0tXOFPSDi1hYp4Aj+qiTxQEmptx5USuou3XS+576LchdX/eNYBMhDeKPfc\n"
        "sdxv11tJegUYYUU/0XbHRMvDmk6EQGWve99PC7DmPWrtJ97Qw9cbBONgdn1rdgm0\n"
        "tyrXgfgIKJPYQQVf8mRFJdj0sNPykTBxK43PycaOJqK/dVWxAkEA8QMixRbz19hz\n"
        "nUuqrRAERtRcpRCu9nUdSk9/K3GelU2XDrCl0+SC3cZhIfYXMoPKBqj/9Sp4ZyoU\n"
        "H7KcHotEpwJBAO0WGGTUdxQqKqr9QHdX/NQTXmhM2OJDa67jQjILT7JAC/iO4wWX\n"
        "3P2td0zsnFKgN/4RDUVA+sxrlv3PlqDyFXcCQD5YOlFTe6Z1NosU/MSh5QlRe9mz\n"
        "NB9K8lW7tMDPNl+W36GMLolejj/CRnQbjaqijsskQnnwD49YQjZk5J++FPcCQCUm\n"
        "KipadFIvjUH/rsNEgTRF2KwlJnLFt7DOoUewKAu5J2cKFJ6CvjjtnlzqUlBMFJn1\n"
        "2At69BFl5mHczjBn3l8CQGeWztRa0APVe4DirwXSnl93+HwflpRykEZ6ZzrFa58P\n"
        "X+kHj80xY5PdTsNJ/zqMCTZd8JtGiftRp4YmJLysx8A=\n"
        "-----END RSA PRIVATE KEY-----\n";
    //std::cout << s.size() << "  " << const_key.size() << "\n";
    //std::cout << (s == const_key) << "\n";

    size_t min_len = std::min(s.size(), const_key.size());
    for (size_t i = 0; i != min_len; ++i) {
        if (s[i] != const_key[i]) {
            //std::cout << i << "\t\t" << (int)s[i] << "\t\t" << (int)const_key[i] << "\n";
        }
    }

    /*
    std::cout << (s.substr(0, 32) == const_key.substr(0, 32)) << "\n";
    std::cout << s[32] << "\n";
    std::cout << const_key[32] << "\n";
    */
    const char*  private_key = const_key.c_str();


    //const char* private_key = s.c_str();
    BIO* key_bio = ::BIO_new_mem_buf((void*)private_key, -1);
    if (!key_bio) {
        std::cout << "key_bio null\n";
        return false;
    }
    /*
    RSA* rsa_ex = ::PEM_read_bio_RSAPrivateKey(key_bio, nullptr, nullptr, nullptr);
    if (!rsa_ex) {
        std::cout << "rsq null\n";
        return false;
    }
    */

    EVP_PKEY* evp_pkey_ex = PEM_read_bio_PrivateKey(key_bio, nullptr, nullptr, nullptr);
    if (!evp_pkey_ex) {
        std::cout << "evp_pkey_ex null\n";
        return false;
    }

    auto content = catFile("x.txt");
    if (content.empty()) {
        std::cout << "content error\n";
        return false;
    }
    //std::cout << content << "\n";

    FILE* f = ::fopen("bk_prv.key", "r");
    RSA* rsa = ::PEM_read_RSAPrivateKey(f, nullptr, nullptr, nullptr);
    if (!rsa) {
        std::cout << "rsq null\n";
         ::ERR_print_errors_fp(stdout);
        return false;
    }

    int ret = 0;

    /*
    EVP_PKEY* evp_pkey = EVP_PKEY_new();
    ret = EVP_PKEY_set1_RSA(evp_pkey, rsa_ex);
    if (ret != 1) {
        std::cout << "EVP_PKEY_set1_RSA error\n";
        ::ERR_print_errors_fp(stdout);
        return false;
    }
    */

    EVP_MD_CTX* evp_md_ctx = EVP_MD_CTX_create();
    ret = EVP_DigestSignInit(evp_md_ctx, nullptr, EVP_sha256(), nullptr, evp_pkey_ex);
    if (ret != 1) {
        std::cout << "EVP_DigestSignInit error\n";
        ::ERR_print_errors_fp(stdout);
        return false;
    }

    ret = EVP_DigestSignUpdate(evp_md_ctx, content.data(), content.size());
    if (ret != 1) {
        std::cout << "EVP_DigestSignUpdate error\n";
        ::ERR_print_errors_fp(stdout);
        return false;
    }

    size_t len = 0;
    ret = EVP_DigestSignFinal(evp_md_ctx, nullptr, &len);
    if (ret != 1) {
        std::cout << "EVP_DigestSignFinal get len error\n";
        ::ERR_print_errors_fp(stdout);
        return false;
    }
    //std::cout << len << "\n";

    std::vector<uint8_t> buff{};
    buff.resize(len+1);
    len += 1;
    ret = EVP_DigestSignFinal(evp_md_ctx, buff.data(), &len);
    if (ret != 1) {
        std::cout << "EVP_DigestSignFinal get buff error\n";
        ::ERR_print_errors_fp(stdout);
        return false;
    }
    //std::cout << len << "\n";

    std::vector<char> b64_out{};
    b64_out.resize(((buff.size() / 3) + 1) * 4 + 1);
    int b64_out_len = base64Encode(buff.data(), len, b64_out.data());
    (void)b64_out_len;
    //std::cout << b64_out_len << "\n";
    std::cout << b64_out.data();
    //std::cout << "(===============)\n";
    return true;
}

bool fun3()
{
    std::string const_key = 
    "-----BEGIN RSA PRIVATE KEY-----\n"
    "MIIEpAIBAAKCAQEAqv9PyJmtt1SqeNn7HnrBTi9uG0159YvYs0Aj7L3brlIMuf0p\n"
    "FeSXUxcIkDswzATz4DAmgd25pavoF7SvsBGgVH1OSyNZ5pL0px1cwM4vxIy++KLC\n"
    "V5ZadaEKasiZQqV2UkxCOIsmrfdHytawydwx/EBC/Vy4Rg66Vdm6Bu3mCf4aTisy\n"
    "SCjSkGSTtyMmRjIVlO47JVjy7lAbHnc19LYDBkE60xfZHO8kF82xyoMG94IXzc2q\n"
    "aoslQkOQfIo19JpmvDlEdzC2WkBU9QxZ8jjW+sqQmzKy354qdVDkfrT3dqvsk3NK\n"
    "RM621kb2Xhd6ZEZDAxc6/iTjPqJPn76aWNfBKwIDAQABAoIBADnZE4mei8k4qE8o\n"
    "Fy8xghHMM+iipj6uZ8oESeL+O8JWWZ3WCj8wzQy7/xakH5b1mfde8rf+ZZ6pcGQM\n"
    "qV7cJ6xNk94RgIljb8bVRRqo5joND1IX5b0xzTp8F4UWhdqpFUU0LTbpxFUYEb6K\n"
    "MNapWnOm3cLOyjxRITKH7MoJU0hPk7Ri/SBb+P3iPPehYlVUs54HsQRMkdLu9U3e\n"
    "mfwl8ouu6ES7J2ZDDqD5mUlLAfpTHe145aP3a/OEsGPEdLDEDFOF5rQLSfSODzkR\n"
    "M1RVOxJcCZQNBVNRtpowXEO76Ha22teVC94phlVO0lqKJE/bXBH6yL+mfoJ0jKtl\n"
    "I8Ib6FkCgYEA2aAJyQ0Vd6VVjDvRY40Ui5B1MgQ6c1wIc2rYZnQXxS+G9+Ih+Lxf\n"
    "vnkX9Dzpwqc7FakR39EmVt2om1ptGPl2Nphew7/CajWV9C6s9OYVofmZ85CaQVCt\n"
    "YF8vb+NQ2CuxOjk1snhr+z5vzS9dS6JxRU+cwm81uQ0b9igpP58OH9cCgYEAySZq\n"
    "RpwojPl0Zg/CM7XENxHXGAZNFmYmuK1aBHsuMl2fF8XLWUndo1kI9yYdUoYMgqOT\n"
    "PQcY8idO+Gg/6M+UOUo4ZbK1WADe8Nct+2+X9jUUQXPe36POC+oyPTGWZNeuwn1X\n"
    "9dHvYFqMm7iE09PcLIHUYzcVtKBcLgeGlaI5js0CgYEAnDMpMtwoP2M1Ht1Ech6t\n"
    "02qRI6AzT9+L1UOoJQlIkmiFiGAPoBec5PrHfa9G3UbouqhKQwo8aXbZcQAbdCSB\n"
    "e7grtHZrlY32nnft/i0y87pSjKaKgTzl5FkNlFJlEXNy1mZ/qySR5Jgw6OYJIaNr\n"
    "h6MX0dq+hZ7qJek3igmguqsCgYBUcQP8UGHVjIMItWTZQFz2oU6ij0KdPJTUwjEf\n"
    "4HYsPpEi8a3D9fZNNHtHBYEZu5xU20ZcQDaAsW+9aEYr/bhDtJyoVLU8FGGCyVJM\n"
    "UzR/7xhpwbK2P5Wn+tDMT5zLCKAclXHviAntcRXF6VgZdL7hED+QxvcdtJP93rro\n"
    "5gkzoQKBgQDFZ720+TSkFfP5j/cpalce0n22V2OgnO9qgJk7Lvy7iUH8LpGJSZiL\n"
    "RkM1J6m1iD4yf4bFYgvu6TFzURHb1qXEBchncU/ZnGrCK+/BkKc40jpSs4YDzQkz\n"
    "juUk2MXBXqJ8ivBjMRKAfrOQnvTWs1XZZ12rALxLdwL2lUH8/A5n8w==\n"
    "-----END RSA PRIVATE KEY-----\n";

    const char*  private_key = const_key.c_str();
    BIO* key_bio = ::BIO_new_mem_buf((void*)private_key, -1);
    if (!key_bio) {
        std::cout << "key_bio null\n";
        return false;
    }

    EVP_PKEY* evp_pkey_ex = PEM_read_bio_PrivateKey(key_bio, nullptr, nullptr, nullptr);
    if (!evp_pkey_ex) {
        std::cout << "evp_pkey_ex null\n";
        return false;
    }

    auto content = catFile("ali_test.txt");
    if (content.empty()) {
        std::cout << "content error\n";
        return false;
    }

    int ret = 0;

    EVP_MD_CTX* evp_md_ctx = EVP_MD_CTX_create();
    ret = EVP_DigestSignInit(evp_md_ctx, nullptr, EVP_sha256(), nullptr, evp_pkey_ex);
    if (ret != 1) {
        std::cout << "EVP_DigestSignInit error\n";
        ::ERR_print_errors_fp(stdout);
        return false;
    }

    ret = EVP_DigestSignUpdate(evp_md_ctx, content.data(), content.size());
    if (ret != 1) {
        std::cout << "EVP_DigestSignUpdate error\n";
        ::ERR_print_errors_fp(stdout);
        return false;
    }

    size_t len = 0;
    ret = EVP_DigestSignFinal(evp_md_ctx, nullptr, &len);
    if (ret != 1) {
        std::cout << "EVP_DigestSignFinal get len error\n";
        ::ERR_print_errors_fp(stdout);
        return false;
    }
    //std::cout << len << "\n";

    std::vector<uint8_t> buff{};
    buff.resize(len+1);
    len += 1;
    ret = EVP_DigestSignFinal(evp_md_ctx, buff.data(), &len);
    if (ret != 1) {
        std::cout << "EVP_DigestSignFinal get buff error\n";
        ::ERR_print_errors_fp(stdout);
        return false;
    }
    //std::cout << len << "\n";

    std::vector<char> b64_out{};
    b64_out.resize(((buff.size() / 3) + 1) * 4 + 1);
    int b64_out_len = base64Encode(buff.data(), len, b64_out.data());
    (void)b64_out_len;
    //std::cout << b64_out_len << "\n";
    std::cout << b64_out.data();
    //std::cout << "(===============)\n";
    return true;
}

bool fun4()
{
    ERR_load_crypto_strings();
    std::string new_sign = "BwKTUnUJI3chWmT4xBfQ+ySoMtnXN/gIaAjoM0C3ZAQICp6ReCZ87j4HPw44JOncM4zWgJ1lHZx+DGHkDgxWV5n/j+Q/AQV6e0rEnFw5ZD+SL9yz3Nr2XUE6vyYWl06yYYE/vf7Nsk0enJBKFuv5TFsQ5+nOVQf5fN2YYlGdbnFpXmOvTM4amPlsZwUv8Y8yhBEnktKx8XiDFGW3p0R3gn0L7vIci4nliP2Qmom3kAxXWBvv4i/auQIM5nekzt70wu9Gtd115QqXBycSC5+4g2MahmBHbf03QP+TMlIGJm+kiWZAXo+amGv7cwiSZwqofjNr0mINLTCWv5P2+CSuVw==";
    std::string new_sign_2 =
    "BwKTUnUJI3chWmT4xBfQ+ySoMtnXN/gIaAjoM0C3ZAQICp6ReCZ87j4HPw44JOnc"
    "M4zWgJ1lHZx+DGHkDgxWV5n/j+Q/AQV6e0rEnFw5ZD+SL9yz3Nr2XUE6vyYWl06y"
    "YYE/vf7Nsk0enJBKFuv5TFsQ5+nOVQf5fN2YYlGdbnFpXmOvTM4amPlsZwUv8Y8y"
    "hBEnktKx8XiDFGW3p0R3gn0L7vIci4nliP2Qmom3kAxXWBvv4i/auQIM5nekzt70"
    "wu9Gtd115QqXBycSC5+4g2MahmBHbf03QP+TMlIGJm+kiWZAXo+amGv7cwiSZwqo"
    "fjNr0mINLTCWv5P2+CSuVw==";

    std::cout << new_sign.size() << " " << new_sign_2.size() << " " << (new_sign == new_sign_2) << "\n";
    new_sign = new_sign_2;

    std::string new_sign_ex = catFile("bbb/sign.txt");
    std::vector<unsigned char> sign_data{};
    {
        std::vector<unsigned char> temp{};
        temp.reserve(new_sign.size());
        for (const auto& v : new_sign) {
            temp.push_back((unsigned char)v);
        }

        std::vector<unsigned char> buff{};
        buff.resize(temp.size());
        int len = openssl::base64Decode(temp.data(), temp.size(), buff.data());
        if (len == -1) {
            std::cout << "error sgin base64Decode\n";
            return false;
        }
        std::cout << "len " << len << "\n";
        const unsigned char* pos = (unsigned char*)buff.data();
        std::copy(pos, pos + len, std::back_inserter(sign_data));
    }

    auto sign_data_ex = catBinary("bbb/s.txt");
    std::cout << sign_data.size() << " " << sign_data_ex.size() << "\n";
    //sign_data = sign_data_ex;

    FILE* f = ::fopen("./alipay.key", "r");
    RSA* alipay_public_key = ::PEM_read_RSA_PUBKEY(f, nullptr, nullptr, nullptr);
    ::fclose(f);
    if (!alipay_public_key) {
        std::cout << "PEM_read_RSA_PUBKEY null\n";
        return false;
    }
    std::string content = "app_id=2017022405847524&auth_app_id=2017022405847524&buyer_id=2088002278416202&buyer_logon_id=gua***@163.com&buyer_pay_amount=0.01&charset=utf-8&fund_bill_list=[{\"amount\":\"0.01\",\"fundChannel\":\"ALIPAYACCOUNT\"}]&gmt_create=2017-03-22 16:36:03&gmt_payment=2017-03-22 16:36:04&invoice_amount=0.01&notify_id=a040984dbd360e1a39ddaeacbba4bcahjm&notify_time=2017-03-22 16:36:05&notify_type=trade_status_sync&out_trade_no=ZALI2017032216354718101700001&point_amount=0.00&receipt_amount=0.01&seller_email=zhangzhuotec123@163.com&seller_id=2088421635751581&subject=游戏充值&total_amount=0.01&trade_no=2017032221001004200225124878&trade_status=TRADE_SUCCESS&version=1.0";
    std::string content_ex = catFile("bbb/x.txt");
    std::cout << (content == content_ex) << "\n";
    content = content_ex;

    std::vector<char> content_buffer(content.begin(), content.end());
    EVP_PKEY* evp_pkey{ ::EVP_PKEY_new() };
    int ret = ::EVP_PKEY_set1_RSA(&*evp_pkey, alipay_public_key);
    if (ret != 1) {
        std::cout << "EVP_PKEY_set1_RSA error";
        return false;
    }

    EVP_MD_CTX* evp_md_ctx{ ::EVP_MD_CTX_create() };
    ret = ::EVP_DigestVerifyInit(&*evp_md_ctx, nullptr, EVP_sha256(), nullptr, &*evp_pkey);
    if (ret != 1) {
        std::cout << "EVP_DigestVerifyInit error";
        return false;
    }

    ret = ::EVP_DigestVerifyUpdate(&*evp_md_ctx, content_buffer.data(), content_buffer.size());
    if (ret != 1) {
        std::cout << "EVP_DigestVerifyUpdate error";
        return false;
    }

    ret = ::EVP_DigestVerifyFinal(&*evp_md_ctx, sign_data.data(), sign_data.size());
    if (ret == 0) {
        std::cout << "EVP_DigestVerifyFinal get sign data error\n";
        //std::cout << ERR_get_error() << "\n";
        std::cout << ERR_error_string(ERR_get_error(), nullptr) << "\n";
        return false;
    }
    std::cout << "success\n";
    return true;
}


bool fun5_openssl()
{
    ERR_load_crypto_strings();
    std::string sign_str =
        "Zfh+5O3s8a1VJITqiqm/BVbZ+IKGrUyhM4sKOWIu5xmBecyyacD9QmUrVTToQiLZkUgqFv4P7SNYT+n07kBkdBHExvoKpYVXqtxL4NjpjPMZ9LtNLWrzdwVraOkDqHNI35Vlj7GW+76auUKsuDa6hOT8qcwiCoA2PoBIunZchNIgtX2CeN0Gg1hlNm8dQVYhBL4F5O/GuykPP6UoeonGk6DJot2X0Z77C+ewyaoDH4HlrwcgqFYZ2tW4DAMDvaTN2Uvo+Gv8eBfG0U2hoA0zrde6vGvn1yn2VFivkzdIRrwHX+SWh90c8LTkbumkldvEvEXlMGVnKMiomzLngjMz4w==";

    FILE* f = ::fopen("./alipay.key", "r");
    RSA* alipay_public_key = ::PEM_read_RSA_PUBKEY(f, nullptr, nullptr, nullptr);
    ::fclose(f);
    if (!alipay_public_key) {
        std::cout << "PEM_read_RSA_PUBKEY null\n";
        return false;
    }
  //std::string content = "app_id=2017022405847524&auth_app_id=2017022405847524&buyer_id=2088002278416202&buyer_logon_id=gua***@163.com&buyer_pay_amount=0.01&charset=utf-8&fund_bill_list=[{\"amount\":\"0.01\",\"fundChannel\":\"ALIPAYACCOUNT\"}]&gmt_create=2017-03-22 16:36:03&gmt_payment=2017-03-22 16:36:04&invoice_amount=0.01&notify_id=a040984dbd360e1a39ddaeacbba4bcahjm&notify_time=2017-03-22 16:36:05&notify_type=trade_status_sync&out_trade_no=ZALI2017032216354718101700001&point_amount=0.00&receipt_amount=0.01&seller_email=zhangzhuotec123@163.com&seller_id=2088421635751581&subject=游戏充值&total_amount=0.01&trade_no=2017032221001004200225124878&trade_status=TRADE_SUCCESS&version=1.0";
    std::string content = "app_id=2017022405847524&auth_app_id=2017022405847524&buyer_id=2088002278416202&buyer_logon_id=gua***@163.com&buyer_pay_amount=0.01&charset=utf-8&fund_bill_list=[{\"amount\":\"0.01\",\"fundChannel\":\"ALIPAYACCOUNT\"}]&gmt_create=2017-03-23 14:11:42&gmt_payment=2017-03-23 14:11:42&invoice_amount=0.01&notify_id=c5eff9ba336ebfbd0cfc634bd833b34hjm&notify_time=2017-03-23 14:11:43&notify_type=trade_status_sync&out_trade_no=ZALI2017032314113071767400001&point_amount=0.00&receipt_amount=0.01&seller_email=zhangzhuotec123@163.com&seller_id=2088421635751581&subject=游戏充值&total_amount=0.01&trade_no=2017032321001004200226494405&trade_status=TRADE_SUCCESS&version=1.0";
    auto succ = openssl::digitVerify_RSA_SHA256_Base64(content.data(), content.size(), sign_str, alipay_public_key);
    if (succ) {
        std::cout << "success\n";
    } else {
        std::cout << "fail\n";
    }
    return true;
}



int main()
{
    fun5_openssl();
    std::string s = "2017-03-23+14%3A11%3A43 a+b+c";

    std::string ss;
    auto cnt = stringReplace(s, '+', "%20", &ss);

    curl::CURL_Guard curl_guard{};
    std::string out;
    if (!curl_guard.urlDecode(ss, &out)) {
        std::cout << "urldecode error\n";
        return 0;
    }
    std::cout << cnt << "\n" << out << "\n";

    return 0;
}
