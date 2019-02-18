#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <sstream>
#include <chrono>
#include <thread>

#include <openssl/rsa.h>
#include <openssl/err.h>
#include <openssl/pem.h>
#include <openssl/aes.h>

std::vector<uint8_t> toBinary(const std::string& str)
{
    return std::vector<uint8_t>{str.begin(), str.end()};
}

std::string toString(const std::vector<uint8_t>& buf)
{
    return std::string{buf.begin(), buf.end()};
}

#define alignment16(a) ((a+16-1)&(~(16-1)))
#define alignment16_ex(a) ((a)&(~(16-1)))

void padding16(std::vector<uint8_t>* data) {
    size_t s = data->size();
    size_t need = alignment16(s);
    data->resize(need);
}

std::string toHexString(const std::vector<uint8_t>& data) {
    std::ostringstream ostm{};
    std::array<char, 10> buff{};
    for (auto c : data) {
        snprintf(buff.data(), buff.size(), "0x%02X ", c);
        ostm << buff.data();
    }
    return ostm.str();
}


bool aesEncode()
{
    std::vector<uint8_t> data{};
    data.resize(64, 0x99);
    padding16(&data);

    {
        auto s = toHexString(data);
        printf("data:\t\t%s\n", s.c_str());
    }

    std::array<uint8_t, AES_BLOCK_SIZE> key{};
    key.fill(0x81);
    AES_KEY aesKey{};
    int ret = 0;
    ret = AES_set_encrypt_key(key.data(), key.size() * 8, &aesKey);
    if (ret != 0) {
        std::cout << __LINE__ << ":ERROR: ret: " << ret << "\n";
        return false;
    }

    std::vector<uint8_t> data_encrypt{};
    data_encrypt.resize(data.size());
    for (size_t i = 0; i != data.size() / 16; ++i) {
        auto* pos_in = data.data() + 16 * i;
        auto* pos_out = data_encrypt.data() + 16 * i;
        AES_ecb_encrypt(pos_in, pos_out, &aesKey, AES_ENCRYPT);
    }

    {
        auto s = toHexString(data_encrypt);
        printf("data_e:\t\t%s\n", s.c_str());
    }

    AES_KEY aesKey2{};
    ret = AES_set_decrypt_key(key.data(), key.size() * 8, &aesKey2);
    if (ret != 0) {
        std::cout << __LINE__ << ":ERROR: ret: " << ret << "\n";
        return false;
    }
    std::vector<uint8_t> data_bk{};
    data_bk.resize(data_encrypt.size());

    for (size_t i = 0; i != data.size() / 16; ++i) {
        auto* pos_in = data_encrypt.data() + 16 * i;
        auto* pos_out = data_bk.data() + 16 * i;
        AES_ecb_encrypt(pos_in, pos_out, &aesKey2, AES_DECRYPT);
    }

    {
        auto s = toHexString(data_bk);
        printf("data_o:\t\t%s\n", s.c_str());
    }
}

bool aesEncode_Ecb()
{
    size_t data_len = 1024 * 10;
    std::vector<uint8_t> data{};
    data.resize(data_len, 0x99);
    padding16(&data);

    std::vector<uint8_t> data_encrypt{};
    data_encrypt.resize(data.size());

    std::vector<uint8_t> data_bk{};
    data_bk.resize(data_encrypt.size());

    std::array<uint8_t, 32> key{};
    key.fill(0x81);
    AES_KEY aesKey{};
    int ret = 0;
    ret = AES_set_encrypt_key(key.data(), key.size() * 8, &aesKey);
    if (ret != 0) {
        std::cout << __LINE__ << ":ERROR: ret: " << ret << "\n";
        return false;
    }

    AES_KEY aesKey2{};
    ret = AES_set_decrypt_key(key.data(), key.size() * 8, &aesKey2);
    if (ret != 0) {
        std::cout << __LINE__ << ":ERROR: ret: " << ret << "\n";
        return false;
    }

    {
        const int count = 100000;
        int cnt = 0;
        auto tbegin = std::chrono::system_clock::now();
        while (cnt++ < count) {
            for (size_t i = 0; i != data.size() / 16; ++i) {
                auto* pos_in = data.data() + 16 * i;
                auto* pos_out = data_encrypt.data() + 16 * i;
                AES_ecb_encrypt(pos_in, pos_out, &aesKey, AES_ENCRYPT);
            }

            for (size_t i = 0; i != data.size() / 16; ++i) {
                auto* pos_in = data_encrypt.data() + 16 * i;
                auto* pos_out = data_bk.data() + 16 * i;
                AES_ecb_encrypt(pos_in, pos_out, &aesKey2, AES_DECRYPT);
            }
        }

        auto tend = std::chrono::system_clock::now();

        std::cout << "data_len: " << data_len
            << " cnt: " << count
            << "  cost:" << std::chrono::duration_cast<std::chrono::milliseconds>(tend - tbegin).count() 
            << "ms"
            << std::endl;
    }

    return true;
    {
        auto s = toHexString(data_encrypt);
        printf("data_e:\t\t%s\n", s.c_str());
    }

    {
        auto s = toHexString(data_bk);
        printf("data_o:\t\t%s\n", s.c_str());
    }
}

void PKCS7Padding(std::vector<uint8_t>* data, size_t k) 
{
    size_t padding_len = static_cast<uint8_t>(k - (data->size() % k));
    uint8_t padding_val = static_cast<uint8_t>(padding_len);
    data->resize(data->size() + padding_len, padding_val);
}

int PKCS7PaddingLength(const void* p, size_t len, size_t k) 
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

bool aesEncode_Cbc()
{
    size_t data_len = 10;
    std::vector<uint8_t> data{};
    data.resize(data_len, 'a');

    std::cout << "data:\n" << toHexString(data) << "\n\n";

    std::vector<uint8_t> data_padding = data;
    PKCS7Padding(&data_padding, 16);
    //std::cout << "data_padding:\n" << toHexString(data_padding) << "\n\n";

    std::vector<uint8_t> data_encrypt{};
    data_encrypt.resize(data_padding.size());

    std::vector<uint8_t> data_bk{};
    data_bk.resize(data_encrypt.size());

    std::array<uint8_t, AES_BLOCK_SIZE> iv;
    iv.fill('a');

    std::array<uint8_t, AES_BLOCK_SIZE> iv2 = iv;

    std::array<uint8_t, 32> key{};
    key.fill(0x81);
    AES_KEY aesKey{};
    int ret = 0;
    ret = AES_set_encrypt_key(key.data(), key.size() * 8, &aesKey);
    if (ret != 0) {
        std::cout << __LINE__ << ":ERROR: ret: " << ret << "\n";
        return false;
    }

    AES_KEY aesKey2{};
    ret = AES_set_decrypt_key(key.data(), key.size() * 8, &aesKey2);
    if (ret != 0) {
        std::cout << __LINE__ << ":ERROR: ret: " << ret << "\n";
        return false;
    }

    {
        const int count = 10000;
        int cnt = 0;
        auto tbegin = std::chrono::system_clock::now();

            /*
            for (size_t i = 0; i != data.size() / 16; ++i) {
                auto* pos_in = data.data() + 16 * i;
                auto* pos_out = data_encrypt.data() + 16 * i;
                AES_ecb_encrypt(pos_in, pos_out, &aesKey, AES_ENCRYPT);
            }
            */

        AES_cbc_encrypt(data_padding.data(), data_encrypt.data(), data_padding.size(), &aesKey, iv.data(), AES_ENCRYPT);

        //std::cout << "encrypt:\n" << toHexString(data_encrypt) << "\n\n";

        AES_cbc_encrypt(data_encrypt.data(), data_bk.data(), data_encrypt.size(), &aesKey2, iv2.data(), AES_DECRYPT);

        //std::cout << "decrypt:\n" << toHexString(data_bk) << "\n\n";

        int ret = PKCS7PaddingLength(data_bk.data(), data_bk.size(), 16);
        if (ret == -1) {
            std::cout << "remove padding error!\n";
            return false;
        }

        data_bk.resize(data_bk.size() - ret);

        //std::cout << "finally:\n" << toHexString(data_bk) << "\n\n";
        /*
        for (size_t i = 0; i != data.size() / 16; ++i) {
            auto* pos_in = data_encrypt.data() + 16 * i;
            auto* pos_out = data_bk.data() + 16 * i;
            AES_ecb_encrypt(pos_in, pos_out, &aesKey2, AES_DECRYPT);
        }
        */

        std::cout << "success: " << (toHexString(data) == toHexString(data_bk)) << "\n";

        auto tend = std::chrono::system_clock::now();

        std::cout << "data_len: " << data_len
            << " cnt: " << count
            << "  cost:" << std::chrono::duration_cast<std::chrono::milliseconds>(tend - tbegin).count() 
            << "ms"
            << std::endl;
    }
}

void TestPKCS7Padding() 
{
    for (int k = 1; k != 17; ++k) {
        std::vector<uint8_t> d{};
        int total = k;
        for (int i = 0; i != total; ++i) {
            d.push_back('a');
        }
        PKCS7Padding(&d, 16);
        std::cout << "size: " << d.size() << " " << toHexString(d) << "\n";
    }
}

void TestRemovePKCS7Padding()
{
    for (size_t k = 1; k != 17; ++k) {
        std::vector<uint8_t> d{};
        int total = k;
        for (int i = 0; i < total; ++i) {
            d.push_back('a');
        }
        PKCS7Padding(&d, 16);
        std::cout << "size: " << d.size() << " " << toHexString(d) << "\n";

        int v = PKCS7PaddingLength(d.data(), d.size(), 16);
        if (v == -1) {
            std::cout << "remove padding error!\n";
            return;
        }
        std::cout << "v: " << v << "\n";
        d.resize(d.size() - v);
        std::cout << "size: " << d.size() << " " << toHexString(d) << "\n";

    }
}

int main()
{
    //aesEncode_Ecb();
    //aesEncode_Cbc();
    //TestPKCS7Padding();
    //TestRemovePKCS7Padding();

    aesEncode_Cbc();

    return 0;
}
