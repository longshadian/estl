#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <sstream>

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
        snprintf(buff.data(), buff.size(), "%02X ", c);
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

int main()
{
    aesEncode();
}
