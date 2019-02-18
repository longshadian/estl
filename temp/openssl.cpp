#include <iostream>
#include <string>
#include <vector>

#include <openssl/rsa.h>
#include <openssl/err.h>
#include <openssl/pem.h>

void encrypt(std::vector<uint8_t> str, std::vector<uint8_t>* out)
{
    FILE* f = ::fopen("./private.key", "r");
    RSA* key = ::PEM_read_RSAPrivateKey(f, nullptr, nullptr, nullptr);
    ::fclose(f);
    if (!key) {
        std::cout << "encrypt PEM_read_RSA_PUBKEY fail\n";
        ::ERR_print_errors_fp(stdout);
        return;
    }

    int rsa_len = ::RSA_size(key);
    out->resize(rsa_len);
    int ret = ::RSA_private_encrypt((int)str.size(), str.data()
        , out->data(), key, RSA_PKCS1_PADDING);
    if (ret == -1) {
        std::cout << "encrypt RSA_private_encrypt fail\n";
        ::ERR_print_errors_fp(stdout);
        return;
    }
    std::cout << "ret: " << ret << "\n";
    ::RSA_free(key);
    out->resize(ret);
    std::cout << "encrypt success size: " << out->size() << " \n";
}

void decrypt(std::vector<uint8_t> str, std::vector<uint8_t>* out)
{
    FILE* f = ::fopen("./public.key", "r");
    RSA* key = ::PEM_read_RSA_PUBKEY(f, nullptr, nullptr, nullptr);
    ::fclose(f);
    if (!key) {
        std::cout << "decrypt PEM_read_RSA_PUBKEY fail\n";
        ::ERR_print_errors_fp(stdout);
        return;
    }

    int rsa_len = ::RSA_size(key);
    out->resize(rsa_len);
    int ret = ::RSA_public_decrypt((int)str.size(), str.data()
        , out->data(), key, RSA_PKCS1_PADDING);
    if (ret == -1) {
        std::cout << "decrypt RSA_public_encrypt fail\n";
        ::ERR_print_errors_fp(stdout);
    }
    ::RSA_free(key);
    out->resize(ret);
    std::cout << "decrypt success out size: " << out->size() << "\n";
}

std::vector<uint8_t> toBinary(const std::string& str)
{
    return std::vector<uint8_t>{str.begin(), str.end()};
}

std::string toString(const std::vector<uint8_t>& buf)
{
    return std::string{buf.begin(), buf.end()};
}

int main()
{
    std::string str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    std::vector<uint8_t> out{};
    encrypt(toBinary(str), &out);

    std::vector<uint8_t> out2{};
    decrypt(out, &out2);

    std::cout << out.size() << "  " << out2.size() << "\n";

    auto str_ex = toString(out2);
    std::cout << (str == str_ex) << "\n";
}
