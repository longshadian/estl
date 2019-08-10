#pragma once

#include <string>

class Encryption
{
public:
    static bool VerifySign(const std::string& json, const std::string& sign);
    static bool StartRsaEncrypt(const std::string& input_data, std::string* output);
    static std::string CatFile(const char* f);
    static bool CatFile(const std::string& path, std::string* out);

private:
    static std::string DecryptSign(const std::string& sign);
    //static std::string rsa_pub_split128_decrypt_new(const std::string& clearText, const std::string& pubKey);
    static std::string StartRsaDecrypt(const std::string& input, const std::string& pub_key);

    static std::string ToUpperCase(const std::string& src);
    static std::string ToLowerCase(const std::string& src);
};
