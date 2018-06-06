#pragma once

#include <cstdint>
#include <cstddef>
#include <array>

#include <openssl/aes.h>

void SimpleEncrypt(const uint8_t* src, size_t len, const uint8_t* key, size_t key_len, uint8_t* out);
void SimpleDecrypt(const uint8_t* src, size_t len, const uint8_t* key, size_t key_len, uint8_t* out);

size_t PKCS7Padding_Length(size_t i, size_t k);
int PKCS7Padding_Check(const void* p, size_t len, size_t k);

class AesCrypto
{
public:
    AesCrypto();
    ~AesCrypto();
    AesCrypto(const AesCrypto& rhs) = delete;
    AesCrypto& operator=(const AesCrypto& rhs) = delete;
    AesCrypto(AesCrypto&& rhs) = delete;
    AesCrypto& operator=(AesCrypto&& rhs) = delete;

protected:
    int SetKey(const void* p, size_t len);

    std::array<uint8_t, 32> m_key;
    size_t                  m_key_length;
    AES_KEY                 m_encrypt_key;
    AES_KEY                 m_decrypt_key;
};

class AesEbc : public AesCrypto
{
public:
    AesEbc();
    ~AesEbc();
    AesEbc(const AesEbc& rhs) = delete;
    AesEbc& operator=(const AesEbc& rhs) = delete;
    AesEbc(AesEbc&& rhs) = delete;
    AesEbc& operator=(AesEbc&& rhs) = delete;

    int SetKey(const void* p, size_t len);
    void Encrypt(const void* p, size_t len, void* out) const;
    void Decrypt(const void* p, size_t len, void* out) const;
};

class AesCbc : public AesCrypto
{
public:
    AesCbc();
    ~AesCbc();
    AesCbc(const AesCbc& rhs) = delete;
    AesCbc& operator=(const AesCbc& rhs) = delete;
    AesCbc(AesCbc&& rhs) = delete;
    AesCbc& operator=(AesCbc&& rhs) = delete;

    int SetKey(const void* p, size_t len);
    void SetIV(const void* p);
    void Encrypt(const void* p, size_t len, void* out) const;
    void Decrypt(const void* p, size_t len, void* out) const;

private:
    std::array<uint8_t, AES_BLOCK_SIZE> m_iv;
};
