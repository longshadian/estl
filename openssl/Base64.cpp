#include "Openssl.h"

#include <cstring>
#include <string>
#include <iostream>
#include <vector>

char* base64_decode(char *input, int length, bool with_new_line)
{
    BIO * b64 = NULL;
    BIO * bmem = NULL;
    char * buffer = (char *)malloc(length);
    memset(buffer, 0, length);

    b64 = BIO_new(BIO_f_base64());
    if (!with_new_line) {
        BIO_set_flags(b64, BIO_FLAGS_BASE64_NO_NL);
    }
    bmem = BIO_new_mem_buf(input, length);
    bmem = BIO_push(b64, bmem);
    BIO_read(bmem, buffer, length);

    BIO_free_all(bmem);

    return buffer;
}

int main()
{
    //std::string s = "hPzVpmmxBDBbuA0YLNKkI0muo14D3ZVDtsphW+ZEcmIHsn8R2p/Poq/4Cjbfwt8fr2oEymOoit2iAMSbqLwS8Ap/r4noombAS+mKEjRqGe2oX9HZXqWeT26Rg+8EWC1QlUNt7XvCr2ki112dbGACWRjR8yg7dFZ/XM+m4pRh0ju6mN+P1Pazu585oa+Umh/WJFDHP3ob6/m1pSTYCi4t2Q7Bdj3CKQfXEkD7Fm3vPbdXI0L67qszpVSb8pRG3wA9RT5+54Bajl3bDmGc8AjvE/I4NZWXWIeUaCHz3akZtqo20lhGgLek4Gj4FieMHj0xLaX+UM56NgcICgQ0hxzhWA==";
    std::string s = "AliWfgq8tkYTxsTn3HAGo0DLYCM43mqjOQt2ZjrtwVopzevbb3BW4FWX3/lIuSJWyJZI2hg4IKE1qYLnKFJivbERHszein7u3Ai4cL0oct3eGYBfetFptRA3ZkP9uZGLVTxW8/sKvbpIyCNyoXOkcrW/iYHlbJPY6ZF4TLQ6OTacjGkmH80bSFNf55y6eacXo/sNblhW9gAMkvrSxGFzs16N1UDIwH9wTG+fvYGMFmi3qH/DtvxlREdHcAIioqAcYEOKar3wMSdFr8gdNWmEwywXKJsWKU5er+vWAEN968XsRChwGc3QjGcJ4LvFxtdUx6XacSQR/ZjfOjCCrrGeEA==";
    std::cout << s.size() << "\n";
    std::vector<unsigned char> temp{};
    for (auto& c : s) {
        temp.push_back((unsigned char)c);
    }

    std::vector<char> buff{};
    buff.resize(temp.size());
    int len =  openssl::base64Decode(temp.data(), (int)temp.size(), buff.data());
    std::cout << len << "\n";
    return 0;
}