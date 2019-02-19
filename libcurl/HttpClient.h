
#include <curl/curl.h>

#include <iostream>
#include <string>
#include <cassert>

class HttpClinet
{
public:
    static const int32_t CONNECT_TIMEOUT = 3;
    static const int32_t TIMEOUT         = 1;

    struct Optional
    {
        Optional();

        int32_t m_connecttimeout;   //连接超时
        int32_t m_timeout;          //超时
    };

    typedef size_t (*WriteFunctionPtr)(void* ptr, size_t size, size_t nmemb, void* args);
public:
    HttpClinet(Optional opt = Optional());
    ~HttpClinet();

    void post(const std::string& url, const std::string& content = {});
    bool responseOK() const;

    static size_t onWrite(void* ptr, size_t size, size_t nmemb, void* args);
public:
    CURL*       m_curl;
    Optional    m_opt;
    CURLcode    m_error;
    std::string m_response;
    WriteFunctionPtr m_write_cb;
    void*       m_write_cb_args;
};
