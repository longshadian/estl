#ifndef _CURL_GUARD_H_
#define _CURL_GUARD_H_

#include <string>
#include <vector>

#include <curl/curl.h>

namespace curl {

void globalInit(long flags = CURL_GLOBAL_DEFAULT);
void globalCleanup();

std::string getErrorMsg(CURLcode e);

class CURL_Guard
{
public:
    CURL_Guard();
    ~CURL_Guard();

    std::string urlEncode(const std::string& s);
    std::string urlDecode(const std::string& s);

    bool urlEncode(const std::string& s, std::string* out);
    bool urlDecode(const std::string& s, std::string* out);

    CURL* getCURL();
private:
    CURL* m_curl;
};

class HttpClinet
{
public:
    static const int32_t CONNECT_TIMEOUT = 3;
    static const int32_t TIMEOUT = 3;

    struct Optional
    {
        Optional();

        int32_t m_connecttimeout;   //连接超时
        int32_t m_timeout;          //超时
    };

    typedef size_t(*WriteFunctionPtr)(void* ptr, size_t size, size_t nmemb, void* args);
public:
    HttpClinet(Optional opt = Optional());
    ~HttpClinet();

    void post(const std::string& url, const std::string& content = {});
    bool responseOK() const;

    static size_t onWriteString(void* ptr, size_t size, size_t nmemb, void* args);
    static size_t onWriteBinary(void* ptr, size_t size, size_t nmemb, void* args);
    
public:
    CURL*               m_curl;
    Optional            m_opt;
    CURLcode            m_error;
    std::string         m_resp_string;
    std::vector<uint8_t> m_resp_binary;
    WriteFunctionPtr    m_write_cb;
    void*               m_write_cb_args;
};


}

#endif
