#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <functional>
#include <string_view>
#include <curl/curl.h>

namespace curl
{

void GlobalInit(long flags = CURL_GLOBAL_DEFAULT);
void GlobalCleanup();

std::string GetErrorMsg(CURLcode e);

class CURLGuard
{
public:
    CURLGuard();
    ~CURLGuard();

    std::string UrlEncode(const std::string& s);
    std::string UrlDecode(const std::string& s);

    bool UrlEncode(const std::string& s, std::string* out);
    bool UrlDecode(const std::string& s, std::string* out);

    CURL* m_curl;
};

class CurlClient
{
public:
    static const std::int32_t CONNECT_TIMEOUT = 10;
    static const std::int32_t TIMEOUT = 0;

    struct Optional
    {
        std::int32_t m_connecttimeout{ CONNECT_TIMEOUT };   //连接超时
        std::int32_t m_timeout{ TIMEOUT };          //超时
    };

    typedef std::size_t(*WriteFunctionPtr)(void* ptr, std::size_t size, std::size_t nmemb, void* args);

    using ResponseFunc = std::function<void(std::uint64_t, std::vector<std::uint8_t>*, std::size_t)>;

public:
    CurlClient();
    explicit CurlClient(Optional opt);
    ~CurlClient();
    CurlClient(const CurlClient&) = delete;
    CurlClient& operator=(const CurlClient&) = delete;
    CurlClient(CurlClient&&) = delete;
    CurlClient& operator=(CurlClient&&) = delete;

    void                        Reset();
    void                        BindResponseCallback(ResponseFunc func);
    void                        AppendHead(const std::string& str);

    void                        Get(const std::string& url);
    void                        Post(const std::string& url, const std::string& content = {});
    bool                        ResponseOK() const;
    std::int32_t                ResponseCode() const;
    std::string                 ResponseContentString() const;
    std::string_view            ResponseContentStringView() const;

private:
    void                        BindOnWriteCB();
    static std::size_t          OnWriteBinary(void* ptr, std::size_t size, std::size_t nmemb, void* args);
    void                        CurlSetOpt();

public:
    CURL*                       m_curl;
    Optional                    m_opt;
    CURLcode                    m_error;
    std::vector<std::uint8_t>   m_response_content;
    ResponseFunc                m_response_cb;
    WriteFunctionPtr            m_write_cb;
    void*                       m_write_cb_args;
    std::int32_t                m_http_response_code;
    struct curl_slist*          m_head_list;

private:
    std::uint64_t               m_write_index;
};

} // namespace curl
