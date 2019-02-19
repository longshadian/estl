#include "HttpClient.h"

HttpClinet::Optional::Optional() 
    : m_connecttimeout(HttpClinet::CONNECT_TIMEOUT)
    , m_timeout(HttpClinet::TIMEOUT)
{
}

HttpClinet::HttpClinet(Optional opt)
    : m_curl(nullptr)
    , m_opt(opt)
    , m_error(CURLE_OK)
    , m_response()
    , m_write_cb(nullptr)
    , m_write_cb_args(nullptr)
{
    m_curl = curl_easy_init();
}

HttpClinet::~HttpClinet()
{
    curl_easy_cleanup(m_curl);
}

void HttpClinet::post(const std::string& url, const std::string& content)
{
    curl_easy_setopt(m_curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(m_curl, CURLOPT_POST, 1);

    if (m_write_cb)
        curl_easy_setopt(m_curl, CURLOPT_WRITEFUNCTION, m_write_cb);
    if (m_write_cb_args)
        curl_easy_setopt(m_curl, CURLOPT_WRITEDATA, m_write_cb_args);

    if (!content.empty()) {
        curl_easy_setopt(m_curl, CURLOPT_POSTFIELDS, content.data());
        curl_easy_setopt(m_curl, CURLOPT_POSTFIELDSIZE, content.size());
    }

    curl_easy_setopt(m_curl, CURLOPT_CONNECTTIMEOUT, m_opt.m_connecttimeout);
    curl_easy_setopt(m_curl, CURLOPT_TIMEOUT, m_opt.m_timeout);
    m_error = curl_easy_perform(m_curl);
}

size_t HttpClinet::onWrite(void* ptr, size_t size, size_t nmemb, void* args)
{
    HttpClinet* p_this = (HttpClinet*)args;
    size_t total = size * nmemb;
    const char* p = (const char*)ptr;
    p_this->m_response.append(p, p + total);
    return nmemb;
}

bool HttpClinet::responseOK() const
{
    return m_error == CURLE_OK;
}