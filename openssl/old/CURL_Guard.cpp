#include "CURL_Guard.h"

namespace curl {

void globalInit(long flags)
{
    ::curl_global_init(flags);
}

void globalCleanup()
{
    ::curl_global_cleanup();
}

CURL_Guard::CURL_Guard()
    :m_curl(::curl_easy_init())
{
}

CURL_Guard::~CURL_Guard()
{
    if (m_curl)
        ::curl_easy_cleanup(m_curl);
}

std::string CURL_Guard::urlEncode(const std::string& s)
{
    std::string out{};
    urlEncode(s, &out);
    return out;
}

std::string CURL_Guard::urlDecode(const std::string& s)
{
    std::string out{};
    urlDecode(s, &out);
    return out;
}

bool CURL_Guard::urlEncode(const std::string& s, std::string* out)
{
    if (s.empty())
        return true;
    char* output = ::curl_easy_escape(m_curl, s.c_str(), (int)s.size());
    if (!output) {
        return false;
    }
    *out = output;
    ::curl_free(output);
    return true;
}

bool CURL_Guard::urlDecode(const std::string& s, std::string* out)
{
    if (s.empty())
        return true;
    char* output = ::curl_easy_unescape(m_curl, s.c_str(), (int)s.size(), nullptr);
    if (!output) {
        return false;
    }
    *out = output;
    ::curl_free(output);
    return true;
}

CURL* CURL_Guard::getCURL()
{
    return m_curl;
}

HttpClinet::Optional::Optional()
    : m_connecttimeout(HttpClinet::CONNECT_TIMEOUT)
    , m_timeout(HttpClinet::TIMEOUT)
{
}

HttpClinet::HttpClinet(Optional opt)
    : m_curl(nullptr)
    , m_opt(opt)
    , m_error(CURLE_OK)
    , m_resp_string()
    , m_resp_binary()
    , m_write_cb(nullptr)
    , m_write_cb_args(nullptr)
{
    //д╛хо
    m_write_cb = &HttpClinet::onWriteString;
    m_write_cb_args = &m_resp_string;

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

size_t HttpClinet::onWriteString(void* ptr, size_t size, size_t nmemb, void* args)
{
    size_t total = size * nmemb;
    const char* p = (const char*)ptr;
    std::string* buff = (std::string*)args;
    buff->append(p, p + total);
    return total;
}

size_t HttpClinet::onWriteBinary(void* ptr, size_t size, size_t nmemb, void* args)
{
    size_t total = size * nmemb;
    const uint8_t* p = (const uint8_t*)ptr;
    std::vector<uint8_t>* buff = (std::vector<uint8_t>*)args;
    buff->insert(buff->end(), p, p + total);
    return total;
}

bool HttpClinet::responseOK() const
{
    return m_error == CURLE_OK;
}

std::string getErrorMsg(CURLcode c)
{
    const char* s = ::curl_easy_strerror(c);
    if (s)
        return s;
    return "";
}

}
