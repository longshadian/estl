#include "CURLClient.h"

namespace curl 
{

void GlobalInit(long flags)
{
    ::curl_global_init(flags);
}

void GlobalCleanup()
{
    ::curl_global_cleanup();
}

std::string GetErrorMsg(CURLcode c)
{
    const char* s = ::curl_easy_strerror(c);
    if (s)
        return s;
    return "";
}

CURLGuard::CURLGuard()
    :m_curl(::curl_easy_init())
{
}

CURLGuard::~CURLGuard()
{
    if (m_curl)
        ::curl_easy_cleanup(m_curl);
}

std::string CURLGuard::UrlEncode(const std::string& s)
{
    std::string out{};
    UrlEncode(s, &out);
    return out;
}

std::string CURLGuard::UrlDecode(const std::string& s)
{
    std::string out{};
    UrlDecode(s, &out);
    return out;
}

bool CURLGuard::UrlEncode(const std::string& s, std::string* out)
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

bool CURLGuard::UrlDecode(const std::string& s, std::string* out)
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

CurlClient::CurlClient()
    : m_curl(nullptr)
    , m_opt()
    , m_error(CURLE_OK)
    , m_response_content()
    , m_response_cb()
    , m_write_cb(nullptr)
    , m_write_cb_args(nullptr)
    , m_http_response_code(-1)
    , m_head_list(nullptr)
    , m_write_index(0)
{
    m_curl = ::curl_easy_init();
    BindOnWriteCB();
}

CurlClient::CurlClient(Optional opt)
    : m_curl(nullptr)
    , m_opt(opt)
    , m_error(CURLE_OK)
    , m_response_content()
    , m_response_cb()
    , m_write_cb(nullptr)
    , m_write_cb_args(nullptr)
    , m_http_response_code(-1)
    , m_head_list(nullptr)
    , m_write_index()
{
    m_curl = ::curl_easy_init();
    BindOnWriteCB();
}

CurlClient::~CurlClient()
{
    if (m_head_list) {
        ::curl_slist_free_all(m_head_list);
        m_head_list = nullptr;
    }
    ::curl_easy_cleanup(m_curl);
}

void CurlClient::Reset()
{
    m_error = CURLE_OK;
    m_response_content.clear();
    m_response_cb = {};
    m_http_response_code = -1;
    m_write_index = 0;
    if (m_head_list) {
        ::curl_slist_free_all(m_head_list);
        m_head_list = nullptr;
    }
    ::curl_easy_reset(m_curl);
    BindOnWriteCB();
}

void CurlClient::BindResponseCallback(ResponseFunc func)
{
    m_response_cb = func;
}

void CurlClient::AppendHead(const std::string& str)
{
    m_head_list = ::curl_slist_append(m_head_list, str.c_str());
}

void CurlClient::Get(const std::string& url)
{
    CurlSetOpt();
    ::curl_easy_setopt(m_curl, CURLOPT_URL, url.c_str());
    m_error = ::curl_easy_perform(m_curl);
}

void CurlClient::Post(const std::string& url, const std::string& content)
{
    CurlSetOpt();
    ::curl_easy_setopt(m_curl, CURLOPT_URL, url.c_str());
    ::curl_easy_setopt(m_curl, CURLOPT_POST, 1);
    if (!content.empty()) {
        ::curl_easy_setopt(m_curl, CURLOPT_POSTFIELDS, content.data());
        ::curl_easy_setopt(m_curl, CURLOPT_POSTFIELDSIZE, content.size());
    }
    m_error = ::curl_easy_perform(m_curl);
}

bool CurlClient::ResponseOK() const
{
    return m_error == CURLE_OK;
}

std::int32_t CurlClient::ResponseCode() const
{
    if (m_http_response_code == -1) {
        ::curl_easy_getinfo(m_curl, CURLINFO_RESPONSE_CODE, &m_http_response_code);
    }
    return m_http_response_code;
}

std::string CurlClient::ResponseContentString() const
{
    return std::string{m_response_content.begin(), m_response_content.end()};
}

std::string_view CurlClient::ResponseContentStringView() const
{
    auto p = reinterpret_cast<const char*>(m_response_content.data());
    return std::string_view{p, m_response_content.size()};
}

void CurlClient::BindOnWriteCB()
{
    m_write_cb = &CurlClient::OnWriteBinary;
    m_write_cb_args = this;
}

std::size_t CurlClient::OnWriteBinary(void* ptr, std::size_t size, std::size_t nmemb, void* args)
{
    CurlClient* pthis = reinterpret_cast<CurlClient*>(args);
    std::size_t total = size * nmemb;
    const std::uint8_t* p = reinterpret_cast<const std::uint8_t*>(ptr);
    pthis->m_response_content.insert(pthis->m_response_content.end(), p, p + total);
    if (pthis->m_response_cb)
        pthis->m_response_cb(pthis->m_write_index, &pthis->m_response_content, total);
    pthis->m_write_index++;
    return total;
}

void CurlClient::CurlSetOpt()
{
    if (m_head_list) {
        ::curl_easy_setopt(m_curl, CURLOPT_HTTPHEADER, m_head_list);
    }

    curl_easy_setopt(m_curl, CURLOPT_NOSIGNAL, 1); //关闭中断信号响应
    curl_easy_setopt(m_curl, CURLOPT_FOLLOWLOCATION, 1);//允许重定向

    if (m_write_cb)
        ::curl_easy_setopt(m_curl, CURLOPT_WRITEFUNCTION, m_write_cb);
    if (m_write_cb_args)
        ::curl_easy_setopt(m_curl, CURLOPT_WRITEDATA, m_write_cb_args);

    if (m_opt.m_connecttimeout > 0)
        ::curl_easy_setopt(m_curl, CURLOPT_CONNECTTIMEOUT, m_opt.m_connecttimeout);
    if (m_opt.m_timeout > 0)
        ::curl_easy_setopt(m_curl, CURLOPT_TIMEOUT, m_opt.m_timeout);
}


} // namespace curl

