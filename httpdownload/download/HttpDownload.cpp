#include "download/HttpDownload.h"

#include <nlohmann/json.hpp>
#include <filesystem>
#include <cctype>
#include <sstream>
#include "download/Openssl.h"
#include "download/Utilities.h"

#define PrintLog(fmt, ...) printf(fmt "\n", ##__VA_ARGS__)

using json = nlohmann::json;

static void SplitString(const std::string& s, char c, std::vector<std::string>* out)
{
    if (s.empty())
        return;
    std::istringstream istm(s);
    std::string temp;
    while (std::getline(istm, temp, c)) {
        temp.clear();
        if (!temp.empty())
            out->emplace_back(std::move(temp));
    }
}


DownloadPkg::DownloadPkg()
    : m_fullName()
    , m_name()
    , m_md5()
    , m_url()
    , m_download_dir()
    , m_dir_list()
{
}

void DownloadPkg::SetName(std::string s)
{
    m_name = std::move(s);
}

void DownloadPkg::SetMd5(std::string s)
{
    m_md5 = std::move(s);
}

void DownloadPkg::SetUrl(std::string s)
{
    m_url = std::move(s);
    while (!m_url.empty()) {
        if (m_url.back() == '/') {
            m_url.pop_back();
        } else {
            break;
        }
    }
}

void DownloadPkg::SetDownloadDir(std::string s)
{
    m_download_dir = std::move(s);
    ::SplitString(m_download_dir, '/', &m_dir_list);
    m_dir_list.erase(std::remove(m_dir_list.begin(), m_dir_list.end(), "."), m_dir_list.end());
    m_dir_list.erase(std::remove(m_dir_list.begin(), m_dir_list.end(), ".."), m_dir_list.end());
}

void DownloadPkg::Init()
{
    m_fullName = Url() + '/' + Name();
}

const std::string& DownloadPkg::Name() const
{
    return m_name;
}

const std::string& DownloadPkg::MD5() const
{
    return m_md5;
}

const std::string& DownloadPkg::Url() const
{
    return m_url;
}

const std::string& DownloadPkg::DownloadDir() const
{
    return m_download_dir;
}

const std::vector<std::string>& DownloadPkg::DownloadDirList() const
{
    return m_dir_list;
}

boost::filesystem::path DownloadPkg::DownloadFullPath(boost::filesystem::path default_path) const
{
    try {
        boost::filesystem::path dest_path = default_path;
        for (const std::string& dir : m_dir_list) {
            dest_path /= dir;
        }
        dest_path /= m_name;
        return dest_path;
    } catch (const std::exception& e) {
        (void)e;
        return default_path;
    }
}

const std::string& DownloadPkg::FullName() const
{
    return m_fullName;
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

HttpDownload::HttpDownload()
    : m_default_path_str()
    , m_default_path(m_default_path_str)
    , m_client(std::make_unique<curl::CurlClient>())
    , m_file()
{
}

HttpDownload::~HttpDownload()
{
    CloseFile();
}

bool HttpDownload::Init(std::string root_path) 
{
    m_default_path_str = std::move(root_path);
    try {
        m_default_path = boost::filesystem::path(m_default_path_str);
        boost::filesystem::create_directory(m_default_path);
        return true;
    } catch (const std::exception& e) {
        PrintLog("ERROR: init failed. reason:[%s]", e.what());
        return false;
    }
}

bool HttpDownload::Init(DownloadConf conf)
{
    m_conf = conf;

    // 检查下载目录是否存在
    try {
        m_default_path = boost::filesystem::path(m_default_path_str);
        boost::filesystem::create_directory(m_default_path);
    } catch (const std::exception& e) {
        PrintLog("ERROR: init failed. 创建下载目录失败[%s] reason:[%s]", m_conf.download_dir.c_str(), e.what());
        return false;
    }

    // 通过url获取固定文件名
    {
        auto it = std::find(m_conf.url.rbegin(), m_conf.url.rend(), '/');
        if (it == m_conf.url.rend()) {
            PrintLog("ERROR: init failed. 无法通过url获取需要下载的固定文件名url:[%s]", m_conf.url.c_str());
            return false;
        }
        std::string temp{ it + 1, m_conf.url.rend() };  // 去除'/'
        if (temp.empty()) {
            PrintLog("ERROR: init failed. 通过url获取需要下载的文件名为空url:[%s]", m_conf.url.c_str());
            return false;
        }
        m_fixed_file_name = temp;
        PrintLog("INFO: url:[%s] 文件名:[%s]", m_conf.url.c_str(), m_fixed_file_name.c_str());
    }

    PrintLog("INFO: 初始化成功 url:[%s] 文件名:[%s] 下载保存路径:[%s]", m_conf.url.c_str(), m_fixed_file_name.c_str(), m_conf.download_dir.c_str());
    return true;
}

void HttpDownload::Launch()
{
    std::string old_file_md5 = CheckFixedFile();

    m_client->Reset();
    m_client->Get(m_conf.url);
    if (!m_client->ResponseOK()) {
        PrintLog("WARNING: HTTP GET [%s] failed response != OK", m_conf.url.c_str());
        return;
    }
    if (m_client->ResponseCode() != 200) {
        PrintLog("WARNING: HTTP GET [%s] failed response code != 200 code:[%d]", m_conf.url.c_str(), m_client->ResponseCode());
        return;
    }

    std::string download_file_content = m_client->ResponseContentString();
    if (download_file_content.empty()) {
        PrintLog("WARNING: HTTP GET [%s] failed donwload file content length == 0", m_conf.url.c_str());
        return;
    }
    std::string download_file_md5 = openssl::MD5(download_file_content);
    if (download_file_md5.empty()) {
        PrintLog("WARNING: HTTP GET [%s] failed donwload file content length != 0 but md5 == 0 content: [%s]", m_conf.url.c_str(), download_file_content.c_str());
        return;
    }

    auto manifest = std::make_shared<Manifest>();
    if (!ParseFixedFile(download_file_content, manifest.get())) {
        PrintLog("WARNING: ParseFixedFile [%s] failed. content: [%s]", m_conf.url.c_str(), download_file_content.c_str());
        return;
    }
    DownloadManifest(*manifest);
}

void HttpDownload::Launch(std::string url)
{
    m_client->Get(url);
    if (m_client->ResponseOK() && m_client->ResponseCode() == 200) {
        auto s = m_client->ResponseContentString();
        auto manifest = std::make_shared<Manifest>();
        if (!ParseFixedFile(s, manifest.get())) {
            return;
        }
        DownloadManifest(*manifest);
    } else {
        PrintLog("ERROR get url: %s failed.", url.c_str());
        return;
    }
}

std::string HttpDownload::CheckFixedFile()
{
    std::string s;
    m_default_path /= m_fixed_file_name;
    s = m_default_path.generic_string();
    return openssl::MD5(s);
}

bool HttpDownload::ParseFixedFile(const std::string& s, Manifest* out)
{
    try {
        json j = json::parse(s);
        if (!j["download"].is_array()) {
            return false;
        }
        const auto& download = j["download"];
        auto cnt = download.size();
        for (std::size_t i = 0; i != cnt; ++i) {
            const auto& cell = download[i];
            auto pkg = std::make_shared<DownloadPkg>();
            pkg->SetName(cell["name"].get<std::string>());
            pkg->SetDownloadDir(cell["download_dir"].get<std::string>());
            pkg->SetMd5(cell["md5"].get<std::string>());
            pkg->SetUrl(cell["url"].get<std::string>());
            pkg->Init();
            out->m_download.push_back(pkg);
        }
        return true;
    } catch (const std::exception& e) {
        PrintLog("ERROR ParseFixedFile faile. %s  reason:[%s]", s.c_str(), e.what());
        return false;
    }
}

bool HttpDownload::DownloadManifest(const Manifest& manifest)
{
    bool succ = true;
    for (const auto& it : manifest.m_download) {
        if (!DownloadFile(*it)) {
            succ = false;
        }
    }
    return succ;
}

bool HttpDownload::DownloadFile(const DownloadPkg& pkg)
{
    if (pkg.MD5().empty()) {
        PrintLog("WARNING: [%s] md5 length == 0 discard. filename: [%s]", pkg.FullName().c_str(), pkg.Name().c_str());
        return false;
    }

    bool same_file = CheckLocalfile(pkg);
    if (same_file) {
        PrintLog("INFO: [%s] 不用下载", pkg.FullName().c_str());
        return true;
    }

    if (!CheckDownloadDir(m_default_path, pkg)) {
        PrintLog("WARNING: CheckDownloadDir download path:[%s] failed", pkg.DownloadDir().c_str());
        return false;
    }

    PrintLog("INFO: [%s] start download", pkg.FullName().c_str());
    boost::filesystem::path dest_file_path = pkg.DownloadFullPath(m_default_path);
    std::int64_t total_size{};
    m_client->Reset();
    m_client->BindResponseCallback(std::bind(&HttpDownload::WriteFile, this, dest_file_path, &total_size, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    m_client->Get(pkg.Url() + "/" + pkg.Name());
    if (!m_client->ResponseOK() || m_client->ResponseCode() != 200) {
        PrintLog("ERROR: [%s] download failed.", pkg.FullName().c_str());
        return false;
    }
    CloseFile();

    // 校验md5
    auto dest_file_path_str = dest_file_path.generic_string();
    auto download_md5 = openssl::MD5(dest_file_path_str);
    if (download_md5.size() != pkg.MD5().size()) {
        PrintLog("ERROR: [%s] md5 different [%s] [%s]", pkg.Url().c_str(), pkg.Name().c_str(), pkg.MD5().c_str(), download_md5.c_str());
        return false;
    }
    for (std::size_t i = 0; i != download_md5.size(); ++i) {
        if (std::toupper(download_md5[i]) != std::toupper(pkg.MD5()[i])) {
            PrintLog("ERROR: url:[%s][%s] md5 different [%s] [%s]", pkg.Url().c_str(), pkg.Name().c_str(), pkg.MD5().c_str(), download_md5.c_str());
            return false;
        }
    }
    PrintLog("INFO: url:[%s][%s] download success size:[%lld] md5:[%s]", pkg.Url().c_str(), pkg.Name().c_str(), total_size, download_md5.c_str());
    return true;
}

void HttpDownload::WriteFile(boost::filesystem::path fpath, std::int64_t* total_size, std::uint64_t index, std::vector<std::uint8_t>* buffer, std::size_t length)
{
    // 第一次，打开文件
    if (index == 0 && !m_file) {
        auto s = fpath.generic_string();
        m_file = std::fopen(s.c_str(), "wb+");
    }
    if (index >= 0 && !m_file) {
        // 打开文件失败，不执行写入操作，清理buffer
        buffer->clear();
        return;
    }

    *total_size += length;
    std::fwrite(buffer->data(), 1, buffer->size(), m_file);
    buffer->clear();
    std::fflush(m_file);
}

void HttpDownload::CloseFile()
{
    if (m_file) {
        std::fclose(m_file);
        m_file = nullptr;
    }
}

bool HttpDownload::CheckDownloadDir(boost::filesystem::path default_path, const DownloadPkg& pkg)
{
    if (pkg.DownloadDirList().empty())
        return true;
    try {
        boost::filesystem::path dest_path = default_path;
        for (const std::string& dir : pkg.DownloadDirList()) {
            dest_path /= dir;
        }
        boost::filesystem::create_directories(dest_path);
        return true;
    } catch (const std::exception& e) {
        PrintLog("chekc download dir faile. name:[%s] dir:[%s] reason:[%s]", pkg.Name().c_str(), pkg.DownloadDir().c_str(), e.what());
        return false;
    }
}

bool HttpDownload::CheckLocalfile(const DownloadPkg& pkg) const
{
    try {
        boost::filesystem::path localfile_path;
        localfile_path = m_default_path;
        localfile_path /= pkg.Name();
        if (!boost::filesystem::is_regular_file(localfile_path)) {
            std::string temp = localfile_path.generic_string();
            PrintLog("INFO: [%s] localfile:[%s] error need download", pkg.FullName().c_str(), temp.c_str());
            return false;
        }

        std::string localfile_path_str = localfile_path.generic_string();
        std::string localfile_md5 = openssl::MD5(localfile_path_str);
        localfile_md5 = Utilities::ToUpperCase(localfile_md5);
        std::string new_md5 = Utilities::ToUpperCase(pkg.MD5());
        if (localfile_md5 != new_md5) {
            PrintLog("INFO: [%s] md5 different [%s] != [%s] need download", pkg.FullName().c_str(), localfile_md5.c_str(), new_md5.c_str());
            return false;
        }
        PrintLog("INFO: [%s] md5 same [%s] == [%s]", pkg.FullName().c_str(), localfile_md5.c_str(), new_md5.c_str());
        return true;
    } catch (const std::exception& e) {
        PrintLog("WARNING: CheckLocalFile exception: [%s] reason:[%s]", pkg.FullName().c_str(), e.what());
        return false;
    }
}

