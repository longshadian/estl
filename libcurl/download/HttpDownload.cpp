#include "HttpDownload.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <cctype>
#include <sstream>
#include "Openssl.h"

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

const std::string& DownloadPkg::GetName() const
{
    return m_name;
}

const std::string& DownloadPkg::GetMD5() const
{
    return m_md5;
}

const std::string& DownloadPkg::GetUrl() const
{
    return m_url;
}

const std::string& DownloadPkg::GetDownloadDir() const
{
    return m_download_dir;
}

const std::vector<std::string>& DownloadPkg::GetDownloadDirList() const
{
    return m_dir_list;
}

std::filesystem::path DownloadPkg::GetDownloadFullPath(std::filesystem::path default_path) const
{
    try {
        std::filesystem::path dest_path = default_path;
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
        m_default_path = std::filesystem::path(m_default_path_str);
        std::filesystem::create_directory(m_default_path);
        return true;
    } catch (const std::exception& e) {
        PrintLog("ERROR: init failed. reason:[%s]", e.what());
        return false;
    }
}

void HttpDownload::Launch(std::string url)
{
    m_client->Get(url);
    if (m_client->ResponseOK() && m_client->ResponseCode() == 200) {
        auto s = m_client->ResponseContentString();
        auto manifest = std::make_shared<Manifest>();
        if (!ParseManifest(s, manifest.get())) {
            return;
        }
        DownloadManifest(*manifest);
    } else {
        PrintLog("ERROR get url: %s failed.", url.c_str());
        return;
    }
}

bool HttpDownload::ParseManifest(const std::string& s, Manifest* out)
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
            out->m_download.push_back(pkg);
        }
        return true;
    } catch (const std::exception& e) {
        PrintLog("ERROR parse manifest faile. %s %s", s.c_str(), e.what());
        return false;
    }
}

bool HttpDownload::DownloadManifest(const Manifest& manifest)
{
    for (const auto& it : manifest.m_download) {
        DownLoadFile(*it);
    }
    return true;
}

bool HttpDownload::DownLoadFile(const DownloadPkg& pkg)
{
    if (!CheckDownloadDir(m_default_path, pkg)) {
        PrintLog("CheckDownloadDir download path:[%s] failed", pkg.GetDownloadDir().c_str());
        return false;
    }

    std::filesystem::path dest_file_path = pkg.GetDownloadFullPath(m_default_path);
    std::int64_t total_size{};
    m_client->Reset();
    m_client->BindResponseCallback(std::bind(&HttpDownload::WriteFile, this, dest_file_path, &total_size, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    m_client->Get(pkg.GetUrl() + "/" + pkg.GetName());
    if (!m_client->ResponseOK() || m_client->ResponseCode() != 200) {
        PrintLog("ERROR download url:[%s] file:[%s] failed.", pkg.GetUrl().c_str(), pkg.GetName().c_str());
        return false;
    }
    CloseFile();

    // 校验md5
    auto dest_file_path_str = dest_file_path.generic_u8string();
    auto download_md5 = openssl::MD5(dest_file_path_str);
    if (download_md5.size() != pkg.GetMD5().size()) {
        PrintLog("ERROR md5 different [%s] [%s]", pkg.GetMD5().c_str(), download_md5.c_str());
        return false;
    }
    for (std::size_t i = 0; i != download_md5.size(); ++i) {
        if (std::toupper(download_md5[i]) != std::toupper(pkg.GetMD5()[i])) {
            PrintLog("ERROR md5 different [%s] [%s]", pkg.GetMD5().c_str(), download_md5.c_str());
            return false;
        }
    }
    PrintLog("INFO: download success file:[%s] size:[%lld] md5:[%s]", pkg.GetName().c_str(), total_size, download_md5.c_str());
    return true;
}

void HttpDownload::WriteFile(std::filesystem::path fpath, std::int64_t* total_size, std::uint64_t index, std::vector<std::uint8_t>* buffer, std::size_t length)
{
    // 第一次，打开文件
    if (index == 0 && !m_file) {
        auto s = fpath.generic_u8string();
        m_file = std::fopen(s.c_str(), "wb+");
    }
    if (index > 0 && !m_file) {
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

bool HttpDownload::CheckDownloadDir(std::filesystem::path default_path, const DownloadPkg& pkg)
{
    if (pkg.GetDownloadDirList().empty())
        return true;
    try {
        std::filesystem::path dest_path = default_path;
        for (const std::string& dir : pkg.GetDownloadDirList()) {
            dest_path /= dir;
        }
        std::filesystem::create_directories(dest_path);
        return true;
    } catch (const std::exception& e) {
        PrintLog("chekc download dir faile. name:[%s] dir:[%s]", pkg.GetName().c_str(), pkg.GetDownloadDir().c_str());
        return false;
    }
}
