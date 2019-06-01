#include "HttpDownload.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <cctype>
#include "Openssl.h"

#define PrintLog(fmt, ...) printf(fmt "\n", ##__VA_ARGS__)

using json = nlohmann::json;

HttpDownload::HttpDownload()
    : m_root_path()
    , m_client(std::make_unique<curl::CurlClient>())
    , m_file()
{
}

HttpDownload::HttpDownload(std::string root_path)
    : m_root_path(std::move(root_path))
    , m_client(std::make_unique<curl::CurlClient>())
    , m_file()
{
}

HttpDownload::~HttpDownload()
{
    CloseFile();
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
            pkg->m_name = cell["name"].get<std::string>();
            pkg->m_download_dir = cell["download_dir"].get<std::string>();
            pkg->m_md5 = cell["md5"].get<std::string>();
            pkg->m_url = cell["url"].get<std::string>();
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
    std::filesystem::path dest_path{};
    std::filesystem::path dest_file_path{};
    try {
        std::filesystem::path temp_path{m_root_path};
        temp_path /= pkg.m_download_dir;
        dest_path = temp_path;
        dest_file_path = dest_path / pkg.m_name;
    } catch (const std::exception& e) {
        PrintLog("ERROR: create destination path failed. root_path:[%s]  download_dir:[%s] reason:[%s]", m_root_path.c_str(), pkg.m_download_dir.c_str(), e.what());
        return false; 
    }
    std::int64_t total_size{};
    m_client->Reset();
    m_client->BindResponseCallback(std::bind(&HttpDownload::WriteFile, this, dest_file_path, &total_size, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    m_client->Get(pkg.m_url);
    if (!m_client->ResponseOK() || m_client->ResponseCode() != 200) {
        PrintLog("ERROR download url:[%s] file:[%s] failed.", pkg.m_url.c_str(), pkg.m_name.c_str());
        return false;
    }
    CloseFile();

    // 校验md5
    auto dest_file_path_str = dest_file_path.generic_u8string();
    auto download_md5 = openssl::MD5(dest_file_path_str);
    if (download_md5.size() != pkg.m_md5.size()) {
        PrintLog("ERROR md5 different [%s] [%s]", pkg.m_md5.c_str(), download_md5.c_str());
        return false;
    }
    for (std::size_t i = 0; i != download_md5.size(); ++i) {
        if (std::toupper(download_md5[i]) != std::toupper(pkg.m_md5[i])) {
            PrintLog("ERROR md5 different [%s] [%s]", pkg.m_md5.c_str(), download_md5.c_str());
            return false;
        }
    }
    PrintLog("INFO: download success file:[%s] size:[%lld] md5:[%s]", pkg.m_name.c_str(), total_size, download_md5.c_str());
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
