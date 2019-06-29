#pragma once

#include <cstdio>
#include <chrono>
#include <string_view>
#include <filesystem>

#include "CURLClient.h"

/*
download.json
{
	"download":
	[
		{
			"name":"1.0.json",
			"download_dir":"",
			"md5":"EC5A6A0D8B9CEA1BD1F71F1343293A3E",
			"url":"127.0.0.1:19001/test"
		},
		{
			"name":"1.0.zip",
			"download_dir":"",
			"md5":"301D9AD826316748BF89F497FA4EA84B",
			"url":"127.0.0.1:19001/test"
		},
		{
			"name":"win7.iso",
			"download_dir":"",
			"md5":"21FD234A68955C815D791047B7A34655",
			"url":"127.0.0.1:19001/test"
		}
	]
}
*/

class DownloadPkg
{
public:
    DownloadPkg() = default;
    ~DownloadPkg() = default;
    DownloadPkg(const DownloadPkg&) = delete;
    DownloadPkg& operator=(const DownloadPkg&) = delete;
    DownloadPkg(DownloadPkg&&) = delete;
    DownloadPkg& operator=(DownloadPkg&&) = delete;

    void SetName(std::string s);
    void SetMd5(std::string s);
    void SetUrl(std::string s);
    void SetDownloadDir(std::string s);

    const std::string& GetName() const;
    const std::string& GetMD5() const;
    const std::string& GetUrl() const;
    const std::string& GetDownloadDir() const;
    const std::vector<std::string>& GetDownloadDirList() const;
    std::filesystem::path GetDownloadFullPath(std::filesystem::path default_path) const;

private:
    std::string m_name;
    std::string m_md5;
    std::string m_url;
    std::string m_download_dir;
    std::vector<std::string> m_dir_list;
};

struct Manifest
{
    std::vector<std::shared_ptr<DownloadPkg>> m_download;
};

class HttpDownload
{
public:
    HttpDownload();
    ~HttpDownload();
    HttpDownload(const HttpDownload&) = delete;
    HttpDownload& operator=(const HttpDownload&) = delete;
    HttpDownload(HttpDownload&&) = delete;
    HttpDownload& operator=(HttpDownload&&) = delete;

    bool Init(std::string root_path);
    void Launch(std::string url);

private:
    bool ParseManifest(const std::string& s, Manifest* out);
    bool DownloadManifest(const Manifest& manifest);
    bool DownLoadFile(const DownloadPkg& pkg);
    void WriteFile(std::filesystem::path fpath, std::int64_t* total_size, std::uint64_t index, std::vector<std::uint8_t>* buffer, std::size_t length);
    void CloseFile();

    static bool CheckDownloadDir(std::filesystem::path default_path, const DownloadPkg& pkg);

private:
    std::string                         m_default_path_str;
    std::filesystem::path               m_default_path;
    std::unique_ptr<curl::CurlClient>   m_client;
    std::FILE*                          m_file;
};

