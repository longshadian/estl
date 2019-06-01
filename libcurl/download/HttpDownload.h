#pragma once

#include <cstdio>
#include <chrono>
#include <string_view>
#include <filesystem>

#include "CURLClient.h"

/*
manifest.json

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

struct DownloadPkg
{
    std::string m_name{};
    std::string m_download_dir{};
    std::string m_md5{};
    std::string m_url{};
};

struct Manifest
{
    std::vector<std::shared_ptr<DownloadPkg>> m_download;
};

class HttpDownload
{
public:
    HttpDownload();
    explicit HttpDownload(std::string root_path);
    ~HttpDownload();
    HttpDownload(const HttpDownload&) = delete;
    HttpDownload& operator=(const HttpDownload&) = delete;
    HttpDownload(HttpDownload&&) = delete;
    HttpDownload& operator=(HttpDownload&&) = delete;

    void Launch(std::string url);

private:
    bool ParseManifest(const std::string& s, Manifest* out);
    bool DownloadManifest(const Manifest& manifest);
    bool DownLoadFile(const DownloadPkg& pkg);
    void WriteFile(std::filesystem::path fpath, std::int64_t* total_size, std::uint64_t index, std::vector<std::uint8_t>* buffer, std::size_t length);
    void CloseFile();

private:
    std::string                         m_root_path;
    std::unique_ptr<curl::CurlClient>   m_client;
    std::FILE*                          m_file;
};

