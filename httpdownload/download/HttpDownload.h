#pragma once

#include <cstdio>
#include <chrono>
#include <boost/filesystem.hpp>

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
    DownloadPkg();
    ~DownloadPkg() = default;
    DownloadPkg(const DownloadPkg&) = delete;
    DownloadPkg& operator=(const DownloadPkg&) = delete;
    DownloadPkg(DownloadPkg&&) = delete;
    DownloadPkg& operator=(DownloadPkg&&) = delete;

    void SetName(std::string s);
    void SetMd5(std::string s);
    void SetUrl(std::string s);
    void SetDownloadDir(std::string s);
    void Init();

    const std::string& Name() const;
    const std::string& MD5() const;
    const std::string& Url() const;
    const std::string& DownloadDir() const;
    const std::vector<std::string>& DownloadDirList() const;
    boost::filesystem::path DownloadFullPath(boost::filesystem::path default_path) const;
    const std::string& FullName() const;

private:
    std::string m_fullName;
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

struct DownloadConf
{
    std::string download_dir;
    std::string url;
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
    bool Init(DownloadConf conf);
    void Launch();
    void Launch(std::string url);

private:
    std::string CheckFixedFile();
    bool ParseFixedFile(const std::string& s, Manifest* out);
    bool DownloadManifest(const Manifest& manifest);
    bool DownloadFile(const DownloadPkg& pkg);
    void WriteFile(boost::filesystem::path fpath, std::int64_t* total_size, std::uint64_t index, std::vector<std::uint8_t>* buffer, std::size_t length);
    void CloseFile();

    static bool CheckDownloadDir(boost::filesystem::path default_path, const DownloadPkg& pkg);
    bool CheckLocalfile(const DownloadPkg& pkg) const;

private:
    DownloadConf                        m_conf;
    std::string                         m_fixed_file_name;   // 下载文件名
    std::string                         m_default_path_str;
    boost::filesystem::path             m_default_path;
    std::unique_ptr<curl::CurlClient>   m_client;
    std::FILE*                          m_file;
};

