#pragma once

#include <string_view>
#include <mutex>
#include <nlohmann/json.hpp>

#include "HttpClient.h"

class PassingService
{
public:
    PassingService() = default;
    ~PassingService() = default;
    PassingService(const PassingService&) = delete;
    PassingService& operator=(const PassingService&) = delete;
    PassingService(PassingService&&) = delete;
    PassingService& operator=(PassingService&&) = delete;

    std::string m_node;
    std::string m_check_id;
    std::string m_name;
    std::string m_status;
    std::string m_notes;
    std::string m_output;
    std::string m_service_id;
    std::string m_service_name;
    std::vector<std::string> m_service_tags;
};
using PassingServicePtr = std::shared_ptr<PassingService>;

class Consul
{
public:
    Consul();
    ~Consul();
    Consul(const Consul&) = delete;
    Consul& operator=(const Consul&) = delete;
    Consul(Consul&&) = delete;
    Consul& operator=(Consul&&) = delete;

    bool Initialize(std::string host, unsigned short port, std::string version = "/v1");

    bool CatalogServices();
    bool HealthService(std::string_view service_name);
    bool HealthNode(std::string node_name);
    bool HealthStatePassing();

    std::unordered_map<std::string, PassingServicePtr> ReleasePassingServices();

private:
    //static std::unordered_map<std::string, ServicePtr> ParseCatalogServices(std::string_view content);
    static PassingServicePtr ParsePassingService(const nlohmann::json& obj);

private:
    std::string                     m_host;
    unsigned short                  m_port;
    std::string                     m_version;
    std::unique_ptr<HttpClient>     m_http_client;

    std::mutex                      m_mutex;
    std::unordered_map<std::string, PassingServicePtr> m_passing_service_map; // key: service_id;
};
