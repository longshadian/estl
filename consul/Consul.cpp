#include "Consul.h"

#include "Types.h"

#include <nlohmann/json.hpp>

using namespace nlohmann;

Consul::Consul()
    : m_host()
    , m_port()
    , m_version()
    , m_http_client(std::make_unique<HttpClient>())
    , m_passing_service_map()
{
}

Consul::~Consul()
{
}

bool Consul::Initialize(std::string host, unsigned short port, std::string version)
{
    m_host = std::move(host);
    m_port = port;
    m_version = std::move(version);
    return true;
}

bool Consul::CatalogServices()
{
    try {
        std::string path = m_version +"/catalog/services";
        http::request<http::empty_body> req{};
        auto rsp = m_http_client->Request(http::verb::get, m_host.c_str(), m_port, path, std::move(req));
        return true;
    } catch (const std::exception& e) {
        LOG(error) << "exception: " << e.what();
        return false;
    }
}

bool Consul::HealthService(std::string_view sv)
{
    return true;
}

bool Consul::HealthNode(std::string node_name)
{
    try {
        std::string path = m_version + "/health/node/" + node_name;
        http::request<http::empty_body> req{};
        auto rsp = m_http_client->Request(http::verb::get, m_host.c_str(), m_port, path, std::move(req));
        try {
            auto content = json::parse(rsp.body());
            for (int i = 0; i != content.size(); ++i) {
                LOG(info) << content[i].dump();
            }
        } catch (std::exception& e) {
            LOG(warning) << "json exception: " << e.what();
        }
    } catch (const std::exception& e) {
        LOG(error) << "exception: " << e.what();
    }
    return true;
}

bool Consul::HealthStatePassing()
{
    std::unordered_map<std::string, PassingServicePtr> current_passing_services{};
    try {
        std::string path = m_version + "/health/state/passing";
        http::request<http::empty_body> req{};
        auto rsp = m_http_client->Request(http::verb::get, m_host.c_str(), m_port, path, std::move(req));
        auto content = json::parse(rsp.body());
        for (int i = 0; i != content.size(); ++i) {
            auto service = ParsePassingService(content[i]);
            if (service == nullptr) {
                continue;
            }
            current_passing_services.emplace(service->m_service_id, service);
        }
    } catch (const std::exception& e) {
        LOG(warning) << "exception: " << e.what();
        return false;
    }

    {
        std::lock_guard<std::mutex> lk{m_mutex};
        std::swap(m_passing_service_map, current_passing_services);
    }
    return true;
}

/*
std::unordered_map<std::string, ServicePtr> Consul::ParseCatalogServices(std::string_view content)
{
    std::unordered_map<std::string, ServicePtr> service_map;
    auto json_root = json::parse(content);
    for (const auto& kv : json_root.items()) {
        auto p = std::make_shared<Service>();
        p->m_service_id = kv.key();
        p->m_service_name = kv.key();
        // TODO
        service_map[p->m_service_id] = p;
    }
    return service_map;
}
*/

PassingServicePtr Consul::ParsePassingService(const nlohmann::json& obj)
{
    auto service = std::make_shared<PassingService>();
    try {
        service->m_node     = obj["Node"].get<std::string>();
        service->m_check_id = obj["CheckID"].get<std::string>();
        service->m_name     = obj["Name"].get<std::string>();
        service->m_status   = obj["Status"].get<std::string>();
        service->m_notes    = obj["Notes"].get<std::string>();
        service->m_output   = obj["Output"].get<std::string>();
        service->m_service_id = obj["ServiceID"].get<std::string>();
        service->m_service_name = obj["ServiceName"].get<std::string>();
        return service;
    } catch (std::exception& e) {
        LOG(warning) << "exception: " << e.what();
        return nullptr;
    }
}

std::unordered_map<std::string, PassingServicePtr> Consul::ReleasePassingServices()
{
    decltype(m_passing_service_map) temp{};
    {
        std::lock_guard<std::mutex> lk{m_mutex};
        std::swap(temp, m_passing_service_map);
    }
    return temp;
}
