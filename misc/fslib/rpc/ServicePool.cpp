#include "ServicePool.h"

#include <google/protobuf/descriptor.h>

#include "RpcServiceImpl.h"

namespace fslib {
namespace grpc {

ServiceSlot::ServiceSlot(std::shared_ptr<::google::protobuf::Service> service)
{
    m_service = service;
    m_descriptor = m_service->GetDescriptor();
    int countor = m_descriptor->method_count();
    m_method_descriptor.reserve(countor);
    for (int i = 0; i != countor; ++i) {
        m_method_descriptor.push_back(m_descriptor->method(i));
    }
}


std::shared_ptr<::google::protobuf::Service> ServiceSlot::getService() const
{
    return m_service;
}

const ::google::protobuf::ServiceDescriptor* ServiceSlot::getServiceDescriptor() const
{
    return m_descriptor;
}

//////////////////////////////////////////////////////////////////////////
ServicePool::ServicePool()
{
}


bool ServicePool::regService(std::shared_ptr<::google::protobuf::Service> service)
{
    std::lock_guard<std::mutex> lk(m_mtx);
    auto name = service->GetDescriptor()->full_name();
    if (m_services.find(name) != m_services.end()) {
        return false;
    }
    m_services.insert({ name, std::make_shared<ServiceSlot>(service) });
    return true;
}

std::shared_ptr<ServiceSlot> ServicePool::findServiceSlot(std::string service_name) const
{
    std::lock_guard<std::mutex> lk(m_mtx);
    auto it = m_services.find(service_name);
    if (it != m_services.end())
        return it->second;
    return nullptr;
}


//////////////////////////////////////////////////////////////////////////
}
}
