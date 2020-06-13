#pragma once

#include <memory>
#include <map>
#include <vector>
#include <mutex>

#include <google/protobuf/service.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/message.h>

namespace fslib {
namespace grpc {

class ServiceSlot
{
public:
                            ServiceSlot(std::shared_ptr<::google::protobuf::Service> service);
                            ~ServiceSlot() = default;
    std::shared_ptr<::google::protobuf::Service> getService() const;
    const ::google::protobuf::ServiceDescriptor* getServiceDescriptor() const;
private:
    std::shared_ptr<::google::protobuf::Service>        m_service;
    const ::google::protobuf::ServiceDescriptor*        m_descriptor = { nullptr };
    std::vector<const ::google::protobuf::MethodDescriptor*>  m_method_descriptor;
};

class ServicePool
{
public:
                                    ServicePool();
                                    ~ServicePool() = default;
    bool                            regService(std::shared_ptr<::google::protobuf::Service> service);
    std::shared_ptr<ServiceSlot>    findServiceSlot(std::string service_name) const;
private:
    mutable std::mutex                                  m_mtx; 
    std::map<std::string, std::shared_ptr<ServiceSlot>> m_services;
};

}
}
