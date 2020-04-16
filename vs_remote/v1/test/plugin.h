#pragma once

#include <string>
#include <memory>
#include <any>
#include <functional>

#include <boost/dll.hpp>

#include "test/log.h"

typedef std::string name_ref_t();
typedef std::string version_ref_t();
typedef std::string desc_ref_t();

typedef std::string (*name_ptr_t)();
typedef std::string (*version_ptr_t)();
typedef std::string (*desc_ptr_t)();

class BoostPlugin
{
public:
    BoostPlugin(const std::string& so_name)
        : so_name_(std::move(so_name))
        , lib_()
        , name_ref_()
        , version_ref_()
        , desc_ref_()
    {
    }

    virtual ~BoostPlugin()
    {

    }

    virtual bool Open()
    {
        try {
            lib_ = std::make_shared<boost::dll::shared_library>(so_name_, boost::dll::load_mode::rtld_lazy);

            name_ref_ = lib_->get<name_ref_t>("name");
            version_ref_ = lib_->get<version_ref_t>("version");
            desc_ref_ = lib_->get<desc_ref_t>("desc");
            return true;
        } catch (const std::exception & e) {
            LOG_WARN << "load dll failed: " << e.what();
            return false;
        }
    }

    virtual void Close()
    {

    }

    std::string                                 so_name_;
    std::shared_ptr<boost::dll::shared_library> lib_;

    std::function<name_ref_t>                     name_ref_;
    std::function<version_ref_t>                  version_ref_;
    std::function<desc_ref_t>                     desc_ref_;
};
typedef std::shared_ptr<BoostPlugin> BoostPluginPtr;

#if defined (__GNUC__)
class Plugin
{
public:
    Plugin(const std::string& so_name)
        : so_name_(so_name)
        , lib_handle_()
        , name_ptr_()
        , version_ptr_()
        , desc_ptr_()
    {
    }

    ~Plugin()
    {
        if (lib_handle_) {
            int n = ::dlclose(lib_handle_);
            std::cout << "dlclode: " << n << "\n";
        }
    }

    bool Open()
    {
        void* hdl = ::dlopen(so_name_.c_str(), RTLD_NOW);
        if (!hdl) {
            LOG_WARN << "dlopen: " << so_name_ << " failure: " << ::dlerror();
            return false;
        }
        lib_handle_ = hdl;

        const char* fun_name = "name";
        name_ptr_ = (name_ptr_t)::dlsym(hdl, fun_name);
        if (!name_ptr_) {
            LOG_WARN << "dlsym: " << fun_name << " failure";
            return false;
        }

        fun_name = "version";
        version_ptr_ = (version_ptr_t)::dlsym(hdl, fun_name);
        if (!version_ptr_) {
            LOG_WARN << "dlsym: " << fun_name << " failure";
            return false;
        }

        fun_name = "desc";
        desc_ptr_ = (desc_ptr_t)::dlsym(hdl, fun_name);
        if (!desc_ptr_) {
            LOG_WARN << "dlsym: " << fun_name << " failure";
            return false;
        }

        return true;
    }

    std::string so_name_;
    void* lib_handle_;
    name_ptr_t          name_ptr_;
    version_ptr_t       version_ptr_;
    desc_ptr_t          desc_ptr_;
};

#endif

