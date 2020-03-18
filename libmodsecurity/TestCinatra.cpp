#include "TestCinatra.h"

#include <string>
#include <memory>
#include <thread>

#include <modsecurity/modsecurity.h>
#include <modsecurity/transaction.h>
#include <modsecurity/rules_set.h>
#include <modsecurity/rule_message.h>
#include <modsecurity/intervention.h>

#include <boost/asio.hpp>

#include <cinatra.hpp>
using namespace cinatra;

#include "FakeLog.h"
#include "Common.h"


std::shared_ptr<modsecurity::ModSecurity> g_modsec;
std::shared_ptr<modsecurity::RulesSet> g_rules;


struct Intervention
{
    Intervention()
    {
        intervention.status = 200;
        intervention.pause = 0;
        intervention.url = nullptr;
        intervention.log = nullptr;
        intervention.disruptive = 0;
    }

    ~Intervention()
    {
        if (intervention.url) {
            free(intervention.url);
        }
        if (intervention.log) {
            free(intervention.log);
        }
    }

    modsecurity::ModSecurityIntervention intervention;
};


static void ModSec_LogCb(void* data, const void* ruleMessagev) 
{
    if (ruleMessagev == NULL) {
        std::cout << "I've got a call but the message was null ;(";
        std::cout << std::endl;
        return;
    }

    const modsecurity::RuleMessage* ruleMessage = \
        reinterpret_cast<const modsecurity::RuleMessage*>(ruleMessagev);

    std::cout << "Rule Id: " << std::to_string(ruleMessage->m_ruleId);
    std::cout << " phase: " << std::to_string(ruleMessage->m_phase);
    std::cout << std::endl;
    if (ruleMessage->m_isDisruptive) {
        std::cout << " * Disruptive action: ";
        std::cout << modsecurity::RuleMessage::log(ruleMessage);
        std::cout << std::endl;
        std::cout << " ** %d is meant to be informed by the webserver.";
        std::cout << std::endl;
    }
    else {
        std::cout << " * Match, but no disruptive action: ";
        std::cout << modsecurity::RuleMessage::log(ruleMessage);
        std::cout << std::endl;
    }
}

static bool GetIP_Port(const boost::asio::ip::tcp::endpoint& ep, std::string& ip, int& port)
{
    try {
        ip = ep.address().to_string();
        port = static_cast<int>(ep.port());
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

static int InitModSec(const std::string& rules_conf)
{
    g_modsec = std::make_shared<modsecurity::ModSecurity>();
    g_modsec->setConnectorInformation("ModSecurity-test v0.0.1-alpha  (ModSecurity test)");
    //g_modsec->setServerLogCb(ModSec_LogCb, modsecurity::RuleMessageLogProperty | modsecurity::IncludeFullHighlightLogProperty);

    g_rules = std::make_shared<modsecurity::RulesSet>();
    if (g_rules->loadFromUri(rules_conf.c_str()) < 0) {
        FAKELOG_WARN("Problems loading file: %s  reason: %s", rules_conf.c_str(), g_rules->m_parserError.str().c_str());
        return -1;
    }
    FAKELOG_DEBUG("loading file: %s success!", rules_conf.c_str());
    return 0;
}

static void PrintRequest(request& req)
{
    auto& sock = req.get_conn()->socket();
    std::string local_ip;
    int local_port = 80;
    std::string remote_ip = "";
    int remote_port = 111;
    GetIP_Port(sock.local_endpoint(), local_ip, local_port);
    GetIP_Port(sock.remote_endpoint(), remote_ip, remote_port);

    std::string http_version = "1.1";
    std::string url = std::string(req.get_url());
    std::string http_method = std::string(req.get_method());
    FAKELOG_DEBUG("local: %s:%d remote: %s:%d %s %s %s", local_ip.c_str(), local_port, remote_ip.c_str(), remote_port,
    url.c_str(), http_method.c_str(), http_version.c_str());
}

struct CheckModSec
{
    bool before(request& req, response& res)
    {
        auto modsecTransaction = std::make_shared<modsecurity::Transaction>(g_modsec.get(), g_rules.get(), nullptr);
        req.get_conn()->set_tag(std::any(modsecTransaction));

        auto& sock = req.get_conn()->socket();
        std::string local_ip;
        int local_port = 80;
        std::string remote_ip = "";
        int remote_port = 0;
        GetIP_Port(sock.local_endpoint(), local_ip, local_port);
        GetIP_Port(sock.remote_endpoint(), remote_ip, remote_port);

        std::string http_version = "1.1";
        std::string url = std::string(req.get_url());
        std::string http_method = std::string(req.get_method());

        modsecTransaction->processConnection(remote_ip.c_str(), remote_port, local_ip.c_str(), local_port);
        modsecTransaction->processURI(url.c_str(), http_method.c_str(), http_version.c_str());

        int ecode = 0;
        auto [h, len] = req.get_headers();
        for (std::size_t i = 0; i != len; ++i) {
            std::string key(h[i].name, h[i].name_len);
            std::string value(h[i].value, h[i].value_len);
            ecode = modsecTransaction->addRequestHeader(key, value);
        }
        ecode = modsecTransaction->processRequestHeaders();

        Intervention it;
        modsecTransaction->intervention(&it.intervention);
        FAKELOG_DEBUG("processRequestHeaders code: %d %s", ecode, common::ToString(it.intervention).c_str());
        if (it.intervention.status != 200) {
            res.set_status_and_content(static_cast<cinatra::status_type>(it.intervention.status));
            modsecTransaction->processLogging();
            return false;
        }

        ecode = modsecTransaction->appendRequestBody(reinterpret_cast<const unsigned char*>(req.body().data()), req.body().length());
        ecode = modsecTransaction->processRequestBody();

        Intervention it2;
        modsecTransaction->intervention(&it2.intervention);
        FAKELOG_DEBUG("processRequestBody code: %d %s", ecode, common::ToString(it2.intervention).c_str());
        if (it2.intervention.status != 200) {
            res.set_status_and_content(static_cast<cinatra::status_type>(it2.intervention.status));
            modsecTransaction->processLogging();
            return false;
        }
        return true;
    }

    bool after(request& req, response& res)
    {
        std::shared_ptr<modsecurity::Transaction> modsecTransaction = nullptr;
        try {
            modsecTransaction = std::any_cast<std::shared_ptr<modsecurity::Transaction>>(req.get_conn()->get_tag());
        } catch (const std::exception&) {
        }
        if (!modsecTransaction) {
            return false;
        }

        // TODO
        /*
        int ecode = 0;
        ecode = modsecTransaction->addResponseHeader("HTTP/1.1", "200 OK");
        ecode = modsecTransaction->processResponseHeaders(403, "HTTP 1.2");
        ecode = modsecTransaction->appendResponseBody((const unsigned char*)response_body, strlen((const char*)response_body));
        ecode = modsecTransaction->processResponseBody();
        */

        modsecTransaction->processLogging();
        return true;
    }
};

int TestCinatra(int argc, char** argv)
{
    std::string rules_conf;
    if (argc == 1) {
        rules_conf = "conf/modsecurity.conf";
    } else {
        rules_conf = argv[1];
    }
    if (InitModSec(rules_conf) != 0) {
        return -1;
    }

    //int max_thread_num = std::thread::hardware_concurrency();
    int max_thread_num = 2;
    http_server server(max_thread_num);
    server.listen("0.0.0.0", "8080");
    server.set_http_handler<GET, POST>("/ping", [](request& req, response& res) {
        //PrintRequest(req);
        res.set_status_and_content(status_type::ok, "hello world success");
    }, CheckModSec{});

    server.run();
    return 0;
}


