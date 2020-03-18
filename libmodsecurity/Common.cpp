#include "Common.h"


#include <modsecurity/modsecurity.h>
#include <modsecurity/transaction.h>
#include <modsecurity/rules_set.h>
#include <modsecurity/rule_message.h>
#include <modsecurity/intervention.h>

namespace common
{

std::string ToString(const modsecurity::ModSecurityIntervention& intervention)
{
    std::ostringstream ostm{};
    ostm << "{ ";
    ostm << "status: " << intervention.status << ", ";
    ostm << "pause: " << intervention.pause << ", ";
    ostm << "url: " << (intervention.url ? intervention.url : "") << ", ";
    ostm << "log: " << (intervention.log ? intervention.log : "") << ", ";
    ostm << "disruptive: " << intervention.disruptive;
    ostm << " }";
    return ostm.str();
}

}
