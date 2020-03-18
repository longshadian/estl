#pragma once

#include <string>

#include <modsecurity/modsecurity.h>
#include <modsecurity/transaction.h>
#include <modsecurity/rules_set.h>
#include <modsecurity/rule_message.h>
#include <modsecurity/intervention.h>

namespace common
{

std::string ToString(const modsecurity::ModSecurityIntervention& intervention);
}
