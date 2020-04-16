#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <set>
#include <thread>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "test/utils.h"
#include "test/log.h"
#include "test/plugin.h"

std::shared_ptr<BoostPlugin> CreatePlugin();

int Test()
{
    auto plugin = CreatePlugin();
    LOG_INFO << "open: " << plugin->Open();

    LOG_INFO << "name: " << plugin->name_ref_();
    LOG_INFO << "version: " << plugin->version_ref_();
    LOG_INFO << "desc: " << plugin->desc_ref_();
    return 0;
}

