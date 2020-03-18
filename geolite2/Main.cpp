#include <cstdio>

#include <string>
#include <iostream>
#include <vector>

#include <maxminddb.h>

void IPV4_to_Addr(const char* p, IN_ADDR* addr);
int LookupIPV4(MMDB_s* mmdb, const std::string& ip, std::string& contry_code, std::string& contry_name);
bool LookupSockInAddr4CountryCode(MMDB_s* mmdb, const IN_ADDR& remote_addr, std::string* contry_code, std::string* contry_name);
bool GeoDbGetCountryData(MMDB_entry_s* db_entry, std::string* country_code, std::string* country_name);
bool GeoDbGetContinentData(MMDB_entry_s* db_entry, std::string* continent_code, std::string* continent_name);

void IPV4_to_Addr(const char* p, IN_ADDR* addr)
{
    addr->S_un.S_addr = inet_addr(p);
}

int LookupIPV4(MMDB_s* mmdb, const std::string& ip, std::string& contry_code, std::string& contry_name)
{
    if (ip.empty())
        return -1;
    std::string name;
    IN_ADDR addr;
    IPV4_to_Addr(ip.c_str(), &addr);

    bool succ = LookupSockInAddr4CountryCode(mmdb, addr, &contry_code, &contry_name);
    if (!succ)
        return -1;
    return 0;
}

bool LookupSockInAddr4CountryCode(MMDB_s* mmdb, const IN_ADDR& remote_addr, std::string* contry_code, std::string* contry_name)
{
    /*
    if (IN4_IS_ADDR_UNSPECIFIED(&RemoteAddress) || IN4_IS_ADDR_LOOPBACK(&RemoteAddress)) {
        return false;
    }
    */

    MMDB_lookup_result_s mmdb_result;
    SOCKADDR_IN ipv4sa;
    int mmdb_error = 0;
    std::memset(&ipv4sa, 0, sizeof(SOCKADDR_IN));
    std::memset(&mmdb_result, 0, sizeof(MMDB_lookup_result_s));

    ipv4sa.sin_family = AF_INET;
    ipv4sa.sin_addr = remote_addr;
    mmdb_result = ::MMDB_lookup_sockaddr(mmdb, (PSOCKADDR)&ipv4sa, &mmdb_error);

    if (mmdb_error == 0 && mmdb_result.found_entry) {
        if (GeoDbGetCountryData(&mmdb_result.entry, contry_code, contry_name))
            return true;
        if (GeoDbGetContinentData(&mmdb_result.entry, contry_code, contry_name))
            return true;
    }
    return false;
}

bool GeoDbGetCountryData(MMDB_entry_s* db_entry, std::string* country_code, std::string* country_name)
{
    MMDB_entry_data_s data_entry;
    std::string code;
    std::string name;
    if (MMDB_get_value(db_entry, &data_entry, "country", "iso_code", NULL) == MMDB_SUCCESS) {
        if (data_entry.has_data && data_entry.type == MMDB_DATA_TYPE_UTF8_STRING) {
            // code = PhConvertUtf8ToUtf16Ex((PCHAR)mmdb_entry.utf8_string, mmdb_entry.data_size);
            code.assign(data_entry.utf8_string, data_entry.data_size);
        }
    }

    if (MMDB_get_value(db_entry, &data_entry, "country", "names", "en", NULL) == MMDB_SUCCESS) {
        if (data_entry.has_data && data_entry.type == MMDB_DATA_TYPE_UTF8_STRING) {
            // name = PhConvertUtf8ToUtf16Ex((PCHAR)mmdb_entry.utf8_string, mmdb_entry.data_size);
            name.assign(data_entry.utf8_string, data_entry.data_size);
        }
    }

    if (!code.empty() && !name.empty()) {
        *country_code = code;
        *country_name = name;
        return true;
    }
    return false;
}

bool GeoDbGetContinentData(MMDB_entry_s* db_entry, std::string* continent_code, std::string* continent_name)
{
    MMDB_entry_data_s data_entry;
    std::string code;
    std::string name;

    if (::MMDB_get_value(db_entry, &data_entry, "continent", "code", NULL) == MMDB_SUCCESS) {
        if (data_entry.has_data && data_entry.type == MMDB_DATA_TYPE_UTF8_STRING) {
            //code = PhConvertUtf8ToUtf16Ex((PCHAR)mmdb_entry.utf8_string, mmdb_entry.data_size);
            code.assign(data_entry.utf8_string, data_entry.data_size);
        }
    }

    if (MMDB_get_value(db_entry, &data_entry, "country", "names", "en", NULL) == MMDB_SUCCESS) {
        if (data_entry.has_data && data_entry.type == MMDB_DATA_TYPE_UTF8_STRING) {
            //name = PhConvertUtf8ToUtf16Ex((PCHAR)mmdb_entry.utf8_string, mmdb_entry.data_size);
            name.assign(data_entry.utf8_string, data_entry.data_size);
        }
    }

    if (!code.empty() && !name.empty()) {
        *continent_code = code;
        *continent_name = name;
        return true;
    }
    return false;
}


int Test()
{
    std::string db_file = "C:/Users/admin/Desktop/geolite2/GeoLite2-Country_20200303/GeoLite2-Country.mmdb";
    MMDB_s mmdb_;
    MMDB_s* mmdb = &mmdb_;
    int status = ::MMDB_open(db_file.c_str(), MMDB_MODE_MMAP, mmdb);
    if (MMDB_SUCCESS != status) {
        printf("ERROR: open db_file: %s  status: %d\n", db_file.c_str(), status);
        return -1;
    }

    std::vector<std::string> vec =
    {
        "39.156.69.79",
        "172.104.181.124",
        "23.239.14.204",
        "81.2.69.160",
    };

    for (size_t i = 0; i != vec.size(); ++i) {
        std::string ip = vec[i];
        std::string code;
        std::string name;
        LookupIPV4(mmdb, ip, code, name);
        printf("ip: [%s] \t\t code: [%s] \t\t name: [%s]\n", ip.c_str(), code, name);
    }

    ::MMDB_close(mmdb);
    return 0;
}

int main()
{
    Test();
    return 0;
}

