#include <cstdio>

#include <string>
#include <iostream>
#include <vector>

#include <maxminddb/maxminddb.h>

void IPV4_to_Addr(const char* p, IN_ADDR* addr);

bool Lookup_City(MMDB_s* mmdb, const IN_ADDR& remote_addr, std::string* city_name);
bool Lookup_Country(MMDB_s* mmdb, const IN_ADDR& remote_addr, std::string* country_code, std::string* country_name);

bool Geo_GetCountryData(MMDB_entry_s* db_entry, std::string* country_code, std::string* country_name);
bool Geo_GetContinentData(MMDB_entry_s* db_entry, std::string* continent_code, std::string* continent_name);
bool Geo_GetCityData(MMDB_entry_s* db_entry, std::string* city_name);

void IPV4_to_Addr(const char* p, IN_ADDR* addr)
{
    addr->S_un.S_addr = inet_addr(p);
}

int LookupIPV4_Country(MMDB_s* mmdb, const std::string& ip, std::string& country_code, std::string& country_name)
{
    if (ip.empty())
        return -1;
    std::string name;
    IN_ADDR addr;
    IPV4_to_Addr(ip.c_str(), &addr);

    bool succ = Lookup_Country(mmdb, addr, &country_code, &country_name);
    if (!succ)
        return -1;
    return 0;
}

int LookupIPV4_City(MMDB_s* mmdb, const std::string& ip, std::string& country_name)
{
    if (ip.empty())
        return -1;
    std::string name;
    IN_ADDR addr;
    IPV4_to_Addr(ip.c_str(), &addr);

    bool succ = Lookup_City(mmdb, addr, &country_name);
    if (!succ)
        return -1;
    return 0;
}

bool Lookup_Country(MMDB_s* mmdb, const IN_ADDR& addr, std::string* country_code, std::string* country_name)
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
    ipv4sa.sin_addr = addr;
    mmdb_result = ::MMDB_lookup_sockaddr(mmdb, (PSOCKADDR)&ipv4sa, &mmdb_error);

    if (mmdb_error == 0 && mmdb_result.found_entry) {
        if (Geo_GetCountryData(&mmdb_result.entry, country_code, country_name))
            return true;
        if (Geo_GetContinentData(&mmdb_result.entry, country_code, country_name))
            return true;
    }
    return false;
}

bool Lookup_City(MMDB_s* mmdb, const IN_ADDR& remote_addr, std::string* city_name)
{
    MMDB_lookup_result_s mmdb_result;
    SOCKADDR_IN ipv4sa;
    int mmdb_error = 0;
    std::memset(&ipv4sa, 0, sizeof(SOCKADDR_IN));
    std::memset(&mmdb_result, 0, sizeof(MMDB_lookup_result_s));

    ipv4sa.sin_family = AF_INET;
    ipv4sa.sin_addr = remote_addr;
    mmdb_result = ::MMDB_lookup_sockaddr(mmdb, (PSOCKADDR)&ipv4sa, &mmdb_error);

    if (mmdb_error == 0 && mmdb_result.found_entry) {
        if (Geo_GetCityData(&mmdb_result.entry, city_name))
            return true;
    }
    return false;
}

bool Geo_GetCountryData(MMDB_entry_s* db_entry, std::string* country_code, std::string* country_name)
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

bool Geo_GetCityData(MMDB_entry_s* db_entry, std::string* city_name)
{
    MMDB_entry_data_s data_entry;
    std::string code;
    std::string name;

    if (MMDB_get_value(db_entry, &data_entry, "city", "names", "en", NULL) == MMDB_SUCCESS) {
        if (data_entry.has_data && data_entry.type == MMDB_DATA_TYPE_UTF8_STRING) {
            //name = PhConvertUtf8ToUtf16Ex((PCHAR)mmdb_entry.utf8_string, mmdb_entry.data_size);
            name.assign(data_entry.utf8_string, data_entry.data_size);
        }
    }

    if (!name.empty()) {
        *city_name = name;
        return true;
    }
    return false;
}

bool Geo_GetContinentData(MMDB_entry_s* db_entry, std::string* continent_code, std::string* continent_name)
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
    //std::string db_file = "C:/Users/admin/Desktop/geolite2/GeoLite2-Country_20200303/GeoLite2-Country.mmdb";
    std::string db_file = "C:/Users/admin/Desktop/geolite2/GeoLite2-City_20200317/GeoLite2-City.mmdb";
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
        std::string country_code;
        std::string country_name;
        LookupIPV4_Country(mmdb, ip, country_code, country_name);

        std::string city_name;
        LookupIPV4_City(mmdb, ip, city_name);
        printf("ip: [%s] \t country: [%s] [%s] \t\t city: [%s]\n"
            , ip.c_str(), country_code.c_str(), country_name.c_str(), city_name.c_str());
    }

    ::MMDB_close(mmdb);
    return 0;
}

int main()
{
    Test();
    return 0;
}

