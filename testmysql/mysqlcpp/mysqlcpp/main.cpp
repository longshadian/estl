#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include <cstdint>
#include <array>
#include <type_traits>

#include <mysql.h>

const std::string IP = "127.0.0.1";
const std::string USER = "root";
const std::string PASSWORD = "123456";
const unsigned int PORT = 3306;

struct MysqlDestroy
{
    void operator()(MYSQL* mysql)
    {
        if (mysql)
            mysql_close(mysql);
    }
};

struct MysqlResDestroy
{
    void operator()(MYSQL_RES* mysql_res)
    {
        if (mysql_res)
            mysql_free_result(mysql_res);
    }
};

struct MysqlStmtDestroy
{
    void operator()(MYSQL_STMT* mysql_stmt)
    {
        if (mysql_stmt)
            mysql_stmt_close(mysql_stmt);
    }
};


std::vector<char> escapeString(MYSQL* mysql, const char* data, size_t data_len)
{
    std::vector<char> buf{};
    buf.resize(data_len * 2 + 1);
    auto len = mysql_real_escape_string(mysql, buf.data(), data, data_len);
    buf.resize(len);
    return buf;
}

bool selectDB()
{
    std::unique_ptr<MYSQL, MysqlDestroy> mysql{mysql_init(nullptr)};
    if (mysql_options(&*mysql, MYSQL_SET_CHARSET_NAME, "utf8") != 0) {
        printf("error mysql_options\n");
        return false;
    }
    if (!mysql_real_connect(&*mysql, IP.c_str(), USER.c_str(), PASSWORD.c_str(), "test", PORT,nullptr,0)) {
        printf("error mysql_real_connect\n");
        return false;
    }

    std::string str = "select fid, fidx, fname, fdatetime, ftext, fdouble, fblob from txx";
    if (mysql_query(&*mysql, str.c_str()) != 0) {
        printf("error mysql_query\n");
        return false;
    }

    std::unique_ptr<MYSQL_RES, MysqlResDestroy> mysql_res{mysql_store_result(&*mysql)};
    if (!mysql_res) {
        printf("error mysql_use_result\n");
        return false;
    }
    printf("query num: %d %d\n", (int)mysql_num_rows(&*mysql_res), (int)mysql_num_fields(&*mysql_res));

    MYSQL_ROW mysql_row = mysql_fetch_row(&*mysql_res);
    while (mysql_row) {
        printf("[%s] [%s] [%s] [%s] [%s] [%s]\n", mysql_row[0], mysql_row[1], mysql_row[2], mysql_row[3], mysql_row[4], mysql_row[5]);
        auto* lengths = mysql_fetch_lengths(&*mysql_res);
        const uint8_t* row_data = (const uint8_t*)mysql_row[6];
        auto row_len = lengths[6];
        std::vector<uint8_t> fblob{row_data, row_data + row_len};

        auto len = fblob.size()/sizeof(int32_t);
        const int32_t* p = (const int32_t*)fblob.data();
        for (size_t i = 0; i != len; ++i) {
            printf("\t\t val:%d\n", p[i]);
        }

        /*
        for (auto v : fblob) {
            printf("\t\t val:%d\n", (int)v);
            (void)v;
        }
        */

        mysql_row = mysql_fetch_row(&*mysql_res);
    }

    return true;
}

bool insertDB()
{
    std::unique_ptr<MYSQL, MysqlDestroy> mysql{ mysql_init(nullptr) };
    if (!mysql_real_connect(&*mysql, IP.c_str(), USER.c_str(), PASSWORD.c_str(), "test", PORT, nullptr, 0)) {
        printf("error mysql_real_connect\n");
        return false;
    }

    std::string sql = "INSERT INTO `txx` (`fname`, `fblob`) VALUES ('";

    std::string val = "sddfa\r\n'a'b'c'";
    auto buff = escapeString(&*mysql, val.data(), val.size());
    sql.append(buff.begin(), buff.end());
    sql += "','";

    std::vector<int32_t> blob = {100,0,200,0,123456789,1234567890};
    buff = escapeString(&*mysql, (const char*)blob.data(), blob.size() * sizeof(int32_t));
    sql.append(buff.begin(), buff.end());
    sql += "')";
    if (mysql_real_query(&*mysql, sql.data(), sql.size()) != 0) {
        printf("error mysql_query\n");
        return false;
    }
    return true;
}

bool insertStmt()
{
    std::unique_ptr<MYSQL, MysqlDestroy> mysql{ mysql_init(nullptr) };
    if (mysql_options(&*mysql, MYSQL_SET_CHARSET_NAME, "utf8") != 0) {
        printf("error mysql_options\n");
        return false;
    }

    if (!mysql_real_connect(&*mysql, IP.c_str(), USER.c_str(), PASSWORD.c_str(), "test", PORT, nullptr, 0)) {
        printf("error mysql_real_connect\n");
        return false;
    }

    std::unique_ptr<MYSQL_STMT, MysqlStmtDestroy> mysql_stmt{mysql_stmt_init(&*mysql)};
    std::string sql = "INSERT INTO `txx` (`fblob`, `fname`, `fdouble`, `fdatetime`) VALUES (?, ?, ?, ?)";
    mysql_stmt_prepare(&*mysql_stmt, sql.data(), sql.size());

    std::string fname = "sdfa'a'b'看看空间x";
    std::vector<int32_t> fblob = { 100,0,200,0,123456789,1234567890 };
    double fdouble = 121133.2223;

    std::array<MYSQL_BIND, 4> b{};
    auto fblob_len = fblob.size() * sizeof(int32_t);
    (void)fblob_len;

    //unsigned long len;
    b[0].buffer_type = MYSQL_TYPE_BLOB;
    b[0].buffer = (void*)fblob.data();
    b[0].length = (unsigned long *)&fblob_len;

    b[1].buffer_type = MYSQL_TYPE_STRING;
    b[1].buffer = (void*)fname.data();
    unsigned long fname_len = fname.size();
    b[1].length = &fname_len;

    b[2].buffer_type = MYSQL_TYPE_DOUBLE;
    b[2].buffer = (void*)&fdouble;

    MYSQL_TIME mt{};
    mt.year = 2016;
    mt.month = 10;
    mt.day = 2;
    mt.hour = 15;
    mt.minute = 19;
    mt.second = 20;
    mt.time_type = MYSQL_TIMESTAMP_DATE;

    b[3].buffer_type = MYSQL_TYPE_DATETIME;
    b[3].buffer = (void*)&mt;

    mysql_stmt_bind_param(&*mysql_stmt, b.data());

    /*
    if (mysql_stmt_send_long_data(&*mysql_stmt, 0, (const char*)fblob.data(), fblob_len)) {
        printf("error mysql_stmt_send_long_data\n");
        return false;
    }
    */
    /*
    for (const auto& val : fblob) {
        if (mysql_stmt_send_long_data(&*mysql_stmt, 0, (const char*)&val, sizeof(int32_t))) {
            printf("error mysql_stmt_send_long_data 0\n");
            return false;
        }
    }
    */

    if (mysql_stmt_execute(&*mysql_stmt) != 0) {
        printf("error mysql_stmt_execute\n");
        return false;
    }
    return true;
}


int main()
{
    //insertDB();
    insertStmt();
    selectDB();

    std::cout << "bool:" << std::is_fundamental<bool>::value << "\n";
    std::cout << "short:" << std::is_fundamental<short>::value << "\n";
    std::cout << "int:" << std::is_fundamental<int>::value << "\n";
    std::cout << "double:" << std::is_fundamental<double>::value << "\n";
    std::cout << "long:" << std::is_fundamental<long>::value << "\n";
    std::cout << "long long:" << std::is_fundamental<long long>::value << "\n";

    std::cout << "unsigned:" << std::is_fundamental<unsigned int>::value << "\n";
    std::cout << "unsigned long:" << std::is_fundamental<unsigned long>::value << "\n";
    std::cout << "unsigned long long:" << std::is_fundamental<unsigned long long>::value << "\n";

    std::cout << "int*:" << std::is_fundamental<int*>::value << "\n";
    std::cout << "const char*:" << std::is_fundamental<const char*>::value << "\n";
    return 0;
}