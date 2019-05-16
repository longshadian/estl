#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include <cstdint>
#include <array>
#include <cstring>

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
        printf("%s %s %s %s %s %s\n", mysql_row[0], mysql_row[1], mysql_row[2], mysql_row[3], mysql_row[4], mysql_row[5]);
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

bool selectDB_stmt()
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

    std::string sql = "select fid, fidx, fname, fdatetime, ftext, fdouble, fblob from txx where fid = ?";
    std::unique_ptr<MYSQL_STMT, MysqlStmtDestroy> mysql_stmt{mysql_stmt_init(&*mysql)};
    mysql_stmt_prepare(&*mysql_stmt, sql.data(), sql.size());

    std::unique_ptr<MYSQL_RES, MysqlResDestroy> mysql_res{mysql_store_result(&*mysql)};
    if (!mysql_res) {
        printf("error mysql_use_result\n");
        return false;
    }
    printf("query num: %d %d\n", (int)mysql_num_rows(&*mysql_res), (int)mysql_num_fields(&*mysql_res));

    MYSQL_ROW mysql_row = mysql_fetch_row(&*mysql_res);
    while (mysql_row) {
        printf("%s %s %s %s %s %s\n", mysql_row[0], mysql_row[1], mysql_row[2], mysql_row[3], mysql_row[4], mysql_row[5]);
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

bool delDB()
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

    std::string str = " DELETE FROM `txx`";
    if (mysql_query(&*mysql, str.c_str()) != 0) {
        printf("error mysql_query\n");
        return false;
    }
    //std::unique_ptr<MYSQL_RES, MysqlResDestroy> mysql_res{mysql_store_result(&*mysql)};
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
    std::string sql = "INSERT INTO `txx` (`fblob`, `fname`) VALUES (?, ?)";
    mysql_stmt_prepare(&*mysql_stmt, sql.data(), sql.size());

    std::string fname = "sdfa'a'b'看看空间x";
    std::vector<int32_t> fblob = { 100,0,200,0,123456789,1234567890 };

    std::array<MYSQL_BIND, 2> b{};
    auto fblob_len = fblob.size() * sizeof(int32_t);
    (void)fblob_len;

    //unsigned long len;
    b[0].buffer_type = MYSQL_TYPE_BLOB;
    //b[0].length = &len;

    b[1].buffer_type = MYSQL_TYPE_STRING;
    b[1].buffer = (void*)fname.data();
    unsigned long fname_len = fname.size();
    b[1].length = &fname_len;


    mysql_stmt_bind_param(&*mysql_stmt, b.data());

    /*
    if (mysql_stmt_send_long_data(&*mysql_stmt, 0, (const char*)fblob.data(), fblob_len)) {
        printf("error mysql_stmt_send_long_data\n");
        return false;
    }
    */
    for (const auto& val : fblob) {
        if (mysql_stmt_send_long_data(&*mysql_stmt, 0, (const char*)&val, sizeof(int32_t))) {
            printf("error mysql_stmt_send_long_data 0\n");
            return false;
        }
    }

    if (mysql_stmt_execute(&*mysql_stmt) != 0) {
        printf("error mysql_stmt_execute\n");
        return false;
    }
    return true;
}

bool insertStmt_mysql_stmt_send_long_data()
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
    std::string sql = "INSERT INTO `txx` (`fblob`, `fname`) VALUES (?, ?)";
    mysql_stmt_prepare(&*mysql_stmt, sql.data(), sql.size());

    std::string fname = "sdfa'a'b'看看空间x";
    std::vector<int32_t> fblob = { 100,0,200,0,123456789,1234567890 };

    std::array<MYSQL_BIND, 2> b{};
    auto fblob_len = fblob.size() * sizeof(int32_t);
    (void)fblob_len;

    //unsigned long len;
    b[0].buffer_type = MYSQL_TYPE_BLOB;
    //b[0].length = &len;

    b[1].buffer_type = MYSQL_TYPE_STRING;
    b[1].buffer = (void*)fname.data();
    unsigned long fname_len = fname.size();
    b[1].length = &fname_len;


    mysql_stmt_bind_param(&*mysql_stmt, b.data());

    /*
    if (mysql_stmt_send_long_data(&*mysql_stmt, 0, (const char*)fblob.data(), fblob_len)) {
        printf("error mysql_stmt_send_long_data\n");
        return false;
    }
    */
    for (const auto& val : fblob) {
        if (mysql_stmt_send_long_data(&*mysql_stmt, 0, (const char*)&val, sizeof(int32_t))) {
            printf("error mysql_stmt_send_long_data 0\n");
            return false;
        }
    }

    if (mysql_stmt_execute(&*mysql_stmt) != 0) {
        printf("error mysql_stmt_execute\n");
        return false;
    }
    return true;
}

bool insertStmt_bind()
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
    std::string sql = "INSERT INTO `txx` (`fblob`, `fname`) VALUES (?, ?)";
    mysql_stmt_prepare(&*mysql_stmt, sql.data(), sql.size());

    std::string fname = "sdfa'a'b'看看空间x";
    //std::vector<int32_t> fblob = { 100,0,200,0,123456789,1234567890 };
    std::vector<uint8_t> fblob;
    fblob.resize(16777208 - 64);

    std::array<MYSQL_BIND, 1> b{};

    auto fblob_len = fblob.size() * sizeof(uint8_t);
    b[0].buffer_type = MYSQL_TYPE_BLOB;
    b[0].buffer = (void *)fblob.data();
    b[0].length = (unsigned long *)&fblob_len;
    std::cout << fblob_len << "\n";

    b[1].buffer_type = MYSQL_TYPE_STRING;
    b[1].buffer = (void*)fname.data();
    unsigned long fname_len = fname.size();
    b[1].length = &fname_len;

    mysql_stmt_bind_param(&*mysql_stmt, b.data());
    auto ret = mysql_stmt_execute(&*mysql_stmt);
    if (ret != 0) {
        printf("error mysql_stmt_execute [%d] [%s]\n", ret, mysql_stmt_error(&*mysql_stmt));
        return false;
    }
    return true;
}

bool insertStmt_bind_null()
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
    std::string sql = "INSERT INTO `txx` (`fname`) VALUES ('haha')";
    mysql_stmt_prepare(&*mysql_stmt, sql.data(), sql.size());

    //mysql_stmt_bind_param(&*mysql_stmt, b.data());
    auto ret = mysql_stmt_execute(&*mysql_stmt);
    if (ret != 0) {
        printf("error mysql_stmt_execute [%d] [%s]\n", ret, mysql_stmt_error(&*mysql_stmt));
        return false;
    }
    return true;
}

bool updateStmtBind()
{
    std::unique_ptr<MYSQL, MysqlDestroy> mysql{ mysql_init(nullptr) };
    if (mysql_options(&*mysql, MYSQL_SET_CHARSET_NAME, "utf8") != 0) {
        printf("error mysql_options\n");
        return false;
    }

    if (!mysql_real_connect(&*mysql, IP.c_str(), USER.c_str(), PASSWORD.c_str(), "mytest", PORT, nullptr, 0)) {
        printf("error mysql_real_connect\n");
        return false;
    }

    std::unique_ptr<MYSQL_STMT, MysqlStmtDestroy> mysql_stmt{mysql_stmt_init(&*mysql)};
    std::string sql = "update test set fname = ?, fsid = ?, ffloat = ? where fid = 1";
    mysql_stmt_prepare(&*mysql_stmt, sql.data(), sql.size());

    std::array<MYSQL_BIND, 3> b{};
    std::memset(b.data(), 0, sizeof(b));

    const char* fname = "hhh";
    b[0].buffer_type = MYSQL_TYPE_STRING;
    b[0].buffer = (void*)fname;
    b[0].buffer_length = std::strlen(fname);

    int32_t fsid = 255;
    b[1].buffer_type = MYSQL_TYPE_TINY;
    b[1].buffer = &fsid;
    b[1].is_unsigned = 0;

    float ffloat = -1231.223;
    //b[2].buffer_type = MYSQL_TYPE_FLOAT;
    b[2].buffer_type = MYSQL_TYPE_NULL;
    b[2].buffer = &ffloat;
    /*
    my_bool is_null = 1;
    b[2].is_null = &is_null;
    */


    mysql_stmt_bind_param(&*mysql_stmt, b.data());
    auto ret = mysql_stmt_execute(&*mysql_stmt);
    if (ret != 0) {
        printf("error mysql_stmt_execute [%d] [%s]\n", ret, mysql_stmt_error(&*mysql_stmt));
        return false;
    }
    return true;
}

bool selectStmt()
{
    std::unique_ptr<MYSQL, MysqlDestroy> mysql{mysql_init(nullptr)};
    if (mysql_options(&*mysql, MYSQL_SET_CHARSET_NAME, "utf8") != 0) {
        printf("error mysql_options\n");
        return false;
    }
    if (!mysql_real_connect(&*mysql, IP.c_str(), USER.c_str(), PASSWORD.c_str(), "mytest", PORT,nullptr,0)) {
        printf("error mysql_real_connect\n");
        return false;
    }

    const std::string sql = "select fid, fname, fsid from test where fid = 1";
    std::unique_ptr<MYSQL_STMT, MysqlStmtDestroy> mysql_stmt{mysql_stmt_init(&*mysql)};
    mysql_stmt_prepare(&*mysql_stmt, sql.data(), sql.size());

    if (mysql_stmt_execute(&*mysql_stmt) != 0) {
        printf("error mysql_stmt_execute\n");
        return false;
    }
    mysql_stmt_store_result(&*mysql_stmt);

    std::array<MYSQL_BIND,3> bind{};
    std::array<std::array<char, 50>, 3> bind_buffer{};    
    std::array<unsigned long, 3> bind_length{};
    std::memset(bind.data(), 0, sizeof(bind));

    bind[0].buffer = bind_buffer[0].data();
    bind[0].buffer_length = 50;
    bind[0].length = &bind_length[0];

    bind[1].buffer = bind_buffer[1].data();
    bind[1].buffer_length = 50;
    bind[1].length = &bind_length[1];

    bind[2].buffer = bind_buffer[2].data();
    bind[2].buffer_length = 50;
    bind[2].length = &bind_length[2];

    if (mysql_stmt_bind_result(&*mysql_stmt, bind.data())) {
        printf("mysql_stmt_bind_result error\n");
        return false;
    }

    while (mysql_stmt_fetch(&*mysql_stmt) == 0) {
        int fid = 0;
        std::memcpy(&fid, bind[0].buffer, *bind[0].length);

        char* p = (char*)bind[1].buffer;
        for (unsigned long i = 0; i != *bind[1].length; ++i) {
            printf("%c ", p[i]);
        }
        printf("\n");
    }
    return true;
}

int main()
{
    //insertDB();
    //insertStmt();

    //delDB();
    //insertStmt_bind_null();
    //insertStmt_bind();
    //selectDB();

    //updateStmtBind();

    selectStmt();
    return 0;
}