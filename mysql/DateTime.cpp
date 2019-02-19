#include <cstdlib>
#include <cstring>
#include <memory>
#include <chrono>
#include <iostream>

#include <mysql.h>

std::string db_name = "my_test";
std::string db_ip   = "192.168.207.128";
std::string db_user = "root";
std::string db_passwd = "123456";
int db_port         = 3306;
std::string db_charset = "utf8";

MYSQL* init()
{
    MYSQL* mysql = ::mysql_init(nullptr);
    if (!mysql) {
        fprintf(stdout, "Could not initialize Mysql connection to database\n");
        return nullptr;
    }

    ::mysql_options(mysql, MYSQL_SET_CHARSET_NAME, "utf8");

    MYSQL* m_mysql = ::mysql_real_connect(mysql, db_ip.c_str(), db_user.c_str(), db_passwd.c_str(), db_name.c_str(), db_port, nullptr, 0);
    if (m_mysql) {
        ::mysql_autocommit(m_mysql, 1);
        ::mysql_set_character_set(m_mysql, "utf8");
        return m_mysql;
    } else {
        printf("Could not connect to MySQL database %s\n", ::mysql_error(mysql));
        ::mysql_close(mysql);
        return nullptr;
    }
}

void printTM(MYSQL_TIME* tm)
{
    fprintf(stdout, " %04d-%02d-%02d %02d:%02d:%02d %lu\n",
        tm->year,
        tm->month,
        tm->day,
        tm->hour,
        tm->minute,
        tm->second,
        tm->second_part
    );
}

void funStmt()
{
    MYSQL* mysql = init();
    if (!mysql) {
        printf("error\n");
        return;
    }

    MYSQL_STMT* stmt = mysql_stmt_init(mysql);
    if (!stmt) {
        fprintf(stderr, " mysql_stmt_init(), out of memory\n");
        return;
    }
    const char* sql = "SELECT `id`, `fdatetime`, `ftime`, `fdate`, `ftimestamp` FROM `test_datetime` ";
    if (mysql_stmt_prepare(stmt, sql, std::strlen(sql))) {
        fprintf(stderr, " mysql_stmt_prepare(), SELECT failed\n");
        fprintf(stderr, " %s\n", mysql_stmt_error(stmt));
        return;
    }

    int param_count = (int)mysql_stmt_param_count(stmt);

    fprintf(stdout, " total parameters in SELECT: %d\n", param_count);
    if (param_count != 0) {
        fprintf(stderr, " invalid parameter count returned by MySQL\n");
        return;
    }

    MYSQL_RES* prepare_meta_result = mysql_stmt_result_metadata(stmt);
    if (!prepare_meta_result) {
        fprintf(stderr, " mysql_stmt_result_metadata(),  returned no meta information\n");
        fprintf(stderr, " %s\n", mysql_stmt_error(stmt));
        return;
    }

    int column_count = mysql_num_fields(prepare_meta_result);
    fprintf(stdout, " total columns in SELECT statement: %d\n", column_count);
    if (column_count != 5) {
        fprintf(stderr, " invalid column count returned by MySQL\n");
        return;
    }

    if (mysql_stmt_execute(stmt)) {
        fprintf(stderr, " mysql_stmt_execute(), failed\n");
        fprintf(stderr, " %s\n", mysql_stmt_error(stmt));
        return;
    }

    MYSQL_BIND bind[5];
    MYSQL_TIME ts; (void)ts;
    unsigned long length[5];
    my_bool is_null[5];
    my_bool error[5];
    int int_data[5];
    unsigned long str_data_len = 50;
    char str_data[5][50];
    MYSQL_TIME my_time[5];

    std::memset(bind, 0, sizeof(bind));
    std::memset(length, 0, sizeof(length));
    std::memset(is_null, 0, sizeof(is_null));
    std::memset(error, 0, sizeof(error));
    std::memset(int_data, 0, sizeof(int_data));
    std::memset(str_data, 0, sizeof(str_data));
    std::memset(my_time, 0, sizeof(my_time));


    bind[0].buffer_type = MYSQL_TYPE_LONG;
    bind[0].buffer = (char *)&int_data[0];
    bind[0].is_null = &is_null[0];
    bind[0].length = &length[0];
    bind[0].error = &error[0];

    bind[1].buffer_type = MYSQL_TYPE_DATETIME;
    bind[1].buffer = (char *)&my_time[1];
    bind[1].buffer_length = str_data_len;
    bind[1].is_null = &is_null[1];
    bind[1].length = &length[1];
    bind[1].error = &error[1];

    bind[2].buffer_type = MYSQL_TYPE_TIME;
    bind[2].buffer = (char *)&my_time[2];
    bind[1].buffer_length = str_data_len;
    bind[2].is_null = &is_null[2];
    bind[2].length = &length[2];
    bind[2].error = &error[2];

    bind[3].buffer_type = MYSQL_TYPE_DATE;
    bind[3].buffer = (char *)&my_time[3];
    bind[3].buffer_length = str_data_len;
    bind[3].is_null = &is_null[3];
    bind[3].length = &length[3];
    bind[3].error = &error[3];

    bind[4].buffer_type = MYSQL_TYPE_TIMESTAMP;
    bind[4].buffer = (char *)&str_data[4];
    bind[4].buffer_length = str_data_len;
    bind[4].is_null = &is_null[4];
    bind[4].length = &length[4];
    bind[4].error = &error[4];

    /* TIMESTAMP COLUMN */
    /*
    bind[4].buffer_type = MYSQL_TYPE_TIMESTAMP;
    bind[4].buffer = &my_time[4];
    bind[4].is_null = &is_null[4];
    bind[4].length = &length[4];
    bind[4].error = &error[4];
    */

    if (mysql_stmt_bind_result(stmt, bind)) {
        fprintf(stderr, " mysql_stmt_bind_result() failed\n");
        fprintf(stderr, " %s\n", mysql_stmt_error(stmt));
        return;
    }

    if (mysql_stmt_store_result(stmt)) {
        fprintf(stderr, " mysql_stmt_store_result() failed\n");
        fprintf(stderr, " %s\n", mysql_stmt_error(stmt));
        return;
    }

    int32_t row = 0;
    while (!mysql_stmt_fetch(stmt)) {
        if (bind[row].buffer_type == MYSQL_TYPE_LONG) {
            std::cout << "MYSQL_TYPE_LONG " << int_data[0] << "\t" << length[0] << "\n";
            //} else if (bind[row].buffer_type == MYSQL_TYPE_TIMESTAMP) {
        }
        std::cout << str_data[1] << "\t==" << length[1] << "==\n";
        std::cout << str_data[2] << "\t==" << length[2] << "==\n";
        std::cout << str_data[3] << "\t==" << length[3] << "==\n";
        std::cout << str_data[4] << "\t==" << length[4] << "==\n";

        printTM(&my_time[1]);
        printTM(&my_time[2]);
        printTM(&my_time[3]);

        std::memcpy(&my_time[4], str_data[4], length[4]);
        printTM(&my_time[4]);
        ++row;
    }
}

void fun()
{
    const char* sql = " SELECT `id`, `fdatetime`, `ftime`, `fdate`, `ftimestamp`, `fyear` FROM `test_datetime` "
        " where `id` = 1";
    MYSQL* mysql = init();
    if (!mysql) {
        printf("error init\n");
        return;
    }

    if (mysql_query(mysql, sql)) {
        printf("error mysql_query\n");
        return;
    }

    MYSQL_RES* mysql_res = ::mysql_store_result(mysql);
    if (!mysql_res) {
        unsigned int field_cnt = ::mysql_field_count(mysql);
        if (field_cnt > 0) {
            return;
        }
    }
    if (!mysql_res) {
        return;
    }

    long long row_count = ::mysql_num_rows(mysql_res);
    /*
    if (row_count == (my_ulonglong)~0)
        return;
        */

    auto* mysql_fields = ::mysql_fetch_fields(mysql_res);
    unsigned int field_count = ::mysql_num_fields(mysql_res);

    MYSQL_TIME tm;
    std::memset(&tm, 0, sizeof(tm));
    for (long long i = 0; i != row_count; ++i) {
        MYSQL_ROW row = ::mysql_fetch_row(mysql_res);
        if (!row) {
            return;
        }
        unsigned long* lengths = ::mysql_fetch_lengths(mysql_res);
        if (!lengths) {
            printf("mysql_fetch_lengths, cannot retrieve value lengths. Error\n");// << ::mysql_error(mysql_res->handle);
            return;
        }

        for (uint32_t j = 0; j < field_count; ++j) {
            std::cout << mysql_fields[j].type << "\t"
                << "\t" << row[j]
                << "\t" << lengths[j]
                << "\n";

            if (j == 4) {
                std::memcpy(&tm, row[j], lengths[j]);
            }
        }
    }

    std::cout << "xxxxxxx\n";
    printTM(&tm);
}

int main()
{
    //funStmt();
    //std::cout << sizeof(MYSQL_TIME) << "\n";
    fun();
    return 0;
}
