
#include <cstdlib>
#include <cstring>
#include <memory>
#include <chrono>

#include <mysql.h>

std::string db_name = "mj_game";
std::string db_ip   = "192.168.0.123";
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

void fun()
{
    MYSQL* mysql = init();
    if (!mysql) {
        printf("error\n");
        return;
    }

    const std::string sql_1 = 
        "SELECT MAX(user_id) AS max_user_id, MIN(user_id) as min_user_id FROM user_basic";

    if (::mysql_real_query(mysql, sql_1.c_str(), sql_1.size()) != 0) {
        printf("mysql_real_query fail\n");
        return;
    }

    MYSQL_RES* mysql_res = ::mysql_store_result(mysql);
    if (!mysql_res) {
        printf("mysql_store_result fail\n");
        return;
    }
    MYSQL_ROW row = ::mysql_fetch_row(mysql_res);
    printf("%s %s\n", row[0], row[1]);
    ::mysql_free_result(mysql_res);

    printf("==============\n");

    std::string sql_2 = "SELECT `user_id`, `user_id` from `user_basic` WHERE "
        " ? <= `user_id` AND `user_id` < ? ";
    MYSQL_STMT* stmt = ::mysql_stmt_init(mysql);
    if (!stmt) {
        fprintf(stderr, " mysql_stmt_init(), out of memory\n");
        return;
    }
    if (::mysql_stmt_prepare(stmt, sql_2.c_str(), sql_2.size())) {
        fprintf(stderr, " mysql_stmt_prepare(), SELECT failed\n");
        fprintf(stderr, " %s\n", mysql_stmt_error(stmt));
        return;
    }

    int min_id = 111806;
    int max_id = 112201;
    MYSQL_BIND bind_params[2];
    std::memset(bind_params, 0, sizeof(bind_params));
    bind_params[0].buffer_type = MYSQL_TYPE_LONG;
    bind_params[0].buffer = (char*)&min_id;
    bind_params[0].is_null = 0;
    bind_params[0].length = 0;

    bind_params[1].buffer_type = MYSQL_TYPE_LONG;
    bind_params[1].buffer = (char*)&max_id;
    bind_params[1].is_null = 0;
    bind_params[1].length = 0;

    int n = 2;
    while (n > 0) {
        --n;
        if (::mysql_stmt_bind_param(stmt, bind_params)) {
            printf("bind params fail\n");
            return;
        }

        if (mysql_stmt_execute(stmt)) {
            fprintf(stderr, " mysql_stmt_execute(), failed\n");
            fprintf(stderr, " %s\n", mysql_stmt_error(stmt));
            return;
        }

        MYSQL_RES* mysql_res = ::mysql_stmt_result_metadata(stmt);
        auto row_count = ::mysql_num_rows(mysql_res);
        if (row_count == (my_ulonglong)~0)
            return;
        MYSQL_FIELD* mysql_fields = ::mysql_fetch_fields(mysql_res);
        auto field_count = ::mysql_num_fields(mysql_res);

        unsigned long length[2];
        my_bool is_null[2];
        my_bool error[2];
        int int_data[2];
        MYSQL_BIND bind[2];

        std::memset(bind, 0, sizeof(bind));
        memset(&int_data, 0, sizeof(int_data));
        bind[0].buffer_type = MYSQL_TYPE_LONG;
        bind[0].buffer = (char *)&int_data[0];
        bind[0].is_null = &is_null[0];
        bind[0].length = &length[0];
        bind[0].error = &error[0];

        bind[1].buffer_type = MYSQL_TYPE_LONG;
        bind[1].buffer = (char *)&int_data[1];
        bind[1].is_null = &is_null[1];
        bind[1].length = &length[1];
        bind[1].error = &error[1];

        if (::mysql_stmt_bind_result(stmt, bind)) {
            fprintf(stderr, " mysql_stmt_bind_result() failed\n");
            fprintf(stderr, " %s\n", mysql_stmt_error(stmt));
            return;
        }

        if (mysql_stmt_store_result(stmt)) {
            fprintf(stderr, " mysql_stmt_store_result() failed\n");
            fprintf(stderr, " %s\n", mysql_stmt_error(stmt));
            return;
        }

        while (!mysql_stmt_fetch(stmt)) {
            //fprintf(stdout, " %d %d\n", int_data[0], int_data[1]);
        }

        ::mysql_free_result(mysql_res);
    }

    /* Close the statement */
    if (mysql_stmt_close(stmt)) {
        fprintf(stderr, " failed while closing the statement\n");
        fprintf(stderr, " %s\n", mysql_stmt_error(stmt));
        return;
    }

}

void fun2()
{
    const uint32_t STRING_SIZE = 50;
    const std::string SELECT_SAMPLE = "SELECT fid, fipbegin, fipend, flocation FROM tiptable";
    //const std::string SELECT_SAMPLE = "SELECT fid, fipbegin, fipend FROM tiptable";

    MYSQL* mysql = init();
    if (!mysql) {
        printf("error\n");
        return;
    }

    /* Prepare a SELECT query to fetch data from test_table */
    MYSQL_STMT* stmt = mysql_stmt_init(mysql);
    if (!stmt) {
        fprintf(stderr, " mysql_stmt_init(), out of memory\n");
        exit(0);
    }

    /*
    my_bool bool_tmp = 1;
    ::mysql_stmt_attr_set(stmt, STMT_ATTR_UPDATE_MAX_LENGTH, &bool_tmp);
    */

    if (mysql_stmt_prepare(stmt, SELECT_SAMPLE.c_str(), SELECT_SAMPLE.size())) {
        fprintf(stderr, " mysql_stmt_prepare(), SELECT failed\n");
        fprintf(stderr, " %s\n", mysql_stmt_error(stmt));
        exit(0);
    }

    fprintf(stdout, " prepare, SELECT successful\n");

    /* Get the parameter count from the statement */
    int param_count = (int)mysql_stmt_param_count(stmt);

    fprintf(stdout, " total parameters in SELECT: %d\n", param_count);
    if (param_count != 0) {
        fprintf(stderr, " invalid parameter count returned by MySQL\n");
        exit(0);
    }

    /* Fetch result set meta information */
    MYSQL_RES* prepare_meta_result = mysql_stmt_result_metadata(stmt);
    if (!prepare_meta_result) {
        fprintf(stderr, " mysql_stmt_result_metadata(),  returned no meta information\n");
        fprintf(stderr, " %s\n", mysql_stmt_error(stmt));
        exit(0);
    }

    /* Execute the SELECT query */
    if (mysql_stmt_execute(stmt)) {
        fprintf(stderr, " mysql_stmt_execute(), failed\n");
        fprintf(stderr, " %s\n", mysql_stmt_error(stmt));
        exit(0);
    }

    auto tbegin = std::chrono::system_clock::now();
    if (mysql_stmt_store_result(stmt)) {
        fprintf(stderr, " mysql_stmt_store_result() failed\n");
        fprintf(stderr, " %s\n", mysql_stmt_error(stmt));
        exit(0);
    }
    auto tend = std::chrono::system_clock::now();
    fprintf(stderr, "mysql_stmt_store_result use %d\n", (int)std::chrono::duration_cast<std::chrono::milliseconds>(tend-tbegin).count());


    /* Get total columns in the query */
    int column_count = mysql_num_fields(prepare_meta_result);
    fprintf(stdout, " total columns in SELECT statement: %d\n", column_count);
    /*
    if (column_count != 4) {
        fprintf(stderr, " invalid column count returned by MySQL\n");
        exit(0);
    }
    */

    int column_count2 = (int)::mysql_stmt_field_count(stmt);
    fprintf(stdout, "field count %d %d\n", column_count, column_count2);


    MYSQL_FIELD* all_fields = ::mysql_fetch_fields(prepare_meta_result);

    //bool has_string = false;
    for (int i = 0; i != column_count; ++i) {
        const MYSQL_FIELD& f = all_fields[i];
        fprintf(stdout, "name:%s type:%d max_len:%d\n", f.name, f.type, (int)f.max_length);
        /*
        if (!has_string) {
            has_string = f.type == MYSQL_TYPE_TINY_BLOB
                || f.type == MYSQL_TYPE_MEDIUM_BLOB
                || f.type == MYSQL_TYPE_LONG_BLOB
                || f.type == MYSQL_TYPE_BLOB
                || f.type == MYSQL_TYPE_STRING
                || f.type == MYSQL_TYPE_VAR_STRING
        }
        */
    }

    return;

    /* Bind the result buffers for all 4 columns before fetching them */

    unsigned long length[4];
    my_bool is_null[4];
    my_bool error[4];
    short small_data;
    (void)small_data;
    int int_data[4];
    char str_data[4][STRING_SIZE];
    MYSQL_BIND bind[4];

    std::memset(bind, 0, sizeof(bind));
    /* INTEGER COLUMN */
    bind[0].buffer_type = MYSQL_TYPE_LONG;
    bind[0].buffer = (char *)&int_data[0];
    bind[0].is_null = &is_null[0];
    bind[0].length = &length[0];
    bind[0].error = &error[0];

    bind[1].buffer_type = MYSQL_TYPE_LONG;
    bind[1].buffer = (char *)&int_data[1];
    bind[1].is_null = &is_null[1];
    bind[1].length = &length[1];
    bind[1].error = &error[1];

    bind[2].buffer_type = MYSQL_TYPE_LONG;
    bind[2].buffer = (char *)&int_data[2];
    bind[2].is_null = &is_null[2];
    bind[2].length = &length[2];
    bind[2].error = &error[2];

    bind[3].buffer_type = MYSQL_TYPE_STRING;
    bind[3].buffer = (char *)&str_data[3];
    bind[3].buffer_length = STRING_SIZE;
    bind[3].is_null = &is_null[3];
    bind[3].length = &length[3];
    bind[3].error = &error[3];


    /* TIMESTAMP COLUMN */
    /*
    bind[3].buffer_type = MYSQL_TYPE_TIMESTAMP;
    bind[3].buffer = (char *)&ts;
    bind[3].is_null = &is_null[3];
    bind[3].length = &length[3];
    bind[3].error = &error[3];
    */

    /* Bind the result buffers */
    if (mysql_stmt_bind_result(stmt, bind)) {
        fprintf(stderr, " mysql_stmt_bind_result() failed\n");
        fprintf(stderr, " %s\n", mysql_stmt_error(stmt));
        exit(0);
    }

    /* Now buffer all results to client (optional step) */

    /*
    auto tbegin = std::chrono::system_clock::now();
    if (mysql_stmt_store_result(stmt)) {
        fprintf(stderr, " mysql_stmt_store_result() failed\n");
        fprintf(stderr, " %s\n", mysql_stmt_error(stmt));
        exit(0);
    }
    auto tend = std::chrono::system_clock::now();
    fprintf(stderr, "mysql_stmt_store_result use %d\n", (int)std::chrono::duration_cast<std::chrono::seconds>(tend-tbegin).count());
    */

    fprintf(stderr, "total rows:%d\n",(int)::mysql_stmt_num_rows(stmt));

    /* Fetch all rows */
    int row_count = 0;
    fprintf(stdout, "Fetching results ...\n");
    while (!mysql_stmt_fetch(stmt)) {
        row_count++;
        fprintf(stdout, " row %d\n", row_count);

        /* column 1 */
        fprintf(stdout, " column1: ");
        if (is_null[0])
            fprintf(stdout, " NULL\t");
        else
            fprintf(stdout, " %d(%ld)\t", int_data[0], length[0]);

        /* column 2 */
        fprintf(stdout, " column2: ");
        if (is_null[1])
            fprintf(stdout, " NULL\t");
        else
            fprintf(stdout, " %d(%ld)\t", int_data[1], length[1]);

        /* column 3 */
        fprintf(stdout, " column3: ");
        if (is_null[2])
            fprintf(stdout, " NULL\t");
        else
            fprintf(stdout, " %d(%ld)\t", int_data[2], length[2]);

        fprintf(stdout, " column4: ");
        if (is_null[3])
            fprintf(stdout, " NULL\t");
        else
            fprintf(stdout, " %s(%ld)\t", str_data[3], length[3]);

        /* column 4 */
        /*
        fprintf(stdout, " column4 (timestamp): ");
        if (is_null[3])
            fprintf(stdout, " NULL\n");
        else
            fprintf(stdout, " %04d-%02d-%02d %02d:%02d:%02d (%ld)\n",
                ts.year, ts.month, ts.day,
                ts.hour, ts.minute, ts.second,
                length[3]);
                */
        fprintf(stdout, "\n");
        if (row_count > 10)
            break;
    }

    /* Validate rows fetched */
    fprintf(stdout, " total rows fetched: %d\n", row_count);

    /* Free the prepared result metadata */
    mysql_free_result(prepare_meta_result);

    /* Close the statement */
    if (mysql_stmt_close(stmt)) {
        fprintf(stderr, " failed while closing the statement\n");
        fprintf(stderr, " %s\n", mysql_stmt_error(stmt));
        exit(0);
    }
}


int main()
{
    fun();

    return 0;
}
