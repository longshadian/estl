#include <mysql.h>

#include <cstdio>
#include <cstring>

#include <iostream>
#include <string>
#include <vector>

void fun() {
	#define INSERT_QUERY "INSERT INTO `utf8mb4` (`name`) VALUES (?); "

	std::string s = {};
	// "xF0 x9F x98 x89"
	std::vector<uint8_t> b{};
	s.push_back(0xF0);
	s.push_back(0x9F);
	s.push_back(0x98);
	s.push_back(0xA4);
	b.assign(s.begin(), s.end());

	MYSQL mysql;
	mysql_init(&mysql);
	//mysql_options(&mysql, MYSQL_READ_DEFAULT_GROUP, "your_prog_name");
	if (!mysql_real_connect(&mysql, "127.0.0.1", "root", "123456", "test_utf8", 0, NULL, 0))
	{
		fprintf(stderr, "Failed to connect to database: Error: %s\n",
			mysql_error(&mysql));
		return;
	}

	MYSQL_BIND bind[1];
	unsigned long length;
	MYSQL_STMT * stmt;
	stmt = mysql_stmt_init(&mysql);
	if (!stmt)
	{
		fprintf(stderr, " mysql_stmt_init(), out of memory\n");
		exit(0);
	}
	if (mysql_stmt_prepare(stmt, INSERT_QUERY, strlen(INSERT_QUERY)))
	{
		fprintf(stderr, "\n mysql_stmt_prepare(), INSERT failed");
		fprintf(stderr, "\n %s", mysql_stmt_error(stmt));
		exit(0);
	}
	memset(bind, 0, sizeof(bind));
	bind[0].buffer_type = MYSQL_TYPE_VAR_STRING;
	bind[0].buffer = (void*)s.c_str();
	bind[0].buffer_length = s.size();
	//bind[0].length = &length;
	//bind[0].is_null = 0;
	/* Bind the buffers */
	if (mysql_stmt_bind_param(stmt, bind))
	{
		fprintf(stderr, "\n param bind failed");
		fprintf(stderr, "\n %s", mysql_stmt_error(stmt));
		exit(0);
	}

	/* Supply data in chunks to server */
	/*
	if (mysql_stmt_send_long_data(stmt, 0, s.c_str(), s.size()))
	{
		fprintf(stderr, "\n send_long_data failed");
		fprintf(stderr, "\n %s", mysql_stmt_error(stmt));
		exit(0);
	}
	*/

	/* Now, execute the query */
	if (mysql_stmt_execute(stmt))
	{
		fprintf(stderr, "\n mysql_stmt_execute failed");
		fprintf(stderr, "\n %s", mysql_stmt_error(stmt));
		exit(0);
	}

}

int main()
{
	fun();
}
