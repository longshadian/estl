

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <thread>
#include <chrono>
#include <iostream>
#include <string>

#include <zookeeper.h>
#include <zookeeper_log.h>

void QueryServer_watcher_g(zhandle_t* zh, int type, int state, const char* path, void* WATCHER_CTX)
{
	if (type == ZOO_SESSION_EVENT) {
		if (state == ZOO_CONNECTED_STATE) {
			std::cout << "connected to zookeeper service successfully\n";
		} else if (state == ZOO_EXPIRED_SESSION_STATE) {
			std::cout << "connected to zookeeper service expired\n";
		}
	}
}

void QueryServer_string_completion(int rc, const char* name, const void* data)
{
	fprintf(stderr, "[%s]: rc = %d\n", (char*)(data == 0 ? "null" : data), rc);
	if (!rc) {
		fprintf(stderr, "\tname = %s\n", name);
	}
}

void QueryServer_accept_query()
{
	printf("QueryServer is running...\n");
}

int main(int argc, const char *argv[])
{
	const char* host = "127.0.0.1:2181,127.0.0.1:2182,"
		"127.0.0.1:2183,127.0.0.1:2184,127.0.0.1:2185";
	int timeout = 30000;

	std::string data = "hello zookeeper";

	zoo_set_debug_level(ZOO_LOG_LEVEL_WARN);
	zhandle_t* zkhandle = zookeeper_init(host,
		QueryServer_watcher_g, timeout, 0, &data[0], 0);
	if (zkhandle == NULL) {
		fprintf(stderr, "Error when connecting to zookeeper servers...\n");
		exit(EXIT_FAILURE);
	}

	// struct ACL ALL_ACL[] = {{ZOO_PERM_ALL, ZOO_ANYONE_ID_UNSAFE}};
	// struct ACL_vector ALL_PERMS = {1, ALL_ACL};
	int ret = zoo_acreate(zkhandle, "/QueryServer", "alive", 5,
		&ZOO_OPEN_ACL_UNSAFE, ZOO_EPHEMERAL,
		QueryServer_string_completion, "zoo_acreate");
	if (ret) {
		fprintf(stderr, "Error %d for %s\n", ret, "acreate");
		exit(EXIT_FAILURE);
	}

	do {
		// ģ�� QueryServer �����ṩ����.
		// Ϊ�˼����, �����ڴ˵���һ���򵥵ĺ�����ģ�� QueryServer.
		// Ȼ������ 5 �룬���������˳�(�������ʱ�Ѿ�����).
		QueryServer_accept_query();
		std::this_thread::sleep_for(std::chrono::seconds{5});
	} while (false);
	zookeeper_close(zkhandle);
	return 0;
}

