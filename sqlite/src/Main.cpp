
#include <string>
#include <thread>
#include <chrono>

#include "console_log.h"
#include "sqliteMgr.h"

void Test()
{
    SqliteMgr* mgr = SqliteMgr::Get();
    if (mgr->Init() != 0) {
        return;
    }

    if (0) {
        for (int i = 0; i != 10; ++i) {
            EASYLOG_INFO << "start delete: " << i;
            mgr->SqliteDb_delRecord(i);
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }

    long long num = -1;
    mgr->SqliteDb_getRecordNum(num);
    EASYLOG_INFO << "count: " << num << "\n";
}

int main(int argc, char **argv)
{
    Test();
    return 0;
}


