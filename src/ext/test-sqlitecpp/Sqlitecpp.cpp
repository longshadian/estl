#include <string>
#include <iostream>

#include <SQLiteCpp/SQLiteCpp.h>

#include "console_log.h"

class SqlMgr
{
public:
    SqlMgr()
        : db_()
    {
    }

    ~SqlMgr()
    {

    }

    bool Init(std::string db_path)
    {
        try {
            db_ = std::make_shared<SQLite::Database>(db_path, SQLite::OPEN_READWRITE);
            return true;
        } catch (const std::exception& e) {
            CONSOLE_LOG_WARN << "exception: " << e.what();
            return false;
        }
    }

    SQLite::Database* GetDB()
    {
        return db_.get();
    }

    std::shared_ptr<SQLite::Database> db_;
};


bool TestInsert(SqlMgr* mgr)
{
    try {
        std::string sql = 
            "INSERT INTO camerasEx( "
            "_ID, CAMARA_NAME, ENABLE "
            " ) VALUES( "
            " ?, ?, ?" 
            " ); "
        ;

        int i = 0;
        SQLite::Statement insert(*mgr->GetDB(), sql);
        insert.bind(1, 111);
        insert.bind(2, "camera_name");
        insert.bind(3, 1);
        insert.exec();
        return true;
    } catch (const std::exception& e) {
        CONSOLE_LOG_WARN << "exception: " << e.what();
        return false;
    }
}

bool TestQuery(SqlMgr* mgr)
{
    try {
        std::string sql = 
            "select * from camerasEx where CAMERA_NAME = ?";
        ;

        int idx = 0;
        SQLite::Statement query(*mgr->GetDB(), sql);
        query.bind(1, "camera_name");
        while (query.executeStep()) {
            ++idx;
            int id = query.getColumn(0);
            std::string name = query.getColumn(1);
            std::string code = query.getColumn(2);

            CONSOLE_LOG_INFO << "idx: " << idx << " " << id << " " << name << " " << code;
        }
        return true;
    } catch (const std::exception& e) {
        CONSOLE_LOG_WARN << "exception: " << e.what();
        return false;
    }
}

bool TestUpdate(SqlMgr* mgr)
{
    try {
        return true;
    } catch (const std::exception& e) {
        CONSOLE_LOG_WARN << "exception: " << e.what();
        return false;
    }
}

void Test()
{
    SqlMgr mgr{};
    assert(mgr.Init("C:/Users/admin/Desktop/QY_ctr.db"));
    //assert(TestInsert(&mgr));
    assert(TestQuery(&mgr));
}

int main()
{
    Test();
    system("pause");
    return 0;
}

