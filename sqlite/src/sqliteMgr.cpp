#include "sqliteMgr.h"

#include <chrono>
#include <memory>
#include <cstring>

#include<iostream>
//#include "../../algMod_localInfo.h"

#include "console_log.h"

static const char* GetSqlite3ErrorMsg(sqlite3* db)
{
    static const char n = '\0';
    const char* p = sqlite3_errmsg(db);
    return p ? p : &n;
}


static int sqlite3_select_callback(void* data, int argc, char** argv, char** azColName)
{
    int i;
    for (i = 0; i < argc; i++) {
        printf("%s = %s\n", azColName[i], argv[i] ? argv[i] : "NULL");
    }
    printf("\n");
    return 0;
}

SqliteMgr* SqliteMgr::sqlite3DbInstance = nullptr;

SqliteMgr::SqliteMgr()
{
    m_sqlite3Db = nullptr;
    umasterID_Key = 0;
    umasterID_best = 0;
}

SqliteMgr::~SqliteMgr() {}

SqliteMgr* SqliteMgr::Get()
{
    static std::mutex mutex;
    if (sqlite3DbInstance == nullptr)
    {
        std::lock_guard< std::mutex > lock{ mutex };
        if (sqlite3DbInstance == nullptr)
        {
            sqlite3DbInstance = new SqliteMgr();
        }
    }
    return sqlite3DbInstance;
}

int SqliteMgr::Init()
{
    const char* f = "C:/Users/Administrator/Desktop/test.db";
    int ret = sqlite3_open(f, &m_sqlite3Db);
    if (ret) {
        EASYLOG_WARN << "Can't open database: " << f << " reason: " << GetSqlite3ErrorMsg(m_sqlite3Db);
        return -1;
    }
    return 0;
}

#if 0
int SqliteMgr::SqliteDb_init()
{
    char* sqlite3ErrMsg = 0;
    int   iCreateTable = 0;
    int   ret;
    char  sql[512] = { 0 };
    std::string devicekey;

    ///获取设备型号
    devicekey = algMod_localInfo_getDeviceKey();
    ///创建数据库路径 
    std::string sqlite_path =  SQLITE3_DB_NAME + devicekey;
    if (0 != access(sqlite_path.c_str(), 0))
    {
        // if this folder not exist, create a new one.
        mkdir(sqlite_path.c_str(),S_IRWXU);   // 返回 0 表示创建成功，-1 表示失败
    }

    std::string sqlite_db = SQLITE3_DB_NAME + devicekey +"/qlite3_db.db";
    std::cout << __FILE__ << "  " << __FUNCTION__ << " " << __LINE__ <<"sqlitename:"<<sqlite_path.c_str()<<"############# tzw ##@@@@@@@@@@@@@@@ "<<std::endl;
    
    if (access(sqlite_db.c_str(), F_OK) != 0)
    {
        std::cout << __FILE__ << "  " << __FUNCTION__ << " " << __LINE__ <<"############# tzw ##@@@@@@@@@@@@@@@ "<<std::endl;
        iCreateTable = 1;
    }

    ret = sqlite3_open(sqlite_db.c_str(), &m_sqlite3Db);
    if (ret)
    {
        printf("SqliteMgr::SqliteDb_init； Can't open database: %s\n", sqlite3_errmsg(m_sqlite3Db));
        return -1;
    }

    if (iCreateTable)
    {  // masterID, uuID, facejpegPath, bodyjpegPath, jsonPath, timeStamp, UploadStatus
        snprintf(
            sql, 512,
            "create table IF NOT EXISTS %s (masterID int primary key not NULL, uuID text not NULL, facejpegPath text NULL, "
            "bodyjpegPath text NULL, jsonPath text NULL, timeStamp INT NOT NULL, UploadStatus "
            "INT NOT NULL);",
            SQLITE3_TB_NMAE_KEY);
        ret = sqlite3_exec(m_sqlite3Db, sql, sqlite3_select_callback, NULL, &sqlite3ErrMsg);
        if (ret != SQLITE_OK)
        {
            printf("SqliteMgr::SqliteDb_init： init  key table %s Err:\n", sqlite3ErrMsg);
            sqlite3_free(sqlite3ErrMsg);
            return -1;
        }
        // masterID, uuID, fileID, filePath, attribute, fileType, timeStamp, UploadStatus
        snprintf(
            sql, 512,
            "create table IF NOT EXISTS %s (masterID int primary key not NULL, uuID text not NULL, filePath text not NULL, "
            " attribute text not NULL, fileType INT NOT NULL, timeStamp INT NOT NULL, UploadStatus "
            "INT NOT NULL);",
            SQLITE3_TB_NMAE_BEST);
        ret = sqlite3_exec(m_sqlite3Db, sql, sqlite3_select_callback, NULL, &sqlite3ErrMsg);
        if (ret != SQLITE_OK)
        {
            printf("SqliteMgr::SqliteDb_init： init best table  Err: %s\n", sqlite3ErrMsg);
            sqlite3_free(sqlite3ErrMsg);
            return -1;
        }

        umasterID_Key = 0;
        umasterID_best = 0;
    }
    else
    {
        SqliteDb_getRecordMaxMasterID(umasterID_Key, SQLITE3_TB_NMAE_KEY);
        SqliteDb_getRecordMaxMasterID(umasterID_best, SQLITE3_TB_NMAE_BEST);
    }
    return 0;
}
#endif

int SqliteMgr::SqliteDb_clearTable(const char* tableName)
{
    std::lock_guard< std::mutex > lockGuard(m_mutex);
    char                          sql[512] = { 0 };
    char*                         sqliteErrMsg = NULL;
    int                           ret = 0;
    if (!m_sqlite3Db)
        return ret;

    snprintf(sql, 512,"delete from %s;", tableName);
    ret = sqlite3_exec(m_sqlite3Db, sql, sqlite3_select_callback, NULL, &sqliteErrMsg);
    if (ret != SQLITE_OK) {
        printf("SqliteMgr::SqliteDb_clear: delete all data Err: %s\n", sqliteErrMsg);
        sqlite3_free(sqliteErrMsg);
        return -1;
    }
    return 0;
}

int SqliteMgr::SqliteDb_setRecord(const std::shared_ptr< DB_KeyFrameData >& keyFrameData, const char* tableName)
{
    std::lock_guard< std::mutex > lockGuard(m_mutex);
    char                          sql[1024] = { 0 };
    sqlite3_stmt*                 stmt = NULL;
    // checkRecordData();
    EN_UPLOADSTATUS enUploadStatus = UPLOAD_KEYFRAME_START;
    snprintf(
        sql, 1024,
        "insert into %s (masterID, uuID, facejpegPath, bodyjpegPath, jsonPath, timeStamp, UploadStatus) "
        "values(%lld, '%s', '%s', '%s', '%s', %lld, %d);",
        tableName, ++umasterID_Key, keyFrameData->uuID.c_str(), keyFrameData->facejpegPath.c_str(), keyFrameData->bodyjpegPath.c_str(),
        keyFrameData->jsonPath.c_str(), keyFrameData->timeStamp, enUploadStatus);

    if (sqlite3_prepare(m_sqlite3Db, sql, -1, &stmt, NULL) != SQLITE_OK)
    {
        printf("SqliteMgr::SqliteDb_setRecord: sqlite3_prepare error\n");
    }
    if (sqlite3_step(stmt) != SQLITE_DONE)
    {
        printf("SqliteMgr::SqliteDb_setRecord: sqlite3_step error:%s", sqlite3_errmsg(m_sqlite3Db));
    }
    sqlite3_finalize(stmt);
    return 1;
}

int SqliteMgr::SqliteDb_setRecord(const std::vector< std::shared_ptr< DB_BestFrameData > > &bestFrameData, const char* tableName)
{
    std::lock_guard< std::mutex > lockGuard(m_mutex);
    char                          sql[1024] = { 0 };
    sqlite3_stmt*                 stmt = NULL;
    EN_UPLOADSTATUS               enUploadStatus = UPLOAD_BESTFRAME_START;
    std::cout << __FILE__ << "  " << __FUNCTION__ << " " << __LINE__ <<"############# tzw ##@@@@@@@@@@@@@@@ "<<std::endl;
    for (auto frameAuto : bestFrameData)
    {

        snprintf(
            sql, 1024,
            "insert into %s (masterID, uuID, filePath, attribute, fileType, timeStamp, UploadStatus) "
            "values(%lld, '%s', '%s', '%s', '%d', %lld, %d);",
            tableName, ++umasterID_best, frameAuto->uuID.c_str(), frameAuto->filePath.c_str(), frameAuto->attribute.c_str(), frameAuto->fileType,
            frameAuto->timeStamp, enUploadStatus);
        if (sqlite3_prepare(m_sqlite3Db, sql, -1, &stmt, NULL) != SQLITE_OK)
        {
            printf("SqliteMgr::SqliteDb_setRecord: sqlite3_prepare error\n");
        }
        if (sqlite3_step(stmt) != SQLITE_DONE)
        {
            printf("SqliteMgr::SqliteDb_setRecord: sqlite3_step error:%s", sqlite3_errmsg(m_sqlite3Db));
        }
        sqlite3_finalize(stmt);
    }

    return 1;
}

int SqliteMgr::SqliteDb_setRecord(
    const char* facejpegPath, const char* bodyjpegPath, const char* jsonPath, time_t& tt, struct timespec& ts, EN_UPLOADSTATUS enUploadStatus,
    const char* tableName)
{
    std::lock_guard< std::mutex > lockGuard(m_mutex);
    char                          sql[1024] = { 0 };
    sqlite3_stmt*                 stmt = NULL;
    if (facejpegPath == NULL || bodyjpegPath == NULL || jsonPath == NULL)
    {
        printf("qlite3Db::SqliteDb_setRecord: in param is null!\n");
        return -1;
    }
    if (m_sqlite3Db == NULL)
    {
        printf("SqliteMgr::SqliteDb_setRecord: sqlite3_open failed\n");
        return -1;
    }
    snprintf(
        sql, 1024,
        "insert into %s (masterID, facejpegPath, bodyjpegPath, jsonPath, time_t, timespec, UploadStatus) "
        "values(%lld, '%s', '%s', '%s', ?, ?, %d);",
        tableName, ++umasterID_Key, facejpegPath, bodyjpegPath, jsonPath, enUploadStatus);
    if (sqlite3_prepare(m_sqlite3Db, sql, -1, &stmt, NULL) != SQLITE_OK)
    {
        printf("SqliteMgr::SqliteDb_setRecord: sqlite3_prepare error\n");
        sqlite3_finalize(stmt);
         return -1;
    }
    if (sqlite3_bind_blob(stmt, 1, &tt, sizeof(tt), NULL) != SQLITE_OK)
    {
        printf("SqliteMgr::SqliteDb_setRecord: sqlite3_bind_blod error:%s", sqlite3_errmsg(m_sqlite3Db));
        return -1;
    }
    if (sqlite3_bind_blob(stmt, 2, &ts, sizeof(ts), NULL) != SQLITE_OK)
    {
        printf("SqliteMgr::SqliteDb_setRecord: sqlite3_bind_blod error:%s", sqlite3_errmsg(m_sqlite3Db));
        return -1;
    }

    if (sqlite3_step(stmt) != SQLITE_DONE)
    {
        printf("SqliteMgr::SqliteDb_setRecord: sqlite3_step error:%s", sqlite3_errmsg(m_sqlite3Db));
        return -1;
    }
    sqlite3_finalize(stmt);
    return 0;
}

int SqliteMgr::SqliteDb_UpdateUploadStatus(const long long masterID, const int lUploadStatus, const char* tableName)
{
    std::lock_guard< std::mutex > lockGuard(m_mutex);
    char                          sql[512] = { 0 };
    char*                         sqliteErrMsg = NULL;
    int                           ret = 0;
    if (!m_sqlite3Db)
        return ret;

    snprintf(sql, 512, "UPDATE %s SET UploadStatus = %d WHERE masterID=%lld;", tableName, lUploadStatus, masterID);
    ret = sqlite3_exec(m_sqlite3Db, sql, sqlite3_select_callback, NULL, &sqliteErrMsg);
    if (ret != SQLITE_OK)
    {
        printf("SqliteMgr::SqliteDb_clear: delete all data Err: %s\n", sqliteErrMsg);
        sqlite3_free(sqliteErrMsg);
        return -1;
    }

    return 0;
}

int SqliteMgr::SqliteDb_getRecord(DB_KeyFrameData* keyFrameData, const char* tableName)
{
    std::lock_guard< std::mutex > lockGuard(m_mutex);
    char                          sql[256] = { 0 };
    int                           iRet = -1;
    sqlite3_stmt*                 stmt = nullptr;
    char                          facejpeg[256] = { 0 };
    char                          bodyjpeg[256] = { 0 };
    char                          json[256] = { 0 };
    char                          uuID[256] = { 0 };
    if (m_sqlite3Db == nullptr)
    {
        printf("SqliteMgr::SqliteDb_getRecord: sqlite3_open failed\n");
        return -1;
    }
    std::string tbName = tableName;
    if (strcmp(tableName, SQLITE3_TB_NMAE_KEY) == 0)
    {
        snprintf(sql, 256, "SELECT * FROM %s LIMIT 1 offset (SELECT COUNT(*) - 1 FROM %s);", tableName, tableName);
        sqlite3_prepare_v2(m_sqlite3Db, sql, strlen(sql), &stmt, NULL);
        while (sqlite3_step(stmt) == SQLITE_ROW)
        {
            keyFrameData->masterID = sqlite3_column_int(stmt, 0);
            snprintf(uuID, 256, "%s", sqlite3_column_text(stmt, 1));
            snprintf(facejpeg, 256,"%s", sqlite3_column_text(stmt, 2));
            snprintf(bodyjpeg, 256,"%s", sqlite3_column_text(stmt, 3));
            snprintf(json, 256,"%s", sqlite3_column_text(stmt, 4));
            keyFrameData->timeStamp = sqlite3_column_int(stmt, 5);
            keyFrameData->lUploadStatus = sqlite3_column_int(stmt, 6);
            keyFrameData->uuID = uuID;
            keyFrameData->facejpegPath = facejpeg;
            keyFrameData->bodyjpegPath = bodyjpeg;
            keyFrameData->jsonPath = json;
            iRet = 0;
            break;
        }
        sqlite3_finalize(stmt);
    }

    return iRet;
}

int SqliteMgr::SqliteDb_getRecord(std::vector< std::shared_ptr< DB_BestFrameData > >& bestFrameData, const char* tableName)
{
    std::lock_guard< std::mutex > lockGuard(m_mutex);
    char                          sql[256] = { 0 };
    sqlite3_stmt*                 stmt = nullptr;

    if (m_sqlite3Db == nullptr)
    {
        printf("SqliteMgr::SqliteDb_getRecord: sqlite3_open failed\n");
        return -1;
    }

    snprintf(sql, 256, "SELECT * FROM %s LIMIT 1 offset (SELECT COUNT(*) - 1 FROM %s);", tableName, tableName);
    sqlite3_prepare_v2(m_sqlite3Db, sql, strlen(sql), &stmt, NULL);
    while (sqlite3_step(stmt) == SQLITE_ROW)
    {
        std::shared_ptr< DB_BestFrameData > frameDataPtr = std::make_shared< DB_BestFrameData >();
        frameDataPtr->masterID = sqlite3_column_int(stmt, 0);
        frameDataPtr->attribute = std::string((char*)sqlite3_column_text(stmt, 3));
        frameDataPtr->uuID = std::string((char*)sqlite3_column_text(stmt, 1));
        frameDataPtr->filePath = std::string((char*)sqlite3_column_text(stmt, 2));
        frameDataPtr->fileType = (EN_FILETYPE)sqlite3_column_int(stmt, 4);
        frameDataPtr->timeStamp = sqlite3_column_int(stmt, 5);
        frameDataPtr->lUploadStatus = sqlite3_column_int(stmt, 6);
        bestFrameData.push_back(frameDataPtr);
        printf(
                "SqliteDb_getRecord::sqlite3_step  uuID=%s  path %s \n", frameDataPtr->uuID.c_str(), frameDataPtr->filePath.c_str());
    }
    sqlite3_finalize(stmt);

    return 0;
}

int SqliteMgr::SqliteDb_getRecordByUUID(std::vector< std::shared_ptr< DB_BestFrameData > >& bestFrameData, const char* uuid, const char* tableName)
{
    std::lock_guard< std::mutex > lockGuard(m_mutex);
    char                          sql[256] = { 0 };
    sqlite3_stmt*                 stmt = nullptr;

    if (m_sqlite3Db == nullptr)
    {
        printf("SqliteMgr::SqliteDb_getRecordByUUID: sqlite3_open failed\n");
        return -1;
    }
    snprintf(sql, 256, "SELECT * FROM %s WHERE uuID='%s'", tableName, uuid);
    sqlite3_prepare_v2(m_sqlite3Db, sql, strlen(sql), &stmt, NULL);
    while (sqlite3_step(stmt) == SQLITE_ROW)
    {
        std::shared_ptr< DB_BestFrameData > frameDataPtr = std::make_shared< DB_BestFrameData >();
        frameDataPtr->masterID = sqlite3_column_int(stmt, 0);
        frameDataPtr->attribute = std::string((char*)sqlite3_column_text(stmt, 3));
        frameDataPtr->uuID = std::string((char*)sqlite3_column_text(stmt, 1));
        frameDataPtr->filePath = std::string((char*)sqlite3_column_text(stmt, 2));
        frameDataPtr->fileType = (EN_FILETYPE)sqlite3_column_int(stmt, 4);
        frameDataPtr->timeStamp = sqlite3_column_int(stmt, 5);
        frameDataPtr->lUploadStatus = sqlite3_column_int(stmt, 6);
        bestFrameData.push_back(frameDataPtr);
    }
    sqlite3_finalize(stmt);

    return 0;
}

int SqliteMgr::SqliteDb_getRecordFirstRecordUUID(DB_BestFrameData& bestFrameData, const char *tableName) {
    std::lock_guard< std::mutex > lockGuard(m_mutex);
    if (m_sqlite3Db == nullptr)
    {
        printf("SqliteMgr::SqliteDb_getRecordByUUID: sqlite3_open failed\n");
        return -1;
    }
    char                          sql[256] = { 0 };
    sqlite3_stmt*                 stmt = nullptr;

    if (strcmp(tableName, SQLITE3_TB_NMAE_BEST) == 0)
    {
        snprintf(sql, 256, "SELECT * FROM %s LIMIT 1 offset (SELECT COUNT(*) - 1 FROM %s);", tableName, tableName);
        sqlite3_prepare_v2(m_sqlite3Db, sql, strlen(sql), &stmt, NULL);
        while (sqlite3_step(stmt) == SQLITE_ROW)
        {
            bestFrameData.attribute = std::string((char*)sqlite3_column_text(stmt, 3));
            bestFrameData.uuID = std::string((char*)sqlite3_column_text(stmt, 1));
            bestFrameData.filePath = std::string((char*)sqlite3_column_text(stmt, 2));
            bestFrameData.fileType = (EN_FILETYPE)sqlite3_column_int(stmt, 4);
            bestFrameData.timeStamp = sqlite3_column_int(stmt, 5);
            bestFrameData.lUploadStatus = sqlite3_column_int(stmt, 6);
            break;
        }
        sqlite3_finalize(stmt);
    }

    return 0;
}

int SqliteMgr::SqliteDb_delRecord(long long masterID, const char* tableName)
{
    std::lock_guard< std::mutex > lockGuard(m_mutex);
    char                          sql[512] = { 0 };
    char*                         sqliteErrMsg = NULL;
    int                           ret = 0;
    if (!m_sqlite3Db) {
        EASYLOG_WARN << "sqlite3_open failed";
        return -1;
    }

    snprintf(sql, 512,"delete from %s where masterID=%lld;", tableName, masterID);
    ret = sqlite3_exec(m_sqlite3Db, sql, sqlite3_select_callback, NULL, &sqliteErrMsg);
    if (ret != SQLITE_OK) {
        EASYLOG_WARN << " delRecord failed " << masterID << " ret: " << ret << " reason: " <<sqliteErrMsg;
        sqlite3_free(sqliteErrMsg);
        sqliteErrMsg = NULL;
        return -1;
    }

    if (NULL != sqliteErrMsg) {
        sqlite3_free(sqliteErrMsg);
        sqliteErrMsg = NULL;
    }
    return 0;
}

int SqliteMgr::SqliteDb_delRecord(const char* uuID, const char* tableName)
{
    std::lock_guard< std::mutex > lockGuard(m_mutex);
    char                          sql[512] = { 0 };
    char*                         sqliteErrMsg = NULL;
    int                           ret = 0;
    if (m_sqlite3Db == NULL)
    {
        printf("SqliteMgr::SqliteDb_delRecord: sqlite3_open failed\n");
        return -1;
    }

    snprintf(sql, 512, "delete from %s where uuID='%s';", tableName, uuID);
    ret = sqlite3_exec(m_sqlite3Db, sql, sqlite3_select_callback, NULL, &sqliteErrMsg);
    if (ret != SQLITE_OK)
    {
        printf("SqliteMgr::SqliteDb_delRecord: delRecord failed %s\n", sqliteErrMsg);
        sqlite3_free(sqliteErrMsg);
        sqliteErrMsg = NULL;
        return -1;
    }

    if (NULL != sqliteErrMsg)
    {
        printf("SqliteMgr::SqliteDb_delRecord: free sqliteErrMsg\n");
        sqlite3_free(sqliteErrMsg);
        sqliteErrMsg = NULL;
    }
    return 0;
}

int SqliteMgr::SqliteDb_getRecordNum(long long& totalNum, const char* tableName)
{
    std::lock_guard< std::mutex > lockGuard(m_mutex);
    char                          sql[128] = { 0 };
    sqlite3_stmt*                 stmt = NULL;
    totalNum = 0;

    if (m_sqlite3Db == NULL)
    {
        printf("SqliteMgr::SqliteDb_getRecordNum: sqlite3_open failed\n");
        return -1;
    }
    snprintf(sql, 128, "select count(masterID) from %s;", tableName);
    sqlite3_prepare_v2(m_sqlite3Db, sql, strlen(sql), &stmt, NULL);

    /// 遍历执行sql语句后的结果集的每一行数据
    while (sqlite3_step(stmt) == SQLITE_ROW)
    {
        totalNum = sqlite3_column_int(stmt, 0);
        break;
    }
    sqlite3_finalize(stmt);
    return 0;
}

#if 0
int SqliteMgr::SqliteDb_getRecordMaxMasterID(long long& masterID, const char* tableName)
{
    std::lock_guard< std::mutex > lockGuard(m_mutex);
    int                           iRet = 0;
    char                          sql[256] = { 0 };
    // int           iRet = -1;  // time_t tt; struct timespec ts;
    sqlite3_stmt* stmt = NULL;
    if (m_sqlite3Db == NULL)
    {
        printf("SqliteMgr::SqliteDb_getRecord: sqlite3_open failed\n");
        return -1;
    }
    snprintf(sql, 256, "SELECT * FROM %s LIMIT 1 offset (SELECT COUNT(*) - 1 FROM %s);", tableName, tableName);
    sqlite3_prepare_v2(m_sqlite3Db, sql, strlen(sql), &stmt, NULL);
    while (sqlite3_step(stmt) == SQLITE_ROW)
    {
        masterID = sqlite3_column_int(stmt, 0);
        iRet = 0;
        break;
    }
    sqlite3_finalize(stmt);

    if (iRet != 0)
    {
        masterID = 0;
    }
    return 0;
}
#endif

