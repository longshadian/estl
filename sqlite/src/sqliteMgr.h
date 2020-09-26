#ifndef _SQLITE_MGR_HH_
#define _SQLITE_MGR_HH_

#include <memory>
#include <vector>
#include <mutex>

#include <sqlite3.h>

#define SQLITE3_DB_NAME "/tmp/db/"
#define SQLITE3_TB_NMAE_KEY "uniubi_table_key"
#define SQLITE3_TB_NMAE_BEST "uniubi_table_best"
#define SQLITE3_RECORD_MOUNTED_DIR "/tmp/db/"

/// 50MB，最小存储大小（低于150M时，不写文件）
#define  DATA_PATH_MIN_SIZE_STOP     (300*1024*1024)


/**
 * 数据上传状态机
 */
typedef enum en_DataUploadStatus
{
    // UPLOAD_NULL = 0,
    // UPLOAD_OSS_START,   /*数据开始上传到OSS*/
    // REPORT_PLAT_START,  /*数据开始上报到平台*/
    // UPLOAD_OSS_RESTART, /*数据重传到OSS*/
    // REPORT_PLAT_RESTART,    /*数据重新上报到平台*/

    UPLOAD_KEYFRAME_START,
    UPLOAD_BESTFRAME_START,
    UPLOAD_KEYFRAME_RESATRT,
    UPLOAD_BESTFRAME_RESTART,
    UPLOAD_END = 0xFF,
}EN_UPLOADSTATUS;

typedef enum en_DBFileType
{
    PIC_FILE_TYPE,
    JSON_FILE_TYPE,
}EN_FILETYPE;

typedef struct DB_KeyFrameData
{
    long long masterID;
    std::string uuID;
    std::string facejpegPath;
    std::string bodyjpegPath;
    std::string jsonPath;
    long long timeStamp;
    int lUploadStatus;
}DB_KEYRECORD_DATA;

typedef struct DB_BestFrameData
{
    long long masterID;
    std::string uuID;
    std::string filePath;
    std::string attribute;
    EN_FILETYPE fileType;
    long long timeStamp;
    int lUploadStatus;
}DB_BESTRECORD_DATA;

class SqliteMgr
{
public:
	static SqliteMgr *Get();
    ~SqliteMgr();

    int Init();

	int SqliteDb_init();
	int SqliteDb_clearTable(const char *tableName = SQLITE3_TB_NMAE_KEY);
	int SqliteDb_delRecord(long long masterID, const char *tableName = SQLITE3_TB_NMAE_BEST);
    int SqliteDb_delRecord(const char* uuID, const char* tableName = SQLITE3_TB_NMAE_BEST);
	int SqliteDb_getRecordNum(long long &totalNum, const char *tableName = SQLITE3_TB_NMAE_KEY);
    int SqliteDb_UpdateUploadStatus(const long long masterID, const int lUploadStatus, const char *tableName = SQLITE3_TB_NMAE_KEY);

    int SqliteDb_getRecord(DB_KeyFrameData* keyFrameData, const char* tableName = SQLITE3_TB_NMAE_KEY);
    int SqliteDb_getRecord(std::vector<std::shared_ptr<DB_BestFrameData>> &bestFrameData, const char* tableName = SQLITE3_TB_NMAE_BEST);
    int SqliteDb_setRecord(const std::shared_ptr<DB_KeyFrameData>& keyFrameData, const char* tableName = SQLITE3_TB_NMAE_KEY);
    int SqliteDb_setRecord(const std::vector<std::shared_ptr<DB_BestFrameData>>& bestFrameData, const char* tableName = SQLITE3_TB_NMAE_BEST);
    int SqliteDb_getRecordByUUID(std::vector<std::shared_ptr<DB_BestFrameData>>& bestFrameData, const char* uuid, const char* tableName = SQLITE3_TB_NMAE_BEST);
    int SqliteDb_getRecordFirstRecordUUID(DB_BestFrameData& bestFrameData, const char* tableName = SQLITE3_TB_NMAE_BEST);
private:
    SqliteMgr();

	int SqliteDb_getRecord(long long &masterID, char *facejpegPath, char *bodyjpegPath, char *jsonPath, const char *tableName = SQLITE3_TB_NMAE_KEY);
    int SqliteDb_setRecord(const char *facejpegPath, const char *bodyjpegPath, const char *jsonPath, time_t &tt,
                                                struct timespec &ts, EN_UPLOADSTATUS enUploadStatus, const char *tableName = SQLITE3_TB_NMAE_KEY);

#if 0
	int SqliteDb_getRecordMaxMasterID(long long &masterID, const char *tableName = SQLITE3_TB_NMAE_KEY);
#endif

	static SqliteMgr *sqlite3DbInstance;
	sqlite3 *m_sqlite3Db;

	long long umasterID_Key;
	long long umasterID_best;

	std::mutex m_mutex;
};

#endif
