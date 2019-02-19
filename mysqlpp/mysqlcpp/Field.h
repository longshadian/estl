#ifndef _MYSQLCPP_FIELD_H
#define _MYSQLCPP_FIELD_H

#include <mysql.h>
#include <cstring>

#include <vector>

#include "Types.h"

/**
    @class Field

    @brief Class used to access individual fields of database query result

    Guideline on field type matching:

    |   MySQL type           |  method to use                         |
    |------------------------|----------------------------------------|
    | TINYINT                | GetBool, GetInt8, GetUInt8             |
    | SMALLINT               | GetInt16, GetUInt16                    |
    | MEDIUMINT, INT         | GetInt32, GetUInt32                    |
    | BIGINT                 | GetInt64, GetUInt64                    |
    | FLOAT                  | GetFloat                               |
    | DOUBLE, DECIMAL        | GetDouble                              |
    | CHAR, VARCHAR,         | GetCString, GetString                  |
    | TINYTEXT, MEDIUMTEXT,  | GetCString, GetString                  |
    | TEXT, LONGTEXT         | GetCString, GetString                  |
    | TINYBLOB, MEDIUMBLOB,  | GetBinary, GetString                   |
    | BLOB, LONGBLOB         | GetBinary, GetString                   |
    | BINARY, VARBINARY      | GetBinary                              |

    Return types of aggregate functions:

    | Function |       Type        |
    |----------|-------------------|
    | MIN, MAX | Same as the field |
    | SUM, AVG | DECIMAL           |
    | COUNT    | BIGINT            |
*/

namespace mysqlcpp {

class DateTime;

class Field
{
    struct Slot
    {
        Slot();
        ~Slot();
        Slot(const Slot& rhs);
        Slot& operator=(const Slot& rhs);
        Slot(Slot&& rhs);
        Slot& operator=(Slot&& rhs);

        enum_field_types    m_type;      // Field type
        bool                m_raw;       // Raw bytes? (Prepared statement or ad hoc)
        std::vector<uint8>  m_buffer;
        uint32              m_length;    // Length (prepared strings only)
    };
public:
    Field();
    ~Field();

    Field(const Field& rhs);
    Field& operator=(const Field& rhs);

    Field(Field&& rhs);
    Field& operator=(Field&& rhs);
public:
    static uint32 sizeForType(MYSQL_FIELD* field);

    bool getBool() const;
    uint8 getUInt8() const;
    int8 getInt8() const;
    uint16 getUInt16() const;
    int16 getInt16() const;
    uint32 getUInt32() const;
    int32 getInt32() const;
    uint64 getUInt64() const;
    int64 getInt64() const;
    float getFloat() const;
    double getDouble() const;
    long double getLongDouble() const;
    //char const* getCString() const;
    std::string getString() const;
    std::vector<uint8> getBinary() const;
    bool isNull() const;
    DateTime getDateTime() const;

    void setByteValue(enum_field_types type, void* src, uint32 src_len, bool raw_bytes);

    bool isType(enum_field_types type) const;
    bool isNumeric() const;

    static char const* fieldTypeToString(enum_field_types type);
private:
    Slot        m_data;
};

}

#endif

