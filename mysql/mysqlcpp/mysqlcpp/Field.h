#pragma once

#include <cstring>
#include <vector>
#include <string_view>
#include <string>

#include <mysql.h>

#include "mysqlcpp/Types.h"
#include "mysqlcpp/detail/SafeString.h"

/**
	MYSQL查询结果对应类型
    |------------------------|----------------------------|
    | TINYINT                | getBool, getInt8, getUInt8 |
    | SMALLINT               | getInt16, getUInt16        |
    | MEDIUMINT, INT         | getInt32, getUInt32        |
    | BIGINT                 | getInt64, getUInt64        |
    | FLOAT                  | getFloat                   |
    | DOUBLE, DECIMAL        | getDouble                  |
    | CHAR, VARCHAR,         | getString                  |
    | TINYTEXT, MEDIUMTEXT,  | getString                  |
    | TEXT, LONGTEXT         | getString                  |
    | TINYBLOB, MEDIUMBLOB,  | getBinary, getString       |
    | BLOB, LONGBLOB         | getBinary, getString       |
    | BINARY, VARBINARY      | getBinary                  |
    | DATE, TIME			 | getDateTime()			  |
	| DATETIME, TIMESTAMP    | getDateTime()              |

	聚合函数返回值：
    |----------|------------|
    | MIN, MAX | 和Field类似 |
    | SUM, AVG | getDouble  |
    | COUNT    | getInt64   |
*/

namespace mysqlcpp {

class DateTime;

class MYSQLCPP_EXPORT Field
{
public:
    Field();
    ~Field();

    Field(const Field& rhs);
    Field& operator=(const Field& rhs);

    Field(Field&& rhs);
    Field& operator=(Field&& rhs);
public:

    bool                getBool() const;
    uint8               getUInt8() const;
    int8                getInt8() const;
    uint16              getUInt16() const;
    int16               getInt16() const;
    uint32              getUInt32() const;
    int32               getInt32() const;
    uint64              getUInt64() const;
    int64               getInt64() const;
    float               getFloat() const;
    double              getDouble() const;
    long double         getLongDouble() const;
    std::string_view    getStringView() const;
    std::string         getString() const;
    std::vector<uint8>  getBinary() const;
    bool                isNull() const;
    DateTime            getDateTime() const;

    void                setBinaryValue(enum_field_types type, void* src, unsigned long src_len, bool raw_bytes);
    void                setNullValue(enum_field_types type);

    const detail::SafeString& getInternalBuffer() const { return m_buffer; }
private:
    enum_field_types    m_type;
    detail::SafeString  m_buffer;
    bool                m_is_binary;
    bool                m_is_null;
};

}
