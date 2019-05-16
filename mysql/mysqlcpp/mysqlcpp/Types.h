#pragma once

#include <memory>
#include <vector>

namespace mysqlcpp {

#ifndef MYSQLCPP_EXPORT
#define MYSQLCPP_EXPORT
#endif

class Statement;
class PreparedStatement;
class ResultSet;
class FieldMeta;
class Field;

using StatementPtr = std::shared_ptr<Statement>;
using PreparedStatementPtr = std::shared_ptr<PreparedStatement>;
using ResultSetPtr = std::shared_ptr<ResultSet>;

using RowData = std::vector<Field>;
using FieldData = std::vector<FieldMeta>;

using uint8	 = uint8_t;
using uint16 = uint16_t;
using uint32 = uint32_t;
using uint64 = uint64_t;

using int8	= int8_t;
using int16	= int16_t;
using int32	= int32_t;
using int64	= int64_t;

}
