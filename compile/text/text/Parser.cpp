#include "Parser.h"
#include "Lexer.h"

const std::string SENTINEL_SUFFIX = ".generate.h";
const std::string UCLASS = "UCLASS";
const std::string UFUNCTION = "UFUNCTION";
const std::string UPROPERTY = "UPROPERTY";


idParser::idParser(idLexer* lexer)
    : m_lexer(lexer)
    , m_token()
    , m_indent(Indent::Normal)
    , m_sentinel_name()
{
    m_sentinel_name = lexer->FileName() + SENTINEL_SUFFIX;
}

idParser::~idParser()
{
}

bool idParser::Start()
{
    bool start_parse = false;
    while (true) {
        if (!GetNextToken()) {
            CommonError("get token error");
            return false;
        }
        if (Token()->AsStringView() == "#" && ParseGenerateSentinel()) {
            // 找到UHT哨兵, 开始解析头文件
            start_parse = true;
            break;
        }
    }

    if (start_parse) {
        StartParse();
    }
}

void idParser::CommonError(std::string_view str)
{

}

bool idParser::GetNextToken()
{
    if (!m_lexer->GetToken(&m_token))
        return false;
    return true;
}

const idToken* idParser::Token() const
{
    return &m_token;
}

void idParser::ResetIndent()
{
    m_indent = Indent::Normal;
}

bool idParser::IndentForClassStruct() const
{
    return m_indent == Indent::Class 
        || m_indent == Indent::Struct;
}

bool idParser::ParseGenerateSentinel()
{
    GetNextToken();
    if (Token()->AsStringView() != "\"") {
        return false;
    }
    GetNextToken();
    if (Token()->AsStringView() != m_sentinel_name) {
        return false;
    }
    GetNextToken();
    if (Token()->AsStringView() != "\"") {
        return false;
    }
    return true;
}

bool idParser::StartParse()
{
    while (true) {
        if (!GetNextToken()) {
            CommonError("get token error");
            return false;
        }
        auto str = Token()->AsTypeStringView();
        if (str == UCLASS) {
            GetNextToken();
            if (Token()->AsStringView() == "(") {
                ParseUClass();
            }
        } else if (str == UFUNCTION && IndentForClassStruct()) {
            GetNextToken();
            if (Token()->AsStringView() == "(") {
                ParseUFunction();
            }
        } else if (str == UPROPERTY && IndentForClassStruct()) {
            GetNextToken();
            if (Token()->AsStringView() == "(") {
                ParseUProperty();
            }
        }
    }
}

bool idParser::ParseUClass()
{
    // TODO
    return true;
}

bool idParser::ParseUFunction()
{
    // TODO
    return true;
}

bool idParser::ParseUProperty()
{
    // TODO
    return true;
}

std::shared_ptr<GenerateResult> idParser::ParseArgs()
{
    // TODO
    return nullptr;
}
