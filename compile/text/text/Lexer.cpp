#include "Lexer.h"

#include <utility>

#include "File.h"
#include "Log.h"

const std::string g_keyword[] =
{
    "asm",
    "do",
    "if",
    "return",
    "auto",
    "double",
    "inline",
    "short",
    "bool",
    "dynamic_cast",
    "int",
    "signed",
    "break",
    "else",
    "long",
    "sizeof",
    "case",
    "enum",
    "mutable",
    "static",
    "catch",
    "explicit",
    "namespace",
    "static_cast",
    "char",
    "export",
    "new",
    "struct",
    "class",
    "extern",
    "operator",
    "switch",
    "const",
    "false",
    "private",
    "template",
    "const_cast",
    "float",
    "protected",
    "this",
    "continue",
    "for",
    "public",
    "throw",
    "default",
    "friend",
    "register",
    "true",
    "delete",
    "goto",
    "reinterpret_cast",
    "constexpr",
    "decltype",
    "noexcept",
    "nullptr",
    "static_assert",
};

//longer punctuations first
const
Punctuation g_default_punctuations[] = 
{
	//binary operators
	{">>=",P_RSHIFT_ASSIGN},
	{"<<=",P_LSHIFT_ASSIGN},
	//
	{"...",P_PARMS},
	//define merge operator
    {"#", P_MACRO},
	{"##",P_PRECOMPMERGE},				// pre-compiler
	//logic operators
	{"&&",P_LOGIC_AND},					// pre-compiler
	{"||",P_LOGIC_OR},					// pre-compiler
	{">=",P_LOGIC_GEQ},					// pre-compiler
	{"<=",P_LOGIC_LEQ},					// pre-compiler
	{"==",P_LOGIC_EQ},					// pre-compiler
	{"!=",P_LOGIC_UNEQ},				// pre-compiler
	//arithmatic operators
	{"*=",P_MUL_ASSIGN},
	{"/=",P_DIV_ASSIGN},
	{"%=",P_MOD_ASSIGN},
	{"+=",P_ADD_ASSIGN},
	{"-=",P_SUB_ASSIGN},
	{"++",P_INC},
	{"--",P_DEC},
	//binary operators
	{"&=",P_BIN_AND_ASSIGN},
	{"|=",P_BIN_OR_ASSIGN},
	{"^=",P_BIN_XOR_ASSIGN},
	{">>",P_RSHIFT},					// pre-compiler
	{"<<",P_LSHIFT},					// pre-compiler
	//reference operators
	{"->",P_POINTERREF},
	//C++
	{"::",P_NAMESPACE},
	{".*",P_REF_MUL},
	//arithmatic operators
	{"*",P_MUL},						// pre-compiler
	{"/",P_DIV},						// pre-compiler
	{"%",P_MOD},						// pre-compiler
	{"+",P_ADD},						// pre-compiler
	{"-",P_SUB},						// pre-compiler
	{"=",P_ASSIGN},
	//binary operators
	{"&",P_BIN_AND},					// pre-compiler
	{"|",P_BIN_OR},						// pre-compiler
	{"^",P_BIN_XOR},					// pre-compiler
	{"~",P_BIN_NOT},					// pre-compiler
	//logic operators
	{"!",P_LOGIC_NOT},					// pre-compiler
	{">",P_LOGIC_GREATER},				// pre-compiler
	{"<",P_LOGIC_LESS},					// pre-compiler
	//reference operator
	{".",P_REF},
	//seperators
	{",",P_COMMA},						// pre-compiler
	{";",P_SEMICOLON},
	//label indication
	{":",P_COLON},						// pre-compiler
	//if statement
	{"?",P_QUESTIONMARK},				// pre-compiler
	//embracements
	{"(",P_PARENTHESES_OPEN},			// pre-compiler
	{")",P_PARENTHESES_CLOSE},			// pre-compiler
	{"{",P_BRACE_OPEN},					// pre-compiler
	{"}",P_BRACE_CLOSE},				// pre-compiler
	{"[",P_SQBRACKET_OPEN},
	{"]",P_SQBRACKET_CLOSE},
	//
	{"\\",P_BACKSLASH},
	//precompiler operator
	{"#",P_PRECOMP},					// pre-compiler
	{"$",P_DOLLAR},
	{NULL, 0}
};

idLexer::idLexer()
    : m_buffer()
    , m_file_name()
    , m_current_p()
    , m_line()
    , m_lastline()
    , m_token()
{
    const char eof_ex = '\0';
    m_current_p = &eof_ex;
}

idLexer::~idLexer()
{
}

int idLexer::LoadFile(const char* full_path)
{
    File f{};
    if (!f.Open(full_path)) {
        LOG(warning) << "open file " << full_path << " failed";
        return 0;
    }
    m_file_name = f.FileName();

    auto length = f.Length();
    m_buffer.resize(length + 1);
    length = f.Read(m_buffer.data(), length);
    if (length + 1 < m_buffer.size()) {
        LOG(warning) << "read file size changed. buffer: " << (m_buffer.size() - 1) << " read length: " << length;
        m_buffer.resize(length + 1);
    }

    m_current_p = m_buffer.data();
	return 1;
}

bool idLexer::EndOfToken() const
{
    if (!*m_current_p)
        return true;
    return false;
}

int idLexer::GetToken(idToken* token)
{
    return ReadToken(token);
}

int idLexer::SkipWhiteSpaceAndComment()
{
	while (1) {
		// skip white space
		while(*m_current_p <= ' ') {
			if (!*m_current_p) {
				return 1;
			}
			if (*m_current_p == '\n') {
				m_line++;
			}
			m_current_p++;
		}

		// skip comments
		if (*m_current_p == '/') {
			// comments //
			if (*(m_current_p + 1) == '/') {
                m_current_p++;
				do {
                    m_current_p++;
					if (!*m_current_p) {
						return 1;
					}
				} while (*m_current_p != '\n');
                m_line++;
                m_current_p++;
				if (!*m_current_p) {
					return 1;
				}
				continue;
			} else if (*(m_current_p + 1) == '*') {
                // comments /* */
                m_current_p++;
				while (1) {
                    m_current_p++;
					if (!*m_current_p) {
						return 0;
					}
					if (*m_current_p == '\n') {
                        m_line++;
					} else if (*m_current_p == '/') {
						if (*(m_current_p - 1) == '*') {
							break;
						}
						if (*(m_current_p + 1) == '*' ) {
                            //LOG(warning) << "nested comment";
						}
					}
				}
                m_current_p++;
				if (!*m_current_p) {
					return 1;
				}
				continue;
			}
		}
		break;
	}
	return 1;
}

int idLexer::Line() const
{
    return m_line;
}

const std::string& idLexer::FileName() const
{
    return m_file_name;
}

int idLexer::ReadToken(idToken* token)
{
    if (!*m_current_p) {
        token->m_type = TokenType::Eof;
        return 1;
    }

	// start of the white space read white space before token
    if (!SkipWhiteSpaceAndComment()) {
        return 0;
    }

    if (!*m_current_p) {
        token->m_type = TokenType::Eof;
        return 1;
    }

    int c = *m_current_p;
    if (('A' <= c && c <= 'Z') ||
        ('a' <= c && c <= 'z') ||
        c == '_') {
        if (!ReadName(token)) {
            return 0;
        }
    } 
    // if there is a number
    else if (('0' <= c && c <= '9') ||
			(c == '.' && (*(m_current_p + 1) >= '0' && *(m_current_p + 1) <= '9'))) {
		if (!ReadNumber(token)) {
			return 0;
		}
	}
	// if there is a leading quote
	else if (c == '\"' || c == '\'') {
		if (!ReadString(token, c)) {
			return 0;
		}
	}
	// check for punctuations
	else if (!ReadPunctuation(token)) {
        LOG(error) << "unknown punctuation " << c;
		return 0;
	}
	// succesfully read a token
	return 1;
}

int idLexer::ReadEscapeCharacter(char* ch) 
{
    int c;

	// step over the leading '\\'
	m_current_p++;
	// determine the escape character
	switch(*m_current_p) {
		case '\\': c = '\\'; break;
		case 'n': c = '\n'; break;
		case 'r': c = '\r'; break;
		case 't': c = '\t'; break;
		case 'v': c = '\v'; break;
		case 'b': c = '\b'; break;
		case 'f': c = '\f'; break;
		case 'a': c = '\a'; break;
		case '\'': c = '\''; break;
		case '\"': c = '\"'; break;
		case '\?': c = '\?'; break;
		case 'x':
		{
			m_current_p++;
            int val = 0;
			for (int i = 0, val = 0; ; i++, m_current_p++) {
				c = *m_current_p;
				if (c >= '0' && c <= '9')
					c = c - '0';
				else if (c >= 'A' && c <= 'Z')
					c = c - 'A' + 10;
				else if (c >= 'a' && c <= 'z')
					c = c - 'a' + 10;
				else
					break;
				val = (val << 4) + c;
			}
			m_current_p--;
			if (val > 0xFF) {
				LOG(warning) << "too large value in escape character";
				val = 0xFF;
			}
			c = val;
			break;
		}
		default: //NOTE: decimal ASCII code, NOT octal
		{
			if (*m_current_p < '0' || *m_current_p > '9') {
				LOG(error) << "unknown escape char";
			}
            int val;
			for (int i = 0, val = 0; ; i++, m_current_p++) {
				c = *m_current_p;
				if (c >= '0' && c <= '9')
					c = c - '0';
				else
					break;
				val = val * 10 + c;
			}
			m_current_p--;
			if (val > 0xFF) {
				LOG(warning) << "too large value in escape character";
				val = 0xFF;
			}
			c = val;
			break;
		}
	}
	// step over the escape character or the last digit of the number
	m_current_p++;
	// store the escape character
	*ch = c;
	// succesfully read escape character
	return 1;
}

/*
================
idLexer::ReadString

Escape characters are interpretted.
Reads two strings with only a white space between them as one string.
================
*/
int idLexer::ReadString(idToken* token, int quote) 
{
	if (quote == '\"') {
        token->m_type = TokenType::String;
	} else {
        token->m_type = TokenType::Literal;
	}

	// leading quote
	m_current_p++;

	while (1) {
		// if there is an escape character and escape characters are allowed
		if (*m_current_p == '\\') {
            char ch = 0;
			if (!ReadEscapeCharacter(&ch)) {
				return 0;
			}
			token->AppendCharacter(ch);
		}
		// if a trailing quote
		else if (*m_current_p == quote) {
			// step over the quote
			m_current_p++;

            const char* tmpscript_p = m_current_p;
            int tmpline = m_line;
			// read white space between possible two consecutive strings
			if (!SkipWhiteSpaceAndComment()) {
				m_current_p = tmpscript_p;
				m_line = tmpline;
				break;
			}

			// if there's no leading qoute
			if (*m_current_p != quote) {
				m_current_p = tmpscript_p;
				m_line = tmpline;
				break;
			}
			// step over the new leading quote
			m_current_p++;
		} else {
			if (*m_current_p == '\0') {
                LOG(error) <<"missing trailing quote";
				return 0;
			}
			if (*m_current_p == '\n') {
				LOG(error) << "newline inside string";
				return 0;
			}
			token->AppendCharacter(*m_current_p++);
		}
	}
    // TODO token data
	//token->data[token->len] = '\0';

	if (token->m_type == TokenType::Literal) {
        token->m_subtype = token->GetCharacter(0);
	} else {
		// the sub type is the length of the string
		token->m_subtype = token->Length();
	}
	return 1;
}

/*
================
idLexer::ReadName
================
*/
int idLexer::ReadName(idToken* token)
{
	char c = 0;
    token->m_type = TokenType::Identifier;
	do {
        token->AppendCharacter(*m_current_p++);
		c = *m_current_p;
	} while ((c >= 'a' && c <= 'z') ||
				(c >= 'A' && c <= 'Z') ||
				(c >= '0' && c <= '9') ||
				c == '_' ||
				// if treating all tokens as strings, don't parse '-' as a seperate token
				(c == '-')
				);
	//the sub type is the length of the name
	token->m_subtype = token->Length();

    if (std::find(std::begin(g_keyword), std::end(g_keyword), token->AsStringView()) != std::end(g_keyword))
        token->m_type = TokenType::Keyword;

	return 1;
}

/*
================
idLexer::CheckString
================
*/
int idLexer::CheckString( const char *str ) const
{
	int i;

	for ( i = 0; str[i]; i++ ) {
		if ( m_current_p[i] != str[i] ) {
			return false;
		}
	}
	return true;
}

int idLexer::ReadNumber(idToken* token) 
{
    token->m_type = TokenType::Number;
	token->m_subtype = 0;
	token->m_intvalue = 0;
	token->m_floatvalue = 0;

    char c = *m_current_p;
    char c2 = *(m_current_p + 1);

	if (c == '0' && c2 != '.') {
		// check for a hexadecimal number
		if (c2 == 'x' || c2 == 'X') {
			token->AppendCharacter(*m_current_p++);
			token->AppendCharacter(*m_current_p++);
			c = *m_current_p;
			while(('0' <= c && c <= '9') ||
						('a' <= c && c <= 'f') ||
						('A' <= c && c <= 'F')) {
				token->AppendCharacter(c);
				c = *(++m_current_p);
			}
			token->m_subtype = TT_HEX | TT_INTEGER;
		}
		// its an octal number
		else {
			token->AppendCharacter(*m_current_p++);
			c = *m_current_p;
			while('0' <= c && c <= '7') {
				token->AppendCharacter(c);
				c = *(++m_current_p);
			}
			token->m_subtype = TT_OCTAL | TT_INTEGER;
		}
	}
	else {
		// decimal integer or floating point number or ip address
        int dot = 0;
		while (1) {
			if ('0' <= c && c <= '9') {
			} else if (c == '.') {
				dot++;
			} else {
				break;
			}
			token->AppendCharacter(c);
			c = *(++m_current_p);
		}
		if (c == 'e' && dot == 0) {
			//We have scientific notation without a decimal point
			dot++;
		}
		// if a floating point number
		if (dot == 1) {
			token->m_subtype = TT_DECIMAL | TT_FLOAT;
			// check for floating point exponent
			if (c == 'e') {
				//Append the e so that GetFloatValue code works
				token->AppendCharacter(c);
				c = *(++m_current_p);
				if (c == '-') {
					token->AppendCharacter(c);
					c = *(++m_current_p);
				} else if (c == '+') {
					token->AppendCharacter(c);
					c = *(++m_current_p);
				}
				while ('0' <= c && c <= '9') {
					token->AppendCharacter(c);
					c = *(++m_current_p);
				}
			}
		} else if ( dot > 1 ) {
			if ( dot != 3 ) {
                LOG(error) <<"ip address should have three dots";
				return 0;
			}
			token->m_subtype = TT_IPADDRESS;
		}
		else {
			token->m_subtype = TT_DECIMAL | TT_INTEGER;
		}
	}

	if (token->m_subtype & TT_FLOAT) {
		if (c > ' ') {
			// single-precision: float
			if (c == 'f' || c == 'F') {
				token->m_subtype |= TT_SINGLE_PRECISION;
				m_current_p++;
			}
			// extended-precision: long double
			else if (c == 'l' || c == 'L') {
				token->m_subtype |= TT_EXTENDED_PRECISION;
				m_current_p++;
			}
			// default is double-precision: double
			else {
				token->m_subtype |= TT_DOUBLE_PRECISION;
			}
		} else {
			token->m_subtype |= TT_DOUBLE_PRECISION;
		}
	} else if (token->m_subtype & TT_INTEGER) {
		if (c > ' ') {
			// default: signed long
			for (int i = 0; i < 2; i++) {
				// long integer
				if (c == 'l' || c == 'L') {
					token->m_subtype |= TT_LONG;
				}
				// unsigned integer
				else if (c == 'u' || c == 'U') {
					token->m_subtype |= TT_UNSIGNED;
				} else {
					break;
				}
				c = *(++m_current_p);
			}
		}
	}
    // TODO token data
	//token->data[token->len] = '\0';
	return 1;
}

/*
================
idLexer::ReadPunctuation
================
*/
int idLexer::ReadPunctuation(idToken* token)
{
    for (auto it = std::begin(g_default_punctuations); it != std::end(g_default_punctuations); ++it) {
        const Punctuation& punc = *it;
        int len = 0;
        for (len = 0; punc.m_p[len] && m_current_p[len]; ++len) {
            if (m_current_p[len] != punc.m_p[len])
                break;
        }
        // 找到标点符号
        if (!punc.m_p[len]) {
            for (int i = 0; i != len; ++i) {
                token->AppendCharacter(punc.m_p[i]);
            }

			m_current_p += len;
            token->m_type = TokenType::Puncatuation;
			// sub type is the punctuation id
            token->m_subtype = punc.m_n;
			return 1;
        }
    }
    return 0;
}

/*
================
idLexer::NumLinesCrossed
================
*/
int idLexer::NumLinesCrossed() 
{
	return m_line - m_lastline;
}

const std::vector<char> idLexer::GetBuffer() const
{
    return m_buffer;
}
