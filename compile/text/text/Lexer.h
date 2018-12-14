#pragma 

#include <vector>

#include "Token.h"

// punctuation ids
enum TokenIdentifie
{
    P_RSHIFT_ASSIGN = 0,
    P_LSHIFT_ASSIGN,
    P_PARMS,
    P_MACRO,
    P_PRECOMPMERGE,
    P_LOGIC_AND,
    P_LOGIC_OR,
    P_LOGIC_GEQ,
    P_LOGIC_LEQ,
    P_LOGIC_EQ,
    P_LOGIC_UNEQ,

    P_MUL_ASSIGN,
    P_DIV_ASSIGN,
    P_MOD_ASSIGN,
    P_ADD_ASSIGN,
    P_SUB_ASSIGN,
    P_INC,
    P_DEC,

    P_BIN_AND_ASSIGN,
    P_BIN_OR_ASSIGN,
    P_BIN_XOR_ASSIGN,
    P_RSHIFT,
    P_LSHIFT,

    P_POINTERREF,
    P_NAMESPACE,
    P_REF_MUL,
    P_MUL,
    P_DIV,
    P_MOD,
    P_ADD,
    P_SUB,
    P_ASSIGN,

    P_BIN_AND,
    P_BIN_OR,
    P_BIN_XOR,
    P_BIN_NOT,

    P_LOGIC_NOT,
    P_LOGIC_GREATER,
    P_LOGIC_LESS,

    P_REF,
    P_COMMA,
    P_SEMICOLON,
    P_COLON,
    P_QUESTIONMARK,

    P_PARENTHESES_OPEN,
    P_PARENTHESES_CLOSE,
    P_BRACE_OPEN,
    P_BRACE_CLOSE,
    P_SQBRACKET_OPEN,
    P_SQBRACKET_CLOSE,
    P_BACKSLASH,

    P_PRECOMP,
    P_DOLLAR,
};

// punctuation
struct Punctuation
{
    const char* m_p;  // punctuation character(s)
	int         m_n;   // punctuation id
};
using punctuation_t = Punctuation;

class idLexer 
{
public:
					idLexer();
					~idLexer();
					idLexer(const idLexer&) = delete;
					idLexer& operator=(const idLexer&) = delete;
					idLexer(idLexer&&) = delete;
					idLexer& operator=(idLexer&&) = delete;
public:
	int				LoadFile(const char* filename);
    bool            EndOfToken() const;
	int				GetToken(idToken* token);
    int             Line() const;

    const std::vector<char> GetBuffer() const;

private:
    int             ReadToken(idToken* token);
	int				SkipWhiteSpaceAndComment();

	int				ReadEscapeCharacter(char* ch);
	int				ReadString(idToken* token, int quote);
	int				ReadName(idToken* token);
	int				ReadNumber(idToken* token);
	int				ReadPunctuation(idToken* token);
	int				CheckString(const char* str) const;
	int				NumLinesCrossed();

private:
    std::vector<char>   m_buffer;
    const char*         m_current_p;            // current pointer in the script
    int                 m_line;
	int				    m_lastline;				// line before reading token
	idToken			    m_token;				// available token
};
