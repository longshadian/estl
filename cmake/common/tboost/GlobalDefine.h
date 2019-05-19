#pragma once

#include <cstdint>

enum {MAX_BUFFERED_USERCMD = 64};
enum {MAX_CLIENTS = 32};
enum {MAX_KEYBOARD_SLOT = 180};

enum {MAX_MESSAGE_SIZE = 16384};

constexpr int USERCMD_HZ = 60;			// 60 frames per second
constexpr int USERCMD_MSEC = 1000 / USERCMD_HZ;

enum class CharacterDir
{
    Left,
    Right,
    Up,
    Down,

    LeftUp45,
    RightUp45,
    LeftDown45,
    RightDown45,

    SLeft,
    SRight,
    SUp,
    SDown,

    SLeftUp45,
    SRightUp45,
    SLeftDown45,
    SRightDown45,

    Nil
};

const int MAX_ASYNC_CLIENTS = 32;

const int MAX_USERCMD_BACKUP = 256;
const int MAX_USERCMD_DUPLICATION = 25;
const int MAX_USERCMD_RELAY = 10;

// unreliable server -> client messages
enum class ServerUnreliableMsg : std::int16_t 
{
    EMPTY = std::int16_t(0),
    PING,
    GAMEINIT,
    SNAPSHOT,
};

// reliable server -> client messages
enum class ServerReliableMsg : std::int16_t
{
    PURE,
    RELOAD,
    CLIENTINFO,
    SYNCEDCVARS,
    PRINT,
    DISCONNECT,
    APPLYSNAPSHOT,
    GAME,
    ENTERGAME,
};

// unreliable client -> server messages
enum class ClientUnreliableMsg : std::int16_t
{
    EMPTY = std::int16_t(0),
    PINGRESPONSE,
    USERCMD
};

// reliable client -> server messages
enum class ClientReliableMsg : std::int16_t
{
    PURE,
    CLIENTINFO,
    PRINT,
    DISCONNECT,
    GAME,
};

// server print messages
enum class ServerPrint 
{
    MISC,
    BADPROTOCOL,
    RCON,
    GAMEDENY,
    BADCHALLENGE
};

enum class ServerDl
{
    REDIRECT,
    LIST,
    NONE
};

enum ServerPak
{
    NO,
    YES,
    END
};



