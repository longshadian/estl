#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include <list>
#include <functional>

/// 一张扑克牌用int32_t表示。只使用最低8位
/// 低8位描述, 高4位代表花色，低4位代表牌值
/// 数值	   0        1       2       3	
/// 花色	 方块     梅花      红桃  	黑桃	

/// 数值	    0       1   2   3   4   5   6   7   8   9   10  11  12  13  14      15
/// 牌值	    空牌    3   4   5   6   7   8   9   10  J   Q   K   A   2   小王	    大王
/// 小王和大王高4位设为0

namespace gp_alg {

/// 花色
enum Color
{
    Diamond  = 0,       // 方块
    Club     = 1,       // 梅花
    Heart    = 2,       // 红桃
    Spade    = 3,       // 黑桃
    Color_Count = 4,
};

/// 牌值
enum Value
{
    Card_Null   = 0,
    Card_3      = 1,
    Card_4      = 2,
    Card_5      = 3,
    Card_6      = 4,
    Card_7      = 5,
    Card_8      = 6,
    Card_9      = 7,
    Card_10     = 8,
    Card_J      = 9,
    Card_Q      = 10,
    Card_K      = 11,
    Card_A      = 12,
    Card_2      = 13,
    Card_B      = 14,   // black joker
    Card_R      = 15,   // red joker
};

/// 牌型
enum class Type 
{
    Type_Null          = 0,	    // 未定义
    Type_Signle        = 1,	    // 单牌
    Type_Single_Ser    = 2,	    // 顺子
    Type_Double        = 3,	    // 对子
    Type_Double_Ser    = 4,	    // 连对
    Type_Triple        = 5,	    // 三个
    Type_Triple_Ser    = 6,	    // 三顺(飞机)
    Type_31            = 7,	    // 3带1单 AAA + ?
    Type_32            = 8,	    // 3带1对
    Type_Bomb          = 9,	    // 炸弹
};

/// 牌面值
enum Face
{
    Face_Null   = 0,
    Face_A      = 1,
    Face_2      = 2,
    Face_3      = 3,
    Face_4      = 4,
    Face_5      = 5,
    Face_6      = 6,
    Face_7      = 7,
    Face_8      = 8,
    Face_9      = 9,
    Face_10     = 10,
    Face_J      = 11,
    Face_Q      = 12,
    Face_K      = 13,

    Face_B      = 14,
    Face_R      = 15,
};

/// 卡牌数量
enum { QUANTITY = 54 };

using Card = int32_t;

struct CardType
{
    CardType();
    CardType(Type type, Value value, int32_t type_len);
    ~CardType();
    CardType(const CardType& rhs);
    CardType& operator=(const CardType& rhs);
    CardType(CardType&& rhs);
    CardType& operator=(CardType&& rhs);

    bool operator<(const CardType& rhs) const;
    bool operator==(const CardType& rhs) const;
    bool operator!=(const CardType& rhs) const;

     // 牌型
    Type        m_type;

    // 牌值:取得是规则化后主牌的最小值，例如:
    // JJJ QQQ KKK 33 44 55,这手牌的牌值是J
    // A K J Q 10 7 8 9,规则化后AKQJ10987 这牌值是7
    Value       m_value;

    // 牌型长度
    int32_t     m_type_len;
};

inline
Card makeColor(Card card, Color color)
{
    return (((card) & 0x0F) | (color) << 4);
}

inline
Card makeValue(Card card, Value value)
{
    return (((card) & 0xF0) | (value));
}

inline 
Card makeCard(Color color, Value value)
{
    Card c{};
    c = makeColor(c, color);
    return makeValue(c, value);
}

inline 
Color getCardColor(Card card)
{
    return static_cast<Color>(((card)&0x30) >> 4);
}

inline 
Value getCardValue(Card card)
{
    return static_cast<Value>((card) & 0x0F);
}

Face getCardFace(Card card);

/** 
 * 初始化牌型
 */
void initCard(std::vector<Card>& card);
std::vector<Card> initCard();

/**
 * 检查选中牌能否压过目标牌,如果src为空或src_len==0，只要检查选中牌是否合法
 * 注意：src hand无需牌型化
 * Params   src,src_len 目标牌
 *          hand, hand_len 选中的牌
 *          out:打出去的牌而且是规则化的。
 *          注意:out缓冲区不能小于hand_len
 * Returns  true:能压过    false:不能压过
 */
bool playCard(const Card* src,   int32_t src_len,
              const Card* hand,  int32_t hand_len,
              Card* out, CardType* out_type = nullptr);

/**
 * 自动选择选中可以压过目标的牌。类似提示功能
 * Params   src,src_len:目标牌
 *          hand,hand_len:手中的牌
 *          out:值不等于0就是选中的能压过目标牌的牌。
 *          注意：out的长度不能少于hand_len,选中的牌是未规则化。
 * Returns  true:能压过    false:不能压过
 */
bool autoSelect(const Card* src,     int32_t src_len,
                const Card* hand,    int32_t hand_len,
                Card* out); 

//去除空牌
void removeCardNull(const Card* src, int32_t len, std::vector<Card>* out);
void removeCardNull(std::vector<Card>* src);

/**
* 判断出牌类型，输出规则化牌型
* Params   src,src_len:需要判断的牌
*          out:输出规则化牌型
* Returns  返回牌型和牌型长度(如果该牌型有牌型长度)
*/
CardType parseCardType(const Card* src, int32_t len, std::vector<Card>* out);
CardType parseCardType(const Card* src, int32_t len);
CardType parseCardType(std::vector<Card>* src);


// 按照牌值，花色降序 2 A K Q J ... 5 4 3
std::function<bool(const Card&, const Card&)> sortDesc_Value_Type();

// 按牌面值升序 A 2 3 4 5 6 ... Q K
std::function<bool(const Card&, const Card&)> sortAsc_Face();

std::function<bool(const Card&)> equalCardValue(Value value);

/************************************************************************
 * 内部实现
 ************************************************************************/
namespace detail {

struct Slot
{
    Slot();
    ~Slot();

    Card popOneCard();


    Value   m_value;
    std::array<int32_t, 4> m_colors;   // 4种花色，每种有几张牌
    int32_t m_total_num;            // 不管花色，总共有几张
};

struct SlotClassify
{
    SlotClassify();
    ~SlotClassify();
    SlotClassify(const SlotClassify& rhs);
    SlotClassify& operator=(const SlotClassify& rhs);
    SlotClassify(SlotClassify&& rhs);
    SlotClassify& operator=(SlotClassify&& rhs);

    std::array<Slot, 15> m_origion_slots;               // 总共15种牌

    Slot* getSlotByValue(Value v);
    const Slot* getSlotByValue(Value v) const;


    // 按照长度倒序排
    void sortByLengthDesc();
    std::vector<Slot>    m_sort_result;         // 牌最多的排前面
    int32_t              m_total_num_length;    // 牌长度相同的有几张
};

bool createSlotClassify(const Card* src, int32_t len, SlotClassify* classify);

// 规格化牌
void normalizeTypeList(const SlotClassify& classify, std::vector<Card>* out);

/// 判断是否顺子
bool isSingleSeries(const Card* src, int32_t len);
bool isSingleSeries_FromA_Or_2(const Card* src, int32_t len, std::vector<Card>* out);

int32_t isSeries(const SlotClassify& classify, int32_t cnt);

int32_t compareCardType(const CardType& type1, const CardType& type2);

int32_t searchSeries_CalcLen(Card src_min, Card src_max);

bool searchAtomic(SlotClassify* classify, Card src_min, int32_t count, std::vector<Card>* out);

bool searchSeries(SlotClassify* classify
    , Card src_min
    , Card src_max
    , int32_t src_count
    , std::vector<Card>* out);

bool searchSeries_From3(SlotClassify* classify
    , Value src_min_val                     // 从哪张牌开始选。
    , int32_t src_len                   // 长度多少
    , int32_t src_count                 // 每种牌选几张
    , std::vector<Card>* out);

bool searchSeries_From2(SlotClassify* classify
    , int32_t src_len                   // 长度多少
    , int32_t src_series_count          // 每种牌选几张
    , std::vector<Card>* out);

bool searchBomb_AAA(SlotClassify* classify, std::vector<Card>* out);
bool searchBomb(SlotClassify* classify, Card src_min, std::vector<Card>* out);

} // detail


/************************************************************************
 * 工具函数                                                                 
 ************************************************************************/
namespace utility {

struct CardDesc
{
    Value       m_value;
    std::string m_str;
};

std::string cardToString(const Card* src, int32_t len);
std::string cardToString(const std::vector<Card>& src);
std::string cardValueToString(Value value);
std::string cardValueToString(const std::vector<Card>& src);
std::string cardTypeToString(Type type);
Value stringToCardValue(const std::string& s);
void selectCards(std::vector<Card>* src, Value value, std::vector<Card>* out, int32_t n);
void paddingCardsCount(std::vector<Card>* src, int32_t n, std::vector<Card>* out);
std::vector<Card> pickCardsFromString(const std::string& s, std::vector<Card>* all_cards);


} // utility

} // gp_alg 
