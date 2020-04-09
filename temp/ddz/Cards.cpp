#include "Cards.h"

#include <algorithm>
#include <sstream>
#include <cstring>
#include <array>

namespace gp_alg {

bool check(const Card* src, int32_t src_len,
           const Card* hand, int32_t hand_len)
{
    if (!hand || hand_len == 0)
        return false;
    std::vector<Card> hand_out(hand_len);
    auto hand_type = parseCardType(hand, hand_len, hand_out.data());

    //手牌没有牌型
    if (hand_type.m_type == CARD_TYPE_NULL)
        return false;

    //目标牌是空，只牌型合法就能出
    if (!src || src_len == 0)
        return true;

    std::vector<Card> src_out(src_len);
    auto src_type = parseCardType(src, src_len, src_out.data());
    if (src_type.m_type == CARD_TYPE_NULL) {
        return false;
    }
    return compareCardType(hand_type, src_type) > 0;
}

bool playCard(const Card* src,       int32_t src_len,
              const Card* selected,  int32_t selected_len,
    Card* out)
{
    if (!selected || selected_len <= 0)
        return false;

    //判断手牌是否合法
    std::vector<Card> selected_out(selected_len);
    auto selected_type = parseCardType(selected, selected_len, selected_out.data());
    if (selected_type.m_type == CARD_TYPE_NULL)
        return false;

    //目标牌是空的,手牌可以出
    if (!src || src_len == 0) {
        //不需要排序，hand已经是排序过的 
        for (int32_t i = 0; i < selected_len; ++i)
            out[i] = selected_out[i];
        return true;
    }

    //判断目标牌型
    std::vector<Card> src_out(src_len);
    CardType src_type = parseCardType(src, src_len, src_out.data());
    if (src_type.m_type == CARD_TYPE_NULL) {
        return false;
    }

    if (compareCardType(selected_type, src_type) > 0) {
        for (int32_t i = 0; i < selected_len; ++i)
            out[i] = selected_out[i];
        return true;
    }
    return false;
}

bool autoSelectEx(const Card* src, int32_t src_len,
                const Card* hand, int32_t hand_len,
    Card* out, CardType* /* out_type */)
{
    if (!src || src_len == 0)
        return false;

    Card src_min_bomb = 0;
    std::vector<Card> src_out(src_len);
    auto src_type = parseCardType(src, src_len, src_out.data());
    if (src_type.m_type == CARD_TYPE_ROCKET)
        return false;

    CardPosition classify[CARD_COUNT] = {};
    detail::classifyHandCard(hand, hand_len, classify);

    switch (src_type.m_type) {
    case CARD_TYPE_SIGNLE: {
        std::vector<Card> temp_out(hand_len);
        if (detail::searchAtomic(classify, src_out[0], 1, hand, temp_out.data())) {
            std::copy(temp_out.begin(), temp_out.end(), out);
            return true;
        }
        break;
    }
    case CARD_TYPE_DOUBLE: {
        std::vector<Card> temp_out(hand_len);
        if (detail::searchAtomic(classify, src_out[0], 2, hand, temp_out.data())) {
            std::copy(temp_out.begin(), temp_out.end(), out);
            return true;
        }
        break;
    }
    case CARD_TYPE_TRIPLE: {
        std::vector<Card> temp_out(hand_len);
        if (detail::searchAtomic(classify, src_out[0], 3, hand, temp_out.data())) {
            std::copy(temp_out.begin(), temp_out.end(), out);
            return true;
        }
        break;
    }
    case CARD_TYPE_SINGLE_SER: {
        std::vector<Card> temp_out(hand_len);
        int32_t series_cnt = src_type.m_type_len;
        if (detail::searchSeries(classify, src_out[0], src_out[series_cnt - 1], 1, hand, temp_out.data())) {
            std::copy(temp_out.begin(), temp_out.end(), out);
            return true;
        }
        break;
    }
    case CARD_TYPE_DOUBLE_SER: {
        std::vector<Card> temp_out(hand_len);
        int32_t series_cnt = src_type.m_type_len;
        if (detail::searchSeries(classify, src_out[0], src_out[2 * (series_cnt - 1)], 2, hand, temp_out.data())) {
            std::copy(temp_out.begin(), temp_out.end(), out);
            return true;
        }
        break;
    }
    case CARD_TYPE_TRIPLE_SER: {
        std::vector<Card> temp_out(hand_len);
        int32_t series_cnt = src_type.m_type_len;
        if (detail::searchSeries(classify, src_out[0], src_out[3 * (series_cnt - 1)], 3, hand, temp_out.data())) {
            std::copy(temp_out.begin(), temp_out.end(), out);
            return true;
        }
        break;
    }
    case CARD_TYPE_31: {
        std::vector<Card> temp_out(hand_len);
        if (detail::searchAtomic(classify, src_out[0], 3, hand, temp_out.data())) {
            //带牌随便选
            if (detail::searchAtomic(classify, CARD_NULL, 1, hand, temp_out.data())) {
                std::copy(temp_out.begin(), temp_out.end(), out);
                return true;
            }
            for (int32_t i = 0; i != CARD_COUNT; ++i)
                classify[i].clear();
            detail::classifyHandCard(hand, hand_len, classify);
        }
        break;
    }
    case CARD_TYPE_31_SER: {
        std::vector<Card> temp_out(hand_len);
        int32_t series_cnt = src_type.m_type_len;
        if (detail::searchSeries(classify, src_out[0], src_out[3 * (series_cnt - 1)], 3, hand, temp_out.data())) {
            int32_t i = 0;
            for (i = 0; i < series_cnt; ++i) {
                if (!detail::searchAtomic(classify, CARD_NULL, 1, hand, temp_out.data())) {
                    break;
                }
            }
            if (i == series_cnt) {
                std::copy(temp_out.begin(), temp_out.end(), out);
                return true;
            }
            for (int32_t j = 0; j != CARD_COUNT; ++j)
                classify[j].clear();
            detail::classifyHandCard(hand, hand_len, classify);
        }
        break;
    }
    case CARD_TYPE_32: {
        std::vector<Card> temp_out(hand_len);
        if (detail::searchAtomic(classify, src_out[0], 3, hand, temp_out.data())) {
            if (detail::searchAtomic(classify, CARD_NULL, 2, hand, temp_out.data())) {
                std::copy(temp_out.begin(), temp_out.end(), out);
                return true;
            }

             for (int32_t i = 0; i != CARD_COUNT; ++i)
                 classify[i].clear();
             detail::classifyHandCard(hand, hand_len, classify);
        }
        break;
    }
    case CARD_TYPE_32_SER: {
        std::vector<Card> temp_out(hand_len);
        int32_t series_cnt = src_type.m_type_len;
        //飞机带翅膀 带牌是对子
        if (detail::searchSeries(classify, src_out[0], src_out[3 * (series_cnt - 1)], 3, hand, temp_out.data())) {
            int32_t i;
            for (i = 0; i < series_cnt; ++i) {
                if (!detail::searchAtomic(classify, CARD_NULL, 2, hand, temp_out.data()))
                    break;
            }
            if (i == series_cnt) {
                std::copy(temp_out.begin(), temp_out.end(), out);
                return true;
            }
            for (int32_t j = 0; j != CARD_COUNT; ++j)
                classify[j].clear();
            detail::classifyHandCard(hand, hand_len, classify);
        }
        break;
    }
    case CARD_TYPE_411: {
        //选两张单牌,两张单牌不能包含火箭
        if (!classify[CARD_B_JOKER].empty() && !classify[CARD_R_JOKER].empty()) {
            classify[CARD_B_JOKER].clear();
            classify[CARD_R_JOKER].clear();
        }

        std::vector<Card> temp_out(hand_len);
        if (detail::searchBomb(classify, src_out[0], hand, temp_out.data())) {
            if (detail::searchAtomic(classify, CARD_NULL, 1, hand, temp_out.data()) &&
                detail::searchAtomic(classify, CARD_NULL, 1, hand, temp_out.data())) {
                std::copy(temp_out.begin(), temp_out.end(), out);
                return true;
            }
            for (int32_t i = 0; i != CARD_COUNT; ++i)
                classify[i].clear();
            detail::classifyHandCard(hand, hand_len, classify);
        }
        break;
    }
    case CARD_TYPE_422: {
        //4带2手对子,对子不能是炸弹
        std::vector<Card> temp_out(hand_len);
        if (detail::searchBomb(classify, src_out[0], hand, temp_out.data())) {
            //去除炸弹
            for (auto& cls : classify) {
                if (cls.size() == 4)
                    cls.clear();
            }
            if (detail::searchAtomic(classify, CARD_NULL, 2, hand, temp_out.data()) &&
                detail::searchAtomic(classify, CARD_NULL, 2, hand, temp_out.data())) {
                std::copy(temp_out.begin(), temp_out.end(), out);
                return true;
            }
            for (int32_t i = 0; i != CARD_COUNT; ++i)
                classify[i].clear();
            detail::classifyHandCard(hand, hand_len, classify);
        }
        break;
    }
    case CARD_TYPE_BOMB: {
        std::vector<Card> temp_out(hand_len);
        src_min_bomb = src_out[0];
        if (detail::searchBomb(classify, src_min_bomb, hand, temp_out.data())) {
            std::copy(temp_out.begin(), temp_out.end(), out);
            return true;
        }
        for (int32_t i = 0; i != CARD_COUNT; ++i)
            classify[i].clear();
        detail::classifyHandCard(hand, hand_len, classify);
        break;
    }
    }

    //选择炸弹
    std::vector<Card> temp_out(hand_len);
    if (src_type.m_type != CARD_TYPE_BOMB &&
        detail::searchBomb(classify, src_min_bomb, hand, temp_out.data())) {
        std::copy(temp_out.begin(), temp_out.end(), out);
        return true;
    }

    //火箭
    if (classify[CARD_B_JOKER].size() > 0 && 
        classify[CARD_R_JOKER].size() >0) {
        int32_t card_idx = classify[CARD_B_JOKER].front();
        out[card_idx] = hand[card_idx];

        card_idx = classify[CARD_R_JOKER].front();
        out[card_idx] = hand[card_idx];
        return true;
    }

    return false;
}

bool autoSelect(const Card* src, int32_t src_len,
                const Card* hand, int32_t hand_len,
    Card* out)
{
    CardType out_type;
    return autoSelectEx(src, src_len, hand, hand_len, out, &out_type);
}

CardType parseCardType(const Card* src, int32_t len, Card* out)
{
    // 13，17，19张牌不能组成合法牌型
    if (len == 13 || len == 17 || len == 19)
        return {};

    std::memcpy(out, src, len);
    if (len == 1)
        return {CARD_TYPE_SIGNLE, getCardValue(src[0]), 1};

    if (len == 2) {
        // 对子
        if (getCardValue(src[0]) == getCardValue(src[1]))
            return {CARD_TYPE_DOUBLE, getCardValue(src[0]), 1};

        // 火箭
        if (getCardValue(src[0]) + getCardValue(src[1]) == CARD_B_JOKER + CARD_R_JOKER)
            return {CARD_TYPE_ROCKET, CARD_B_JOKER, 1};
        return {};
    }

    if (len == 3) {
        // 三张
        int32_t val = getCardValue(src[0]);
        if (val == getCardValue(src[1]) &&
            val == getCardValue(src[2])) {
            return {CARD_TYPE_TRIPLE, val, 1};
        }
        return {};
    }

    std::vector<Card> temp(src, src+len);
    detail::sort(temp.data(), len, false);

    uint16_t same_seq[CARD_COUNT] = { 0 };
    int32_t same_seq_count = detail::getSameValueSeq(temp.data(), len, same_seq, true);

    if (len == 4) {
        if ((same_seq[0] >> 8) == 4)
            return {CARD_TYPE_BOMB, getCardValue(src[0]), 1};
        if ((same_seq[0] >> 8) == 3) {
            detail::normalizeTypeList(same_seq, same_seq_count, out);
            return {CARD_TYPE_31, getCardValue(out[0]), 1};
        }
        return {};
    }

    //判断是否单顺,相同牌只有1张,牌值是顺子 9 8 10 J Q K A
    if ((same_seq[0] >> 8) == 1 && len < 13 && detail::isSeries(temp.data(), len)) {
        detail::normalizeTypeList(same_seq, same_seq_count, out);
        return {CARD_TYPE_SINGLE_SER, getCardValue(out[0]), len};
    }

    //判断是否双顺,相同牌有2张，牌值是顺子 JJ QQ KK AA
    if ((same_seq[0] >> 8) == 2 && len % 2 == 0 && 
        detail::isSeries(same_seq, same_seq_count, 2) == same_seq_count) {
        detail::normalizeTypeList(same_seq, same_seq_count, out);
        return {CARD_TYPE_DOUBLE_SER, getCardValue(out[0]), same_seq_count};
    }

    //判断是否三顺 JJJ QQQ KKK AAA
    int32_t trible_series_len = detail::isSeries(same_seq, same_seq_count, 3);
    if ((same_seq[0] >> 8) == 3 && len % 3 == 0 && 
        trible_series_len == same_seq_count) {
        detail::normalizeTypeList(same_seq, same_seq_count, out);
        return {CARD_TYPE_TRIPLE_SER, getCardValue(out[0]), same_seq_count};
    }

    //3+2牌型 AAA 33
    if (len == 5 && (same_seq[0] >> 8) == 3 && (same_seq[1] >> 8) == 2) {
        detail::normalizeTypeList(same_seq, same_seq_count, out);
        return {CARD_TYPE_32, getCardValue(out[0]), 1};
    }

    //4+1+1牌型 AAAA 3 4
    if (len == 6 && (same_seq[0] >> 8) == 4) {
        //排除炸弹带火箭
        if (same_seq_count == 3 && 
            getCardValue(same_seq[1]) + getCardValue(same_seq[2]) == CARD_B_JOKER + CARD_R_JOKER) {
            return {CARD_TYPE_NULL, 0 ,0};
        }
        detail::normalizeTypeList(same_seq, same_seq_count, out);
        return {CARD_TYPE_411, getCardValue(out[0]), 1};
    }

    //4+2+2
    if (len == 8 && (same_seq[0] >> 8) == 4 && (same_seq[1] >> 8) == 2 && (same_seq[2] >> 8) == 2) {
        detail::normalizeTypeList(same_seq, same_seq_count, out);
        return {CARD_TYPE_422, getCardValue(out[0]), 1};
    }

    //飞机31 KKK AAA 5 9 
    //飞机31 101010 JJJ QQQ AAA
    //飞机31 JJJ QQQ KKK AAA 55 4 9
    //飞机31 101010 JJJ QQQ KKK AAA 333 5 9
    if ((same_seq[0] >> 8) == 3 && 
        (trible_series_len > 1 && len == trible_series_len * 3 + trible_series_len)) {
        detail::normalizeTypeList(same_seq, same_seq_count, out);
        return {CARD_TYPE_31_SER, getCardValue(out[0]), trible_series_len};
    }

    //特殊处理3+3_3_3 666 JJJ QQQ KKK
    if (len == 12 && (same_seq[0] >> 8) == 3) {
        int32_t temp_trible_series = detail::isSeries(same_seq + 1, same_seq_count - 1, 3);
        if (temp_trible_series == 3) {
            uint16_t tmp = same_seq[0];
            memmove(same_seq, same_seq + 1, 6);
            same_seq[3] = tmp;
            detail::normalizeTypeList(same_seq, same_seq_count, out);
            return {CARD_TYPE_31_SER, getCardValue(out[0]), trible_series_len};
        }
    }

    //特殊处理4+3_3_3_3 AAAA 666 777 888 999
    if (len == 16 && (same_seq[0] >> 8) == 4) {
        int32_t temp_trible_series = detail::isSeries(same_seq + 1, same_seq_count - 1, 3);
        if (temp_trible_series == 4) {
            uint16_t tmp = same_seq[0];
            memmove(same_seq, same_seq + 1, 8);
            same_seq[4] = tmp;
            detail::normalizeTypeList(same_seq, same_seq_count, out);
            return {CARD_TYPE_31_SER, getCardValue(out[0]), temp_trible_series};
        }
    }

    //特殊处理3+3_3_3_3+1  (444,555) 666 777 888 999 K
    if (len == 16 && (same_seq[0] >> 8) == 3) {
        int32_t temp_trible_series = detail::isSeries(same_seq + 1, same_seq_count - 1, 3);
        if (temp_trible_series == 4 && (same_seq[5] >> 8) == 1) {
            uint16_t tmp = same_seq[0];
            memmove(same_seq, same_seq + 1, 8);
            same_seq[4] = tmp;
            detail::normalizeTypeList(same_seq, same_seq_count, out);
            return {CARD_TYPE_31_SER, getCardValue(out[0]), temp_trible_series};
        }
    }

    //特殊处理4+3_3_3_3_3+1 AAAA 555 666 777 888 999 K
    if (len == 20 && (same_seq[0] >> 8) == 4) {
        int32_t temp_trible_series = detail::isSeries(same_seq + 1, same_seq_count - 1, 3);
        if (temp_trible_series == 5 && (same_seq[6] >> 8) == 1) {
            uint16_t tmp = same_seq[0];
            memmove(same_seq, same_seq + 1, 10);
            same_seq[5] = tmp;
            detail::normalizeTypeList(same_seq, same_seq_count, out);
            return {CARD_TYPE_31_SER, getCardValue(out[0]), temp_trible_series};
        }
    }

    //特殊处理3+3_3_3_3_3+(2|1、1) 444 555 666 777 888 999 (K K, 3 Q)
    if (len == 20 && (same_seq[0] >> 8) == 3) {
        int32_t temp_trible_series = detail::isSeries(same_seq + 1, same_seq_count - 1, 3);
        if (temp_trible_series == 5) {
            uint16_t tmp = same_seq[0];
            memmove(same_seq, same_seq + 1, 10);
            same_seq[5] = tmp;
            detail::normalizeTypeList(same_seq, same_seq_count, out);
            return {CARD_TYPE_31_SER, getCardValue(out[0]), temp_trible_series};
        }
    }

    //特殊处理3_3_3_3_3+1 555 666 777 888 999 K
    if (len == 16 && trible_series_len == 5) {
        uint16_t tmp = same_seq[0];
        memmove(same_seq, same_seq + 1, 8);
        same_seq[4] = tmp;
        detail::normalizeTypeList(same_seq, same_seq_count, out);
        return {CARD_TYPE_31_SER, getCardValue(out[0]), trible_series_len - 1};
    }

    //飞机2 888 999 KK AA
    if ((same_seq[0] >> 8) == 3 && 
        trible_series_len > 1 && 
        len == trible_series_len * 3 + trible_series_len * 2) {
        int32_t n = 0;
        for (n = 0; n < trible_series_len; ++n) {
            if ((same_seq[n + trible_series_len] >> 8) != 2)
                break;
        }
        if (n == trible_series_len) {
            detail::normalizeTypeList(same_seq, same_seq_count, out);
            return {CARD_TYPE_32_SER, getCardValue(out[0]), trible_series_len};
        }
    }

    if ((same_seq[0] >> 8) == 4) {
        int32_t temp_trible_series = detail::isSeries(same_seq + 1, same_seq_count - 1, 3);

        //特殊处理4+33  AAAA 888 999
        if (len == 10 && temp_trible_series == 2) {
            uint16_t tmp = same_seq[0];
            memmove(same_seq, same_seq + 1, 4);
            same_seq[2] = tmp;
            detail::normalizeTypeList(same_seq, same_seq_count, out);
            return {CARD_TYPE_32_SER, getCardValue(out[0]), temp_trible_series};
        }

        //特殊处理4+333+2 AAAA 777 888 999 KK
        if (len == 15 && temp_trible_series == 3 && (same_seq[4] >> 8) == 2) {
            uint16_t tmp = same_seq[0];
            memmove(same_seq, same_seq + 1, 6);
            same_seq[3] = tmp;
            detail::normalizeTypeList(same_seq, same_seq_count, out);
            return {CARD_TYPE_32_SER, getCardValue(out[0]), 3};
        }

        //特殊处理44+3333 QQQQ AAAA 666 777 888 999
        if ((same_seq[1] >> 8) == 4) {
            temp_trible_series = detail::isSeries(same_seq + 2, same_seq_count - 2, 3);
            if (temp_trible_series == 4) {
                same_seq[6] = same_seq[0];
                same_seq[7] = same_seq[1];
                memmove(same_seq, same_seq + 2, 12);
                detail::normalizeTypeList(same_seq, same_seq_count, out);
                return {CARD_TYPE_32_SER, getCardValue(out[0]), 4};
            }
        }
    }
    return {};
}

CardType parseCardType(const Card* src, int32_t len)
{
    std::vector<Card> buffer(len);
    return parseCardType(src, len, buffer.data());
}

CardType parseCardType(std::vector<Card>* src)
{
    std::vector<Card> buffer(src->size());
    auto type = parseCardType(src->data(), (int32_t)src->size(), buffer.data());
    *src = std::move(buffer);
    return type;
}

//0-12	方块3~10-J-Q-K-A-2	13-25 梅花3~10-J-Q-K-A-2
//26-38 红桃3~10-J-Q-K-A-2	39-51 黑桃3~10-J-Q-K-A-2
//52 53	小王 大王
void initCard(std::vector<Card>& card)
{
    Card c;
    for (Card i = 0; i < 52; ++i) {
        c = static_cast<Card>(setCardColor(c, i / 13));	    //花色从数值0(方块)开始到数值3(黑桃)结束
        c = static_cast<Card>(setCardValue(c, i % 13 + 1));	//牌值从数值1(牌值3)开始到数值13(牌值2)结束
        card.push_back(c);
    }
    c = static_cast<Card>(setCardColor(c, 0));				//小王花色的数值为0
    c = static_cast<Card>(setCardValue(c, 14));				//小王牌值的数值为14
    card.push_back(c);

    c = static_cast<Card>(setCardColor(c, 0));				//大王花色的数值为0
    c = static_cast<Card>(setCardValue(c, 15));				//大王牌值的数值为15
    card.push_back(c);
    //std::random_shuffle(card.begin(), card.end());
}

bool autoComplete(const Card* src,       int32_t src_len,
                  const Card* selected,  int32_t selected_len,
                  const Card* hand,      int32_t hand_len,
    Card* out)
{
    //选中的两张牌不是对子,试着去选顺子，
    if (selected_len == 2 && getCardValue(selected[0]) != getCardValue(selected[1])) {
        //选择单顺，连对和三顺不考虑了
        CardPosition classify[CARD_COUNT] = {};
        detail::classifyHandCard(hand, hand_len, classify);

        Card min_val = static_cast<Card>(getCardValue(selected[1]) - 1);
        Card max_val = static_cast<Card>(getCardValue(selected[0]) - 1);
        return detail::searchSeries(classify, min_val, max_val, 1, hand, out);
    } else if (selected_len == 1 && src_len > 0 && src != NULL) {
        std::vector<Card> temp(src_len);
        CardType src_type = parseCardType(src, src_len, temp.data());
        if (src_type.m_type == CARD_TYPE_NULL)
            return false;

        //选中的牌比出的牌小
        if (getCardValue(selected[0]) <= getCardValue(temp[0]))
            return false;

        if (src_type.m_type == CARD_TYPE_DOUBLE) {
            //出的牌是对子，选中对子
            for (int32_t i = 0; i < hand_len; ++i) {
                if (hand[i] == selected[0]) {
                    //前面一张是否相同牌
                    if (i > 0 && getCardValue(hand[i - 1]) == getCardValue(hand[i])) {
                        out[i-1] = hand[i-1];
                        out[i] = hand[i];
                        return true;
                    }

                    //后面一张是否相同牌
                    if (i < hand_len - 1 && getCardValue(hand[i]) == getCardValue(hand[i + 1])) {
                        out[i] = hand[i];
                        out[i+1] = hand[i+1];
                        return true;
                    }
                    return false;
                }
            }
        } else if (src_type.m_type == CARD_TYPE_SINGLE_SER) {
            //出的牌是单顺，试着去选顺子
            if (CARD_10 < getCardValue(selected[0]))
                return false;

            //选择单顺，自动选出能压过的顺子出来n,连对和三顺不考虑了 
            CardPosition classify[CARD_COUNT] = {};
            detail::classifyHandCard(hand, hand_len, classify);

            Card min_val = static_cast<Card>(getCardValue(selected[0]) - 1);
            Card max_val = static_cast<Card>(min_val + src_type.m_type_len - 1);
            return detail::searchSeries(classify, min_val, max_val, 1, hand, out);
        }
    }
    return false;
}

bool autoSlide(const Card* selected, int32_t selected_len,
               const Card* hand, int32_t hand_len,
    Card* out)
{
    if (!selected || selected_len <= 0)
        return false;

    //选中的牌已经是符合牌型,不用补全
    std::vector<Card> temp(selected_len);
    CardType selected_type = parseCardType(selected, selected_len, temp.data());
    if (selected_type.m_type != CARD_TYPE_NULL) {
        std::memcpy(out, selected, selected_len);
        return true;
    }

    //默认找单顺，不考虑连对和三顺
    if (getCardValue(selected[0]) <= CARD_A && 5 < selected_len) {
        int32_t ser_len = getCardValue(selected[0]) - getCardValue(selected[selected_len-1]) + 1;
        if (ser_len < 5)
            return false;
        CardPosition classify[CARD_COUNT] = {};
        detail::classifyHandCard(hand, hand_len, classify);

        Card min_val = static_cast<Card>(getCardValue(selected[selected_len-1]) - 1);
        Card max_val = static_cast<Card>(min_val + ser_len - 1);
        return detail::searchSeries(classify, min_val, max_val, 1, hand, out);
    }
    return false;
}

std::vector<Card> autoPlayMinCards(const Card* hand, int32_t hand_len)
{
    if (!hand || hand_len == 0)
        return {};

    std::vector<Card> out = detail::autoPlayMinCardsDetail(hand, hand_len);
    //先排序,在规则化
    std::sort(out.begin(), out.end(), PreGreaterSort());
    if (!out.empty()) {
        std::vector<Card> buffer(out.size());
        parseCardType(out.data(), static_cast<int32_t>(out.size()), buffer.data());
        return buffer;
    }
    return out;
}

std::vector<Card> simpleAutoPlayMinCards(const Card* hand, int32_t hand_len)
{
    if (!hand || hand_len == 0)
        return {};

    std::vector<Card> remain_cards(hand, hand + hand_len);
    std::vector<Card> out;
    //只出最小牌
    Card min_card = remain_cards.back();
    for (auto it = remain_cards.rbegin(); it != remain_cards.rend(); ++it) {
        if (getCardValue(*it) == getCardValue(min_card))
            out.push_back(*it);
        else 
            break;
    }
    return out;
}

std::vector<Card> removeCardNull(const Card* src, int32_t len)
{
    std::vector<Card> out;
    out.reserve(len);
    for (int32_t i = 0; i != len; ++i) {
        if (src[i] != 0) {
            out.push_back(src[i]);
        }
    }
    return out;
}

std::vector<Card> removeCardNull(const std::vector<Card>& src)
{
    return removeCardNull(src.data(), (int32_t)src.size());
}

int32_t compareCardType(CardType type1, CardType type2)
{
    //火箭最大
    if (type1.m_type == CARD_TYPE_ROCKET)
        return 1;

    if (type2.m_type == CARD_TYPE_ROCKET)
        return -1;

    //炸弹
    if (type1.m_type == CARD_TYPE_BOMB && type2.m_type == CARD_TYPE_BOMB) {
        return type1.m_value < type2.m_value ? -1 : 1;
    } else if (type1.m_type == CARD_TYPE_BOMB) {
        return 1;
    } else if (type2.m_type == CARD_TYPE_BOMB) {
        return -1;
    }

    //其他牌型
    if (type1.m_type == type2.m_type) {
        if (type1.m_type_len != type2.m_type_len)
            return -1;
        if (type1.m_value > type2.m_value)
            return 1;
        else if (type1.m_value < type2.m_value)
            return -1;
        return 0;
    }
    return -1;
}

std::vector<Card> sort_for_show(std::vector<Card>* src)
{
	int32_t len = (int32_t)src->size();
	std::vector<Card> buffer(len);
	auto type = parseCardType(src->data(), len, buffer.data());
	auto out = buffer.data();

	switch (type.m_type)
	{
	case CARD_TYPE_SINGLE_SER:
	case CARD_TYPE_DOUBLE:
	case CARD_TYPE_DOUBLE_SER:
	case CARD_TYPE_TRIPLE:
	case CARD_TYPE_TRIPLE_SER:
	case CARD_TYPE_BOMB:
	case CARD_TYPE_ROCKET:
		detail::sort(out, len, true);
		break;

	case CARD_TYPE_31:
	case CARD_TYPE_31_SER:
	case CARD_TYPE_32:
	case CARD_TYPE_32_SER:
	{
		auto triple_card_count = type.m_type_len * 3;
		detail::sort(out, triple_card_count, true); //3张降序排
		detail::sort(out + triple_card_count, len - triple_card_count, true); //带牌排序
	}
	break;
	case CARD_TYPE_411:
	case CARD_TYPE_422:
	
		detail::sort(out, 4, true);
		detail::sort(out + 4, len - 4, true);
		break;

	default:
		break;
	}

	return buffer;
}


std::vector<Card> bombCard(const Card* hand, int32_t hand_len)
{
    std::vector<Card> bomb{};
    std::vector<Card> src(hand, hand + hand_len);
    for (size_t i = 0; i != src.size(); ++i) {
        auto c = src[i];
        if (c == 0)
            continue;
        if (getCardValue(c) == CARD_B_JOKER || getCardValue(c) == CARD_R_JOKER) {
            detail::pickupRocket(src.data() + i, src.data() + src.size(), &bomb);
            continue;
        }
        if (std::count_if(src.begin() + i, src.end(), PreCardValue(getCardValue(c))) == 4) {
            int cnt = 0;
            for (size_t j = i; j != src.size(); ++j) {
                if (getCardValue(src[j]) == getCardValue(c)) {
                    bomb.push_back(src[j]);
                    src[j] = 0;
                    if (++cnt == 4)
                        break;
                }
            }
        }
    }
    return bomb;
}

std::vector<Card> initCard()
{
    std::vector<Card> all_cards;
    initCard(all_cards);
    std::sort(all_cards.begin(), all_cards.end(), PreGreaterSort());
    return all_cards;
}

} 

/************************************************************************
 * 实现函数                                                                 
 ************************************************************************/
namespace detail {

std::vector<Card> autoPlayMinCardsDetail(const Card* hand, int32_t hand_len)
{
    if (!hand || hand_len == 0)
        return {};

    //先提取火箭和炸弹
    std::vector<Card> rocket;
    std::vector<std::vector<Card>> bomb;
    std::vector<Card> src = detail::exceptBombRocket(std::vector<Card>(hand, hand + hand_len), &rocket, &bomb);

    if (!src.empty()) {
        std::reverse(src.begin(), src.end());
        int val = getCardValue(src[0]);
        int current_cout = 0;
        int32_t i = 0;
        for (; i != (int32_t)src.size(); ++i) {
            if (val == getCardValue(src[i])) {
                current_cout++;
            } else {
                break;
            }
        }

        int next_count = detail::sameValueCount(src.data() + i, (int32_t)src.size() - i);
        if (current_cout == 1) {
            if (next_count == 3) {
                return std::vector<Card>(src.data(), src.data() + 4);
            }
            return {src[0]};
        } else if (current_cout == 2) {
            if (next_count == 3) {
                return std::vector<Card>(src.data(), src.data() + 5);
            }
            return {src[0], src[1]};
        } else if (current_cout == 3) {
            if (next_count == 1) {
                return std::vector<Card>(src.data(), src.data() + 4);
            } else if (next_count == 2) {
                return std::vector<Card>(src.data(), src.data() + 5);
            }
            return std::vector<Card>(src.data(), src.data() + 3);
        }
    }
    //出炸弹和火箭
    for (size_t i = 0; i != bomb.size(); ++i) {
        return bomb[i];
    }
    return rocket;
}

std::vector<Card> exceptBombRocket(std::vector<Card> hand,
    std::vector<Card>* rocket,
    std::vector<std::vector<Card>>* bomb)
{
    std::array<int, CARD_COUNT> cards_arr = {0};
    for (auto c : hand) {
        cards_arr[getCardValue(c)]++;
    }

    for (int val = CARD_3; val <= CARD_2; ++val) {
        if (cards_arr[val] == 4) {
            std::vector<Card> temp;
            tools::selectCards(&hand, val, &temp, 4);
            bomb->push_back(temp);
        }
    }
    if (cards_arr[CARD_R_JOKER] == 1 && cards_arr[CARD_R_JOKER] == 1) {
        tools::selectCards(&hand, CARD_R_JOKER, rocket, 1);
        tools::selectCards(&hand, CARD_B_JOKER, rocket, 1);
    }
    return removeCardNull(hand);
}

int32_t sameValueCount(const Card* hand, int32_t hand_len)
{
    if (!hand || hand_len == 0)
        return 0;
    int32_t n = 0;
    int32_t val = getCardValue(hand[0]);
    for (int32_t i = 0; i != hand_len; ++i) {
        if (val == getCardValue(hand[i])) {
            n++;
        } else {
            break;
        }
    }
    return n;
}

void classifyHandCard(const Card* hand, int32_t hand_len, CardPosition* card_cls)
{
    for (int32_t i = 0; i < hand_len; ++i) {
        int32_t card_value = getCardValue(hand[i]);
        card_cls[card_value].push_back((Card)i);
    }
}

void sort(Card* src, int32_t len, bool desc)
{
    if (desc) {
        for (int32_t i = 1; i < len; ++i) {
            for (int32_t j = 0; j < len - i; ++j) {
                if (compareCard(src[j], src[j + 1]) < 0) {
                    std::swap(src[j], src[j+1]);
                }
            }
        }
    } else {
        for (int32_t i = 1; i < len; ++i) {
            for (int32_t j = 0; j < len - i; ++j) {
                if (compareCard(src[j], src[j + 1]) > 0) {
                    std::swap(src[j], src[j+1]);
                }
            }
        }
    }
}

int32_t compareCard(Card card1, Card card2)
{
    if (card1 == card2)
        return 0;
    if (getCardValue(card1) > getCardValue(card2))
        return 1;
    else if (getCardValue(card1) < getCardValue(card2))
        return -1;
    return (getCardColor(card1) > getCardColor(card2)) ? 1 : -1;
}


int32_t getSameValueSeq(const Card* src, int32_t len, uint16_t* same_seq, bool need_sort)
{
    if (len <= 0)
        return 0;

    Card cur_same = 1;
    Card last_same_val = getCardValue(src[0]);
    Card card_color = static_cast<Card>(1 << (4 + getCardColor(src[0])));
    int32_t same_card_cnt = 0;
    for (int32_t i = 1; i < len; ++i) {
        if (getCardValue(src[i]) == last_same_val) {
            ++cur_same;
            card_color |= static_cast<Card>(1 << (4 + getCardColor(src[i])));
        } else {
            same_seq[same_card_cnt++] = static_cast<uint16_t>((cur_same << 8) | (card_color & 0xF0) | getCardValue(last_same_val));
            cur_same = 1;
            last_same_val = getCardValue(src[i]);
            card_color = static_cast<Card>(1 << (4 + getCardColor(src[i])));
        }
    }
    same_seq[same_card_cnt++] = static_cast<uint16_t>((cur_same << 8) | (card_color & 0xF0) | getCardValue(last_same_val));
    if (need_sort) {
        for (int32_t i = 1; i < same_card_cnt; ++i) {
            for (int32_t j = 0; j < same_card_cnt - i; ++j) {
                if ((same_seq[j] & 0xFF00) < (same_seq[j + 1] & 0xFF00) || 
                    ((same_seq[j] & 0xFF00) == (same_seq[j + 1] & 0xFF00) && 
                    (same_seq[j] & 0x0F)>(same_seq[j + 1] & 0x0F))) {
                    std::swap(same_seq[j], same_seq[j+1]);
                }
            }
        }
    }
    return same_card_cnt;
}

/// 判断是否顺子
bool isSeries(const Card* src, int32_t len)
{
    for (int32_t i = 1; i < len; ++i) {
        Card card_val = getCardValue(src[i - 1]);
        Card next_val = getCardValue(src[i]);
        if (card_val < CARD_3 || next_val > CARD_A || next_val - card_val != 1)
            return false;
    }
    return true;
}

/// 判断是否多张顺，返回顺的次数
int32_t isSeries(const uint16_t* src, int32_t len, int32_t cnt)
{
    int32_t seriles_len = 1;
    for (int32_t i = 1; i < len; ++i) {
        Card card_val = static_cast<Card>(src[i-1] & 0x0F);
        Card next_val = static_cast<Card>(src[i] & 0x0F);
        Card card_count = static_cast<Card>((src[i-1] >> 8));
        Card next_count = static_cast<Card>((src[i] >> 8));

        if (CARD_3 <= card_val && next_val <= CARD_A &&
            next_val - card_val == 1 &&
            card_count == cnt && next_count == cnt) {
            seriles_len++;
        } else {
            break;
        }
    }
    return seriles_len;
}

bool searchAtomic(CardPosition* card_cls, Card card_val, int32_t count,
                  const Card* out_src, Card* out)
{
	for (int32_t n = count; n < 4; ++n) {
		for (int32_t i = getCardValue(card_val) + 1; i <= CARD_R_JOKER; ++i) {
            if (int32_t(card_cls[i].size()) == n) {
                while (count--) {
                    int32_t card_idx = card_cls[i].front();
                    out[card_idx] = out_src[card_idx];
                    card_cls[i].pop_front();
                }
                return true;
            }
		}
	}
	return false;
}
 
bool searchSeries(CardPosition* card_cls, Card src_min_card, Card src_max_card, int32_t src_series_count,
                  const Card* out_src, Card* out)
{
    int32_t src_min_value = getCardValue(src_min_card);                 //目标顺子最小值
    int32_t src_max_value = getCardValue(src_max_card);                 //目标顺子最大值
    int32_t src_series_len = src_max_value - src_min_value + 1;         //目标顺子长度

    //目标顺子最大值是A,没有顺子能压过A
    if (src_max_value >= CARD_A)
        return false;

    for (int32_t val = src_min_value + 1; val <= CARD_A; ++val) {
        //可选顺子的个数不足
        if (CARD_A - val + 1 < src_series_len)
            return false;

        //找到满足条件的第一张牌
        if (int32_t(card_cls[val].size()) >= src_series_count) {
            //后续的等量牌是否存在
            int32_t ser_len = 0;
            for (; ser_len < src_series_len; ++ser_len) {
                if (int32_t(card_cls[val + ser_len].size()) < src_series_count)
                    break;
            }

            //找到了等量的牌
            if (ser_len == src_series_len) {
                for (int32_t i = val; i < val + ser_len; ++i) {
                    int32_t tmp = src_series_count;
                    while (tmp--) {
                        int32_t card_idx = card_cls[i].front();
                        out[card_idx] = out_src[card_idx];
                        card_cls[i].pop_front();
                    }
                }
                return true;
            }
        }
    }
    return false;
}

bool searchBomb(CardPosition* card_cls, Card card_val, const Card* out_src, Card* out)
{
    for (int32_t val = getCardValue(card_val) + 1; val <= CARD_2; ++val) {
        if (card_cls[val].size() == 4) {
            int32_t temp = 4;
            while(temp--) {
                int32_t card_idx = card_cls[val].front();
                out[card_idx] = out_src[card_idx];
                card_cls[val].pop_front();
            }
            return true;
        }
    }
    return false;
}

/*
bool searchBombSer(CardPosition* card_cls, Card card_val, int32_t ser_count,
                    const Card* out_src, Card* out)
{
    if (ser_count == 1)
        return searchBomb(card_cls, card_val, out_src, out);

    //QQQQ KKKK AAAA Q+3不会比这个大 
    int32_t val = getCardValue(card_val);
    if (val + ser_count >= CARD_2)      
        return false;

    //比这个大
    val += 1;
    for (; val <= CARD_A; ++val) {
        //可选顺子的个数不足
        if (CARD_A - val + 1 < ser_count)
            break;

        if (card_cls[val].size() == 4) {
            bool find = true;
            for (int32_t j = 0; j < ser_count; ++j) {
                if (card_cls[j + val].size() < 4) {
                    find = false;
                    break;
                }
            }

            if (find) {
                for (int32_t i = val; i < val + ser_count; ++i) {
                    int32_t temp = 4;
                    while(temp--) {
                        int32_t card_idx = card_cls[i].front();
                        out[card_idx] = out_src[card_idx];
                        card_cls[i].pop_front();
                    }
                }
                return true;
            }
        }
    }

    if (card_val != CARD_NULL) {
        //当前个数连炸不够，长度+1的连炸
        return searchBombSer(card_cls, CARD_NULL, ser_count + 1, out_src, out);
    }
    return false;
}
*/

/*
bool searchTwoSingleExceptRocket(CardPosition* card_cls, const Card* out_src, Card* out)
{
    for (int32_t n = 1; n < 4; ++n) {
        int32_t index_0 = -1;
        int32_t index_1 = -1;
        for (int32_t i = CARD_3; i <= CARD_R_JOKER; ++i) {
            if (int32_t(card_cls[i].size()) == n) {
                int32_t card_idx = card_cls[i].front();
                if (index_0 == -1) {
                    index_0 = card_idx;
                } else if (index_1 == -1) {
                    index_1 = card_idx;
                }
                out[card_idx] = out_src[card_idx];
                card_cls[i].pop_front();

                if (index_0 != -1 && index_1 != -1) {
                    //不能是火箭
                    if (getCardValue(out[index_0]) + getCardValue(out[index_1]) == CARD_B_JOKER + CARD_R_JOKER)
                        continue;
                    return true;
                }
            }
        }
    }
    return false;
}
*/

bool searchTwoDoubleExceptBomb(CardPosition* card_cls, const Card* out_src, Card* out)
{
    int32_t index_0 = -1;
    int32_t index_1 = -1;
    for (int32_t n = 2; n < 4; ++n) {
        for (int32_t i = CARD_3; i <= CARD_R_JOKER; ++i) {
            if (int32_t(card_cls[i].size()) == n) {
                int32_t* p_index = NULL;
                if (index_0 == CARD_NULL) {
                    p_index = &index_0;
                } else if (index_1 == CARD_NULL) {
                    p_index = &index_1;
                }
                //选2张
                int32_t cnt = 2;
                while (cnt--) {
                    int32_t card_idx = card_cls[i].front();
                    out[card_idx] = out_src[card_idx];
                    *p_index = card_idx;
                    card_cls[i].pop_front();
                }
                
                if (index_0 != -1 && index_0 != -1) {
                    return true;
                }
            }
        }
    }
    return false;
}


bool pickupRocket(Card* src, Card* src_end, std::vector<Card>* out)
{
    auto it_r = std::find_if(src, src_end, PreCardValue(CARD_R_JOKER));
    if (it_r == src_end)
        return false;
    auto it_b = std::find_if(src, src_end, PreCardValue(CARD_B_JOKER));
    if (it_b == src_end)
        return false;
    out->push_back(*it_r);
    out->push_back(*it_b);
    *it_r = 0;
    *it_b = 0;
    return true;
}

} // detail

/************************************************************************
 * 工具函数                                                                
 ************************************************************************/
namespace utility {

const CardNumString g_card_num_string[] =
{
    {CARD_NULL,    "?"},
    {CARD_3,       "3"},
    {CARD_4,       "4"},
    {CARD_5,       "5"},
    {CARD_6,       "6"},
    {CARD_7,       "7"},
    {CARD_8,       "8"},
    {CARD_9,       "9"},
    {CARD_10,      "10"},
    {CARD_J,       "J"},
    {CARD_Q,       "Q"},
    {CARD_K,       "K"},
    {CARD_A,       "A"},
    {CARD_2,       "2"},
    {CARD_B_JOKER, "BJoker"},
    {CARD_R_JOKER, "RJoker"}
};

std::string printCards(const Card* src, int32_t len)
{
    std::ostringstream os;
    for (int32_t i = 0; i != len; ++i) {
        os << (int)src[i] << " ";
    }
    return os.str();
}

std::string printCards(const std::vector<Card>& src)
{
    std::ostringstream os;
    for (auto c : src) {
        os << (int)c << " ";
    }
    return os.str();
}

std::string printCardValue(const std::vector<Card>& src)
{
    std::ostringstream os;
    for (auto c : src) {
        os << cardValueToString(getCardValue(c)) << " ";
    }
    return os.str();
}

std::string cardTypeToString(int32_t card_type)
{
    switch (card_type) {
    case CARD_TYPE_SIGNLE :         return "CARD_TYPE_SIGNLE";
    case CARD_TYPE_SINGLE_SER :     return "CARD_TYPE_SINGLE_SER";
    case CARD_TYPE_DOUBLE :         return "CARD_TYPE_DOUBLE";
    case CARD_TYPE_DOUBLE_SER :     return "CARD_TYPE_DOUBLE_SER";
    case CARD_TYPE_TRIPLE :         return "CARD_TYPE_TRIPLE";
    case CARD_TYPE_TRIPLE_SER :     return "CARD_TYPE_TRIPLE_SER";
    case CARD_TYPE_31 :             return "CARD_TYPE_31";
    case CARD_TYPE_31_SER :         return "CARD_TYPE_31_SER";
    case CARD_TYPE_32 :             return "CARD_TYPE_32";
    case CARD_TYPE_32_SER :         return "CARD_TYPE_32_SER";
    case CARD_TYPE_411 :            return "CARD_TYPE_411";
    case CARD_TYPE_422 :            return "CARD_TYPE_422";
    case CARD_TYPE_BOMB :           return "CARD_TYPE_BOMB";
    case CARD_TYPE_ROCKET :         return "CARD_TYPE_ROCKET";
    case CARD_TYPE_MASK :           return "CARD_TYPE_MASK";
    default:
        break;
    }
    return "CARD_TYPE_NULL";
}

std::string cardValueToString(int32_t value)
{
    if (value >= CARD_COUNT)
        return g_card_num_string[CARD_NULL].m_string;
    return g_card_num_string[value].m_string;
}

int32_t stringToCardValue(std::string s)
{
    for (auto it = std::begin(g_card_num_string); it != std::end(g_card_num_string); ++it) {
        if (it->m_string == s)
            return it->m_num;
    }
    return CARD_NULL;
}

void selectCards(std::vector<Card>* src, int32_t value, std::vector<Card>* out, int32_t n)
{
    int cnt = 0;
    for (auto& c : *src) {
        if (c == 0)
            continue;
        if (getCardValue(c) == value) {
            out->push_back(c);
            c = 0;
            if (++cnt == n)
                return;
        }
    }
}

void paddingCardsCount(std::vector<Card>* src, int32_t n, std::vector<Card>* out)
{
    for (auto& c : *src) {
        if (c == 0)
            continue;
        if ((int32_t)out->size() >= n)
            return;
        out->push_back(c);
        c = 0;
    }
}

std::vector<Card> pickCardsFromString(std::string s, std::vector<Card>* all_cards)
{
    std::vector<Card> out;
    std::istringstream istm(s);
    std::string temp;
    while (istm >> temp) {
        int32_t card_val = tools::stringToCardValue(temp);
        //std::cout << card_val << ":" << temp << "\n";
        if (card_val != 0) {
            tools::selectCards(all_cards, card_val, &out, 1);
        }
    }
    return out;
}

} // utility

} // gp_alg
