
#include "client_http.hpp"
#include "client_https.hpp"

#include <iostream>

int main()
{
    std::string s =
        "<xml>"
        "<appid>wx44260df0d471fd8e</appid>"
        "<body>wx44260df0d471fd8e-”Œœ∑≥‰÷µ</body>"
        "<mch_id>1378628302</mch_id>"
        "<nonce_str>7068641224577739721</nonce_str>"
        "<notify_url>http://www.5173game.cn:21011/wechatOrderNotify</notify_url>"
        "<out_trade_no>WXP2016122016490800000002</out_trade_no>"
        "<spbill_create_ip>192.168.125.1</spbill_create_ip>"
        "<total_fee>12</total_fee>"
        "<trade_type>APP</trade_type>"
        "<sign>D809FEAACAEA2A492B86447C4B83E63D</sign>"
        "</xml>";

    try {
    //::SimpleWeb::Client<::SimpleWeb::HTTPS> https_client{"https://api.mch.weixin.qq.com/pay/unifiedorder"};
        ::SimpleWeb::Client<::SimpleWeb::HTTPS> https_client{"api.mch.weixin.qq.com"};
        auto resp = https_client.request("POST", "/pay/unifiedorder", s);
        std::string ss;
        resp->content >> ss;
        std::cout << ss << "\n";
    } catch(const std::exception & e) {
        std::cout << "exception:" << e.what() << "\n";
        return 0;
    }

    std::cout << "xxxx\n";

    /*
    auto wechat_resp = https_client.request("POST", wechat_conf->m_wechat_path, wechat_req);
    if (!wechat_resp) {
        sendRespError(m_response);
        return;
    }
    */
    return 0;
}
