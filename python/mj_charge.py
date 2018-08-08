import json
import sys
import os
import types
import hashlib
import urllib.request
import datetime
import time

SIGN_KEY ="5os6ee572bcbc75830d044e66ab429bc"

def calcMd5(str=""):
    m = hashlib.md5()
    m.update(str.encode(encoding="utf-8"))
    return m.hexdigest()

#### 充值
def postCharge(url="", user_id=0, order_id="", fang_ka=0, tm=0):
    json_req = {}
    json_req["user_id"] = user_id
    json_req["order_id"] = order_id
    json_req["fang_ka"] = fang_ka
    json_req["tm"] = tm
    str = "user_id={user_id}&order_id={order_id}&fang_ka={fang_ka}&tm={tm}&key={key}"\
        .format(user_id=user_id, order_id=order_id, fang_ka=fang_ka, tm=tm, key=SIGN_KEY)
    json_req["sign"] = calcMd5(str)
    post_content = json.dumps(json_req, sort_keys=True,indent=4)
    req = urllib.request.Request(url=url,data=post_content.encode(encoding="utf-8"),method="POST")
    with urllib.request.urlopen(req) as f:
        rsp_content = f.read().decode("utf-8")
        print(rsp_content)

#### 设置参数
def postSetDBConf(url="", conf_id=0, conf_value=""):
    json_req = {}
    json_req["id"] = conf_id
    json_req["value"] = conf_value
    post_content = json.dumps(json_req, sort_keys=True, indent=4)
    req = urllib.request.Request(url=url, data=post_content.encode(encoding="utf-8"), method="POST")
    with urllib.request.urlopen(req) as f:
        rsp_content = f.read().decode("utf-8")
        print(rsp_content)

#### 减少用户房卡数
def postChangeFangKa(url="", user_id=0, fang_ka=0):
    json_req = {}
    json_req["user_id"] = user_id
    json_req["fang_ka"] = fang_ka
    post_content = json.dumps(json_req, sort_keys=True, indent=4)
    req = urllib.request.Request(url=url, data=post_content.encode(encoding="utf-8"), method="POST")
    with urllib.request.urlopen(req) as f:
        rsp_content = f.read().decode("utf-8")
        print(rsp_content)

#### get请求
def getHttp(url=""):
    req = urllib.request.Request(url=url, method="GET")
    with urllib.request.urlopen(req) as f:
        rsp_content = f.read().decode("utf-8")
        print(rsp_content)

def main():
    url = "http://192.168.0.123:21011/game/console/"
    #postCharge(url=url + "charge", user_id=111947, order_id="x114", fang_ka=20, tm=int(time.time()))
    #postSetDBConf(url=url + "set_db_conf", conf_id=14, conf_value="2")
    #postChangeFangKa(url=url + "change_fang_ka", user_id=111947, fang_ka=1)

    ### 设置获取参数
    #getHttp(url=url + "get_db_conf")

    ### 设置获取服务器信息
    #getHttp(url=url + "server_info")

    pass

if __name__ == '__main__':
    main()

