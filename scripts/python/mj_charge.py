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

'''
##生成sql文件
def parseXmlToSql(desc_data=DescData(), xml_path=""):
    xml_tree = ET.parse(xml_path)
    xml_root = xml_tree.getroot()

    sql_data = SqlData()
    sql_data.m_table_name = desc_data.m_to

    for record in xml_root:
        field_name = []
        row = []
        for field in record:
            to = field.attrib["to"]
            type = field.attrib["type"]
            value = field.text
            if not isServer(to):
                continue
            if value is None:
                value = ""
            field_name.append(field.tag)
            row.append(ATTR_TYPES[type](value))
        sql_data.m_filed_name = field_name
        sql_data.m_rows.append(row)
        if len(field_name) == 0:
            return
    if len(sql_data.m_rows) == 0:
        return
    sql = genInsertSql(sql_data)
    sql_name = os.path.join(desc_data.m_to_dir, desc_data.m_to) + ".sql"
    print("生成sql文件:{}".format(sql_name))
    fw = open(sql_name, "w", encoding="utf-8")
    fw.write(sql)
    fw.close()

'''

def chargeUserID(uid, cnt):
    url = "http://192.168.0.123:21011/game/console/"
    order_id = "0203-{cnt}-{uid}".format(cnt=cnt, uid=uid)
    postCharge(url=url + "charge", user_id=uid, order_id=order_id, fang_ka=200, tm=int(time.time()))

def readFromFile(fpath):
    fr = open(fpath, 'r')
    user_list = []
    for uid in fr:
        user_list.append(int(uid))
    return user_list

def main():
    user_list = readFromFile(r'C:\Users\Administrator\Desktop\02-03.uid.unique.sort')
    cnt = 0
    for u in user_list:
        cnt += 1
        #print(u)
        order_id = "0203-{cnt}-{uid}".format(cnt=cnt, uid=u)
        print(order_id)
    print(len(user_list))

if __name__ == '__main__':
    main()

