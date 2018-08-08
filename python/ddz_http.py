#!/usr/bin/python3 -B

import http.client
import json
import sys


IP = "192.168.0.242"
PORT = 21005

DESC="""
add_coin        [userid] [count]
add_ticket      [userid] [count]
charge          [userid] [shopid]
account_lock    [userid]
account_unlock  [userid]
set_vip         [userid]  [vip]
add_vip         [userid]  [vip]
setvar          [key]  [value]
"""

g_shop = {
    1001:  600,
    1002: 1200,
    1003: 1200,
    1004: 9800,
    1005: 48800,
    1006: 12800,
    2001: 1200,
    2002: 1200,
    5001: 600,
    6001: 3000,
    6002: 9800,
    6003: 29800,
    7001: 1200,
    7002: 1200,
    7003: 3000,
    7004: 5000,
    7005: 9800,
    7006: 99800,
    7007: 99800,
    7008: 5000,
    7009: 99800,
    7010: 1200,
    7011: 1200,
    7012: 1200,
    7013: 3000,
    7014: 3000,
    7015: 5000,
}


def post_http(path, post_data):
    conn = http.client.HTTPConnection(IP, PORT, timeout=10)
    conn.request("POST", path, body=post_data)
    resp = conn.getresponse()
    data = resp.read()
    print(data)


def charge(userid, shopid):
    if shopid not in g_shop:
        print("error shopid {}".format(shopid))
        print(g_shop)
        return
    ps = {}
    ps["cmd"] = 1
    ps["orderid"] = "fake_orderid"
    ps["userid"] = userid
    ps["shopid"] = shopid
    ps["rmb"] = g_shop[shopid]
    post_data = json.dumps(ps)
    post_http("/gamecharge", post_data)


def change_item(userid, id, count):
    if count <= 0:
        return
    ps = {}
    ps["cmd"] = 8
    ps["userid"] = userid
    ps["item_id"] = id
    ps["item_count"] = count
    post_data = json.dumps(ps)
    post_http("/webconsole", post_data)

def account_lock_unlock(userid, type):
    if type != 0 and type != 1:
        return
    ps = {}
    ps["cmd"] = 6
    ps["userid"] = userid
    ps["type"] = type
    post_data = json.dumps(ps)
    post_http("/webconsole", post_data)
    
def set_vip(userid, vip):
    if vip < 0:
        return
    ps = {}
    ps["cmd"] = 3
    ps["userid"] = userid
    ps_set_vip = {}
    ps_set_vip["vip"] = vip
    ps_set_vip["expired_time"] = 0
    ps["set_vip"] = ps_set_vip
    post_data = json.dumps(ps)
    post_http("/webconsole", post_data)

def add_vip(userid, vip):
    if vip <= 0:
        return
    ps = {}
    ps["cmd"] = 3
    ps["userid"] = userid
    ps_add_vip = {}
    ps_add_vip["vip"] = vip
    ps_add_vip["expired_time"] = 0
    ps["add_vip"] = ps_add_vip
    post_data = json.dumps(ps)
    post_http("/webconsole", post_data)
    
def setvar(key, value):
    if len(key) == 0:
        return
    ps = {}
    ps["key"] = key
    ps["value"] = value
    post_data = json.dumps(ps)
    post_http("/setvar", post_data)
 
def main():
    cmd_name = sys.argv[1]
    if cmd_name == "add_coin" and len(sys.argv) == 4:
        change_item(int(sys.argv[2]), 510001, int(sys.argv[3]))
    elif cmd_name == "add_ticket" and len(sys.argv) == 4:
        change_item(int(sys.argv[2]), 530001, int(sys.argv[3]))
    elif cmd_name == "charge" and len(sys.argv) == 4:
        charge(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd_name == "account_unlock" and len(sys.argv) == 3:
        account_lock_unlock(int(sys.argv[2]), 0)
    elif cmd_name == "account_lock" and len(sys.argv) == 3:
        account_lock_unlock(int(sys.argv[2]), 1)
    elif cmd_name == "set_vip" and len(sys.argv) == 4:
        set_vip(int(sys.argv[2]), 1)
    elif cmd_name == "add_vip" and len(sys.argv) == 4:
        add_vip(int(sys.argv[2]), 1)
    elif cmd_name == "setvar" and len(sys.argv) == 4:
        setvar(sys.argv[2], sys.argv[3])
    else:
        print(DESC)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print(DESC)
        exit(-1)
    main()
