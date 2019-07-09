import json
import datetime
import http.client


def statis(version="", game="", type=0, body="", tm="", openid="", channel="", uid=0):
    statis = {}
    statis["version"] = version
    statis["game"] = game
    statis["type"] = type
    statis["tm"] = tm
    statis["openid"] = openid
    statis["uid"] = uid
    statis["channel"] = channel
    statis["body"] = body
    return json.dumps(statis)

def testPath():
    body = {}
    body["path"] = "aaaaaaa"
    bodyStr = json.dumps(body)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = statis(version="100", game="mcdzz_test", type=1, tm=timestamp, body=bodyStr, openid="xxopenid",
                     channel="aaxx",
                     uid=13)
    print("send: {}".format(content))
    httpPost(content)

def testVideo():
    body = {}
    body["interfacex"] = "interface_name"
    body["type"] = "type"
    body["error_code"] = "error_code"
    body["error_msg"] = "error_msg"

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = statis(version="100", game="mcdzz_test", type=2,tm=timestamp, body=json.dumps(body),
                     openid="xxopenid", channel="xxchannel", uid=233)
    print("send: {}".format(content))
    httpPost(content)

def testAppReport():
    body = {}
    body["classify"] = 1
    body["code"] = 122
    body["msg"] = "aaaaa"
    body["content"] = "unknown"
    bodyStr = json.dumps(body)
    timestamp= datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = statis(version="100", game="mcdzz_test", type=3, tm=timestamp, body=bodyStr,
                     openid="xxopenid", channel="xxchannel", uid=8887)
    print("send: {}".format(content))
    httpPost(content)


def httpPost(content):
    conn = http.client.HTTPConnection("192.168.0.251", 12010)
    #conn = http.client.HTTPSConnection("xiaoyouxi.17tcw.com", 12010)
    conn.request("POST", "/frontend/save", body=content)
    resp = conn.getresponse()
    respStr = resp.read().decode("utf-8")
    print("resp: {}".format(respStr))


def main():
    #testPath()
    #testVideo()
    testAppReport()


if __name__ == "__main__":
    #print(random.randint(10,10000))
    main()