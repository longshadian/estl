
import http.client
import json
import random
import ssl
import threading
import urllib.request
import time
import datetime

def createBody():
    json_req = {}
    json_req["key"] = "abc"
    content = json.dumps(json_req)
    return content.encode("utf-8")

def testHttpClient(index):
    d = createBody()
    #print(d)
    #conn = http.client.HTTPSConnection("https://192.168.0.251:10100/dqgs/common/v1/commit_score")
    conn = http.client.HTTPSConnection("wxxyx.17tcw.com", "6081")
    #conn = http.client.HTTPConnection("192.168.0.251", 6081)
    conn.request("POST", "/cldld/ws/server_all", body=d)
    resp = conn.getresponse()
    s = resp.read().decode('utf-8')
    return s


def threadPool():
    for i in range(0, 1000):
        testHttpClient(i)
        time.sleep(1)
def postOne():
    content = testHttpClient(0)
    jsonObj = json.loads(content)
    code = jsonObj["code"]
    msg = jsonObj["msg"]
    all = jsonObj["data"]["all"]
    print("code: {} {}".format(code, msg))
    print("{}".format(json.dumps(all, indent=2)))

def main():
    #ssl._create_default_https_context = ssl._create_unverified_context
    postOne()

if __name__ == '__main__':
    main()
