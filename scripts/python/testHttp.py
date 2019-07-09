
import http.client
import json
import random
import ssl
import threading
import urllib.request
import time
import datetime

def createBody():
    openid = "id:" + str(random.randint(1, 100000))
    json_req = {}
    json_req["openid"] = openid
    json_req["path"] = "aaf=33sdf&krq_gamebox=123&name=xxs"
    content = json.dumps(json_req)
    return content.encode("utf-8")

def testHttpClient(index):
    d = createBody()
    #print(d)
    #conn = http.client.HTTPSConnection("https://192.168.0.251:10100/dqgs/common/v1/commit_score")
    #conn = http.client.HTTPSConnection("192.168.0.251", "10100")
    #conn = http.client.HTTPSConnection("xyx.17tcw.com", "10200")
    conn = http.client.HTTPConnection("192.168.0.251", 10400)
    conn.request("POST", "/gamebox_v2/login/v1/statis", body=d)
    resp = conn.getresponse()
    s = resp.read().decode('utf-8')
    print("{} {} {}".format(index, datetime.datetime.today(), s))


def threadPool():
    for i in range(0, 1000):
        testHttpClient(i)
        time.sleep(1)

def main():
    #ssl._create_default_https_context = ssl._create_unverified_context
    threadPool()

if __name__ == '__main__':
    #for i in range(0, 1):
    #    print(random.randint(1, 10000))
    main()
