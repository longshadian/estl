
import http.client
import json
import random
import time
import datetime
import configparser
import os

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
    conn = http.client.HTTPConnection("127.0.0.1", 9092)
    #conn.request("GET", "/zhwkUpdatePkg/localserverupdate.json", body=d)
    conn.request("GET", "/zhwkUpdatePkg/localserverupdate.json")
    resp = conn.getresponse()
    s = resp.read().decode('utf-8')
    print("{} {} {}".format(index, datetime.datetime.today(), s))
    conn.close()

def threadPool():
    for i in range(0, 1000):
        testHttpClient(i)
        time.sleep(1)
        break

def TestIni():
    print(os.path.abspath(__file__))
    current_path = os.path.dirname(os.path.realpath(__file__))
    current_path = os.path.join(current_path, "python")
    conf_path = os.path.join(current_path, "updateconfig.ini")
    print(conf_path)
    conf = configparser.ConfigParser()
    conf.read_file(open(conf_path))

    LocalServerUpdate=conf.get("config", "LocalServerUpdate")
    LocalServerURL = conf.get("config", "LocalServerURL")
    LocalServerExe = conf.get("config", "LocalServerExe")
    LocalServerPath = conf.get("config", "LocalServerPath")
    print("{} {} {} {}".format(LocalServerUpdate, LocalServerURL, LocalServerExe, LocalServerPath))

def main():
    #ssl._create_default_https_context = ssl._create_unverified_context
    #threadPool()
    TestIni()

if __name__ == '__main__':
    for i in range(0, 1):
        print(random.randint(1, 10000))
    main()
