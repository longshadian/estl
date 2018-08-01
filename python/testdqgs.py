
import http.client
import json
import random
import ssl
import threading
import urllib.request


def createBody():
    openid = "id:" + str(random.randint(1, 100000))
    avatar = ""
    for i in range(0, 128):
        avatar += 'a'
    json_req = {}
    json_req["openid"] = openid
    json_req["avatar"] = avatar
    json_req["nickname"] = openid
    json_req["score"] = random.randint(1, 10000)
    json_req["type"] = 0
    content = json.dumps(json_req)
    return content.encode("utf-8")


def testUrlLib():
    d = createBody()
    print(d)
    head = {"Content-type": "application/x-www-form-urlencoded"}

    req = urllib.request.Request(url="https://192.168.0.251:10100/dqgs/common/v1/commit_score"
                                 , data=d, headers=head, method="POST")
    with urllib.request.urlopen(req) as f:
        rsp = f.read().decode('utf-8')
        print(rsp)


def testHttpClient(cnt):
    for i in range(0, cnt):
        d = createBody()
        #print(d)
        #conn = http.client.HTTPSConnection("https://192.168.0.251:10100/dqgs/common/v1/commit_score")
        #conn = http.client.HTTPSConnection("192.168.0.251", "10100")
        conn = http.client.HTTPSConnection("xyx.17tcw.com", "10200")
        conn.request("POST", "/dqgs/common/v1/commit_score", body=d)
        resp = conn.getresponse()
        s = resp.read().decode('utf-8')
        #print(s)
        #print("thread_id:{} {}".format(threading.current_thread().getName(), s))


def threadPool():
    thread_list = []
    for i in range(0, 10):
        t = threading.Thread(target=testHttpClient, args=(1,), name=i)
        thread_list.append(t)
    for t in thread_list:
        t.start()
    for t in thread_list:
        t.join()

def main():
    ssl._create_default_https_context = ssl._create_unverified_context
    #testHttpClient()
    threadPool()

if __name__ == '__main__':
    #for i in range(0, 1):
    #    print(random.randint(1, 10000))
    main()
