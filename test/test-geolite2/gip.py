######################################################################
#### GeoIP2查询需要安装下列库
#### pip3 install maxminddb -i https://mirrors.aliyun.com/pypi/simple/
#### pip3 install geoip2 -i https://mirrors.aliyun.com/pypi/simple/
####
######################################################################

import geoip2.database

def Main():
    p = r'GeoLite2-City.mmdb'
    reader = geoip2.database.Reader(p)

    ip_arr = [
        "39.156.69.79",
        "172.104.181.124",
        "23.239.14.204",
        "81.2.69.160",
        "87.248.100.201",
        "67.195.231.22",
        "202.214.194.147",
        "31.13.69.86",
        "98.136.100.146",
        "104.22.29.202",
        "66.220.147.47",
        "61.129.7.47",
        "39.156.69.79",
        "123.58.180.7",
        "203.107.42.62"
    ]
    for ip in ip_arr :
        response = reader.city(ip)
        #print(response.country)
        #print(response.city)
        country = response.country
        city = response.city
        print("{}\t {} \t {}".format(ip, country.iso_code, city.names))
    print()

if __name__ == '__main__':
    Main()
