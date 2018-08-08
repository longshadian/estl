#!/usr/bin/python3 -B

import shutil
import os
import datetime
import subprocess

class MysqlConn:
    ip = "127.0.0.1"
    port = 3306
    password = "123456"
    user = "root"
    db_name = ""
    opt = ""

## 导出数据结构
def exportScheme(conn=MysqlConn, table_list={}, dest=""):
    for t in table_list:
        cmd = "mysqldump -u {user} -p{password} -P {port} {opt} --host={ip} {db_name} -d {table} > {file}/{table}.sql" \
            .format(
            opt=conn.opt,
            user=conn.user,
            password=conn.password,
            port=conn.port,
            ip=conn.ip,
            db_name=conn.db_name,
            table=t,
            file=dest
        )
        # print(cmd)
    subprocess.call(cmd, shell=True)

## 导出数据结构和数据
def exportSchemeAndRecord(conn=MysqlConn, table_list={}, dest=""):
    for t in table_list:
        cmd = "mysqldump -u {user} -p{password} -P {port} {opt} --host={ip} {db_name} {table} > {file}/{table}.sql"\
            .format(
            opt=conn.opt,
            user=conn.user,
            password=conn.password,
            port=conn.port,
            ip=conn.ip,
            db_name=conn.db_name,
            table=t,
            file=dest
        )
        #print(cmd)
        subprocess.call(cmd, shell=True)

def main():
    exportScheme()

if __name__ == '__main__':
    main()
