#!/usr/bin/python3 -B

import subprocess
import datetime
import sys
import os
import shutil


def get_pid(bin_name):
    pid = subprocess.check_output("ps x|grep {src} | grep -v 'grep' | awk '{{print $1}}'".format(src=bin_name), shell=True, universal_newlines=True)
    if pid:
        pid = pid[:-1]
    return pid


def test():
    child = subprocess.Popen("mysql -u root -p123456 ddzgame_bk", stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, shell=True, universal_newlines=True)
    out, err = child.communicate("select * from tvip;")
    for row in err.splitlines():
        # mysql 警告
        if row != "Warning: Using a password on the command line interface can be insecure.":
            print("err", row, "\n")
    for row in out.splitlines():
        print(row)
    child.wait()


def backup_ddzgame(user, passwd, database, sqlname):
    cmd = "mysqldump -u {user} -p{passwd} {database} > {sqlname}".format(
        user=user, passwd=passwd, database=database, sqlname=sqlname)
    child = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             shell=True, universal_newlines=True)
    out, err = child.communicate()
    for row in err.splitlines():
        if row != "Warning: Using a password on the command line interface can be insecure.":
            print("err:backup_ddzgame:", row)
    for row in out.splitlines():
        print("out:backup_ddzgame:", row)
    child.wait()


def start_back(bk_dir):
    if not os.path.exists(bk_dir):
        os.mkdir(bk_dir)
    if not os.path.isdir(bk_dir):
        print("error:{} isn't dir".format(bk_dir))
        return False
    if os.listdir(bk_dir):
        print("error:{} isn't empty".format(bk_dir))
        return False

    db_user = "root"
    db_passwd = "123456"
    db_database ="ddzgame_bk"
    bk_sql = "x.sql"
    bk_sql = os.path.join(bk_dir, bk_sql)

    copy_file_list = [
        ["/home/cgy/gameserver2/ddzgame2/bin/ddzserver2",          "ddzserver2"],
        ["/home/cgy/gameserver2/ddzpersistance/bin/ddzpst",        "ddzpst"],
        ["/home/cgy/gameserver2/redis-ddz/dump.rdb",                "redis-ddz_dump.rdb"],
        ["/home/cgy/gameserver2/redis-rank/dump.rdb",               "redis-rank_dump.rdb"],
    ]
    backup_ddzgame(user=db_user, passwd=db_passwd, database=db_database, sqlname=bk_sql)

    for cp in copy_file_list:
        src = cp[0]
        dst = os.path.join(bk_dir, cp[1])
        shutil.copyfile(src, dst)


def main():
    usage = '''\
usage ./ddz_bk.py [directory]
    directory   备份至哪个目录
'''

    if len(sys.argv) != 2:
        print("请输入备份的目录!")
        print(usage)
        return False
    start_back(sys.argv[1])

if __name__ == '__main__':
    main()
