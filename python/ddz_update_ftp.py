
import os
import hashlib
import sys
import ftplib
import datetime
from functools import partial

class Apk:
    apk_type = 0             # 0:so 1:hall 2:game 3:so_patch
    gameid = 0
    version = ''
    md5 = ''
    size = 0
    patch_merge_md5 = 0
    description = ''

    postfix = ''
    file_name = ''
    dir_name = ''
    path = ''
    root_path = ''
    apk_type_str = ''


APK_TYPE = {
        "libs": {"patch": {"apk_type": 3,  "apk_type_str": "so_patch",  "gameid": 0},
                 "zip": {"apk_type": 0, "apk_type_str": "so",       "gameid": 0}},
        "hall": {"zip": {"apk_type": 1, "apk_type_str": "hall",    "gameid": 0}},
        "bcbm": {"zip": {"apk_type": 2, "apk_type_str": "game",    "gameid": 600001}},
        "ddz":  {"zip": {"apk_type": 2, "apk_type_str": "game",    "gameid": 20001}},
        "dxc":  {"zip": {"apk_type": 2, "apk_type_str": "game",    "gameid": 600003}},
        "dzp":  {"zip": {"apk_type": 2, "apk_type_str": "game",    "gameid": 500002}},
        "lhd":  {"zip": {"apk_type": 2, "apk_type_str": "game",    "gameid": 600002}},
        "dn":  {"zip": {"apk_type": 2, "apk_type_str": "game",    "gameid": 30001}},
    }


def set_apk_type(apk):
    apk.apk_type = APK_TYPE[apk.dir_name][apk.postfix]["apk_type"]
    apk.apk_type_str = APK_TYPE[apk.dir_name][apk.postfix]["apk_type_str"]
    apk.gameid = APK_TYPE[apk.dir_name][apk.postfix]["gameid"]
    return


def md5sum(filename):
    with open(filename, 'rb') as f:
        d = hashlib.md5()
        for buf in iter(partial(f.read, 128), b''):
            d.update(buf)
    return d.hexdigest()


def calc_md5(file):
    with open(file, 'rb') as fp:
        m = hashlib.md5()
        while True:
            blk = fp.read(4096)
            if not blk:
                break
            m.update(blk)
    return m.hexdigest()


def set_md5(apk):
    m = hashlib.md5()
    with open(os.path.join(apk.path, apk.file_name), 'rb') as fp:
        while True:
            blk = fp.read(4096)
            if not blk:
                break
            m.update(blk)
        apk.size = fp.tell()
    apk.md5 = m.hexdigest()


def set_version(apk):
    if apk.postfix == 'zip':
        apk.version = apk.file_name[len(apk.dir_name)+1: -4]
    elif apk.postfix == 'patch':
        begin_version = apk.file_name.split('-')[0]
        apk.version = apk.file_name[len(begin_version)+1: -6]


def get_dir(path):
    dir_list = []
    files = os.listdir(path)
    for f in files:
        if os.path.isdir(os.path.join(path, f)):
            if f == '.' or f == '..' or f == 'so':
                pass
            else:
                dir_list.append(f)
    return dir_list


def get_file(path, dir_name):
    full_path = os.path.join(path, dir_name)
    file_list = []
    files = os.listdir(full_path)
    for f in files:
        if os.path.isfile(os.path.join(full_path, f)):
            apk = Apk()
            apk.dir_name = dir_name
            apk.path = full_path
            apk.root_path = path
            apk.file_name = str(f)
            apk.postfix = apk.file_name.split('.')[-1]
            set_md5(apk)
            set_apk_type(apk)
            set_version(apk)
            file_list.append(apk)
    return file_list


def check_path(path):
    if not os.path.isdir(path):
        print("error:{} is not dir!".format(path))
        return False
    for d in get_dir(path):
        if d not in APK_TYPE:
            print("error:unknown dir {0}".format(d))
            return False
    return True


def apk_sql(apks):
    sql_format = "INSERT INTO tapk(fapk_type,fgameid,fversion,fmd5,fsize,fpatch_merge_md5,fdescription,ffile,foperator,fcreatetime) "\
                "VALUES ('{0}', '{1}', '{2}', '{3}', '{4}', '{5}', '{6}', '{7}', '{8}', NOW());"
    for apk in apks:
        if apk.postfix == 'patch':
            #获取so md5
            so_file_name = apk.file_name.split(".patch")[0]
            so_file_name = "libddz_" + so_file_name.split("-")[-1] + ".so"

            so_file_path = os.path.join(apk.root_path, "so")
            so_file_path = os.path.join(so_file_path, so_file_name)
            so_md5 = calc_md5(so_file_path)
            print(sql_format.format(apk.apk_type, apk.gameid, apk.version, apk.md5, apk.size, so_md5, apk.file_name, apk.dir_name + "/" + apk.file_name, ''))
        else:
            print(sql_format.format(apk.apk_type, apk.gameid, apk.version, apk.md5, apk.size, '', apk.file_name,  apk.dir_name + "/" + apk.file_name, ''))


def ftp_upload_file(ftp, src, dst):
    try:
        file_handler = open(src, "rb")
        ftp.storbinary("STOR {0}".format(dst), file_handler)
        return True
    except ftplib.all_errors as er:
        print(er)
        return False


def ftp_upload_path(ftp, path, tag, apk_list):
    for apk in apk_list:
        src = os.path.join(apk.path, apk.file_name)
        dst = os.path.join(tag + "-" + apk.file_name)
        ftp.cwd(apk.dir_name)
        if ftp_upload_file(ftp, src, dst):
            print("upload {0:<80} {1:<10} {2:<50} success!".format(src, apk.dir_name, dst))
        else:
            return False
        ftp.cwd("..")


def upload_file(path):
    try:
        tag = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        print("tag:", tag)

        apk_list = []
        dir_list = get_dir(path)
        for d in dir_list:
            apks = get_file(path, d)
            for apk in apks:
                apk_list.append(apk)

        ftp = ftplib.FTP(host="192.168.3.3")
        ftp.login(user="abc", passwd="123456")
        ftp.cwd("xx")
        ftp_upload_path(ftp, path, tag, apk_list)
        ftp.quit()
    except BaseException as er:
        print(er)


def main(path):
    format_str = "\t\t{0:<20} {1:<10} {2:<10} {3:<10} {4:<10} {5:<35} {6:<10}"
    print(format_str.format('file_name', 'apk_type', 'apk_type', 'gameid', 'version', 'md5', 'size'))
    sql_apks = []
    dir_list = get_dir(path)
    for d in dir_list:
        print(d)
        apks = get_file(path, d)
        for apk in apks:
            sql_apks.append(apk)
            print(format_str.format(apk.file_name, apk.apk_type_str, apk.apk_type, apk.gameid, apk.version, apk.md5, apk.size))
    apk_sql(sql_apks)


if __name__ == '__main__':
    #path = r'Y:\ddzgame\client\update_pack\upload'
    path = r'Y:\ddzgame\client\update_pack\1.1.6'
    if check_path(path):
        #upload_file(path)
        main(path)
