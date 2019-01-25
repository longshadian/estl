#!/bin/bash

. /etc/profile

cd /mnt/data/zw/tbj_server/kaixinmaxituantbj
python36 ./auto_restart.py >> ./temp/auto_restart.log 2>&1