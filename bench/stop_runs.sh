#!/bin/sh
# stop background test/benchmark processes without matching the calling shell
pkill -x walker 2>/dev/null
ps -eo pid,args | awk '$2 ~ /python3$/ && ($3=="-m" && $4=="pytest" || $3 ~ /bench_/ || $4 ~ /bench_/) {print $1}' | xargs -r kill 2>/dev/null
exit 0
