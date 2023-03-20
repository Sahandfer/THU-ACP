#!/bin/bash
username=$(whoami)
/hadoop/bin/hdfs dfs -mkdir /user/$username/stream/
/hadoop/bin/hdfs dfs -rm /user/$username/stream/*

/hadoop/bin/hdfs dfs -mkdir /user/$username/hw10output/
/hadoop/bin/hdfs dfs -rm /user/$username/hw10output/*

while [ 1 ]; do
    tmplog="words.`date +'%s'`.txt"
    python3 data.py;
    /hadoop/bin/hdfs dfs -put words.txt /user/$username/stream/$tmplog
    echo "`date +"%F %T"` generating $tmplog succeed"
    rm words.txt
    sleep 56;
done
