#!/bin/bash

OUTPUTPATH="od_res/"

echo ===== OutDegreeSorted JAVA VERSION =====

echo ===== Compile =====
javac -classpath `/hadoop/bin/yarn classpath` OutDegreeSorted.java
jar cf wc.jar OutDegreeSorted*.class
echo
echo ===== Clear old output files on HDFS =====
/hadoop/bin/hdfs dfs -rm -r $OUTPUTPATH
echo
echo ===== RUN CASE1=====
/hadoop/bin/yarn jar wc.jar OutDegreeSorted /hw5_data/case1 $OUTPUTPATH"case1" 2
echo
# echo ===== RUN CASE2=====
/hadoop/bin/yarn jar wc.jar OutDegreeSorted /hw5_data/case2 $OUTPUTPATH"case2" 20
# echo

echo DONE!

echo You can use "/hadoop/bin/hdfs dfs -get od_res/ {your_local_path}" to get the result file
echo
