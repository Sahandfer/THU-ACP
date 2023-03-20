#!/bin/bash
num_partitions=(2 3 4 8)
graphs=(roadNet-PA.graph synthesized-1b.graph twitter-2010.graph)
thresholds=(3 16 100)
types=(edge vertex greedy hybrid)

g++ -o ./main -std=c++11 ./main.cpp

for index in ""${!graphs[@]}""
do
    for N in "${num_partitions[@]}"
    do
        for type in "${types[@]}"
        do
            ./main data/${graphs[$index]} $type $N ${thresholds[$index]}
        done
	done
done
