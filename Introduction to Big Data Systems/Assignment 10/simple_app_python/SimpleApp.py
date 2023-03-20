from __future__ import print_function
import os
import sys
from pyspark import SparkContext
from pyspark.streaming import StreamingContext


# This function sums old and new wordcounts
def update_count(new_count, old_count):
    return sum(new_count, old_count if old_count else 0)


# This functions receives collected RDDs and saves them to a file
def save_res(res):
    if res:
        for i in range(5):
            if not os.path.exists(f"results_{i+1}.txt"):
                with open(f"results_{i+1}.txt", "w") as f:
                    for j in range(min(len(res), 100)):
                        f.write(f"{res[j][0]}: {res[j][1]}\n")
                f.close()
                break


# Create spark context for the program
sc = SparkContext(appName="Py_HDFSWordCount")
# Listen to changes every 60 seconds
ssc = StreamingContext(sc, 60)
# Save checkpoints to keep track of every minute's wordcounts
ssc.checkpoint("hdfs://intro00:9000/user/2022380024/hw10output")
# Read the lines from the stream (current minute)
lines = ssc.textFileStream("hdfs://intro00:9000/user/2022380024/stream")
# Count the number of words (current minute + previous minute if exists)
counts = (
    lines.flatMap(lambda line: line.split(" "))  # Split lines into words
    .map(lambda x: (x, 1))  # Make tuples for each word
    .reduceByKey(lambda a, b: a + b)  # Sum up 1s of tuples with same key
    .updateStateByKey(update_count)  # Add up results from current and previous minutes
    .transform(lambda r: r.sortBy(lambda x: -x[1]))  # Sort RDDs in the descending order
)
# Print and save the top-100 most frequent words.
counts.pprint(100)
counts.foreachRDD(lambda rdd: save_res(rdd.collect()))

ssc.start()
ssc.awaitTermination()
