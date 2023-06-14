import re
import sys
from pyspark import SparkConf, SparkContext
import time


if __name__ == "__main__":
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    lines = sc.textFile(sys.argv[1])

    first = time.time()

    # Students: Implement Word Count!
    # Split lines to words
    words = lines.flatMap(lambda x: re.split(r"[^\w]+", x))
    # Calculate the wordcount using mapreduce
    word_counts = words.map(lambda x: (x, 1)).reduceByKey(lambda v1, v2: v1 + v2)
    # Sort in descending order and print top-10
    for item in word_counts.sortBy(lambda x: -x[1]).collect()[:10]:
        print(item)

    last = time.time()

    print("Total program time: %.2f seconds" % (last - first))
    sc.stop()
