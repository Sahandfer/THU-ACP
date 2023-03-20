import sys
from pyspark import SparkConf, SparkContext
import time

if __name__ == "__main__":

    # Create Spark context.
    d = 0.8
    num_iter = 100
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    sc.setCheckpointDir("./save/")
    lines = sc.textFile(sys.argv[1])

    first = time.time()

    # Students: Implement PageRank!

    # Read the file and make a list of (node, list of edges) tuples
    node_edge_list = (
        lines.map(lambda x: x.split("\t"))
        .map(lambda item: (item[0], item[1]))
        .distinct()
        .groupByKey()
        .map(lambda x: (x[0], (list(x[1]))))
        .cache()
    )
    # Count the number of nodes
    num_nodes = node_edge_list.count()
    # Set equal probability for each node --> list of (node, rank) tuples
    node_rank_list = node_edge_list.map(lambda x: (x[0], 1.0 / num_nodes))

    for i in range(num_iter):
        # Create the input for the PageRank algorithm ---> list of (node, (edges, rank))
        data = node_edge_list.join(node_rank_list)
        # Calculate the contribution of each edge in the node
        contributions = data.flatMap(
            lambda x: [(e, x[1][1] / len(x[1][0])) for e in x[1][0]]
        ).reduceByKey(lambda v1, v2: v1 + v2)
        # Update Ranks
        node_rank_list = contributions.mapValues(lambda r: d * r + (1 - d) / num_nodes)

        # Create checkpoints for memory overflow issues.
        if i % 10 == 0:
            node_rank_list.checkpoint()

    # Sort in descending order and print top-5
    highest = node_rank_list.sortBy(lambda x: -x[1]).collect()
    print("5 highest:", highest[:5])

    last = time.time()

    print("Total program time: %.2f seconds" % (last - first))
    sc.stop()
