from graphframes import GraphFrame
from pyspark import SparkContext
from pyspark.sql import SparkSession, SQLContext, Row
import os
import itertools
import sys
import time

start = time.time()
os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 pyspark-shell"
sc = SparkContext('local[*]', 'HW4_task1')
sc.setLogLevel("WARN")



spark = SparkSession.builder.appName("GraphFrame").getOrCreate()

# Create an SQLContext
sqlContext = SQLContext(sc)


input_file = "ub_sample_data.csv"
output_file = "out_task1.txt"
threshold = 7

RDD = sc.textFile(input_file)
header = RDD.first()
RDD = RDD.filter((lambda x: x != header))

# load rdd to (user_id,business_id) format
user_businesses_dict = RDD.map(lambda x: x.split(",")).map(lambda x: (x[0], x[1])).groupByKey().mapValues(list)\
    .map(lambda x: (x[0], sorted(x[1]))).collectAsMap()
#print(user_businesses_dict)

user_pairs = list(itertools.combinations(user_businesses_dict.keys(),2))
#print(user_pairs)

vertices = set()
edges = list()

for i in user_pairs:
    set_0 = set(user_businesses_dict[i[0]])
    set_1 = set(user_businesses_dict[i[1]])
    num_intersect = set_1.intersection(set_0)
    if len(num_intersect) >= threshold:
        vertices.add(i[0])
        vertices.add(i[1])
        edges.append((i[0], i[1]))
        edges.append((i[1], i[0]))

vertices_rdd = sc.parallelize(list(vertices)).map(lambda x: Row(x))
vertices_df = spark.createDataFrame(vertices_rdd, ["id"])
#vertices_df.show()

edges_df = sc.parallelize(edges).toDF(["src", "dst"])
#edges_df.show()

gf = GraphFrame(vertices_df, edges_df)
lpa_df = gf.labelPropagation(maxIter=5)
#print(lpa_df.rdd.take(100))


ans = lpa_df.rdd.map(lambda x: (x[1],x[0])).groupByKey().mapValues(lambda x: sorted(list(x)))\
    .sortBy(lambda x: (len(x[1]), x[1])).map(lambda x: x[1])

#print(ans.take(1000))

ans_list = ans.collect()

with open(output_file, "w") as file:
    for i in ans_list:
        line = ", ".join(f"'{x}'" for x in i)
        file.write(line + "\n")

end_time = time.time() - start

print(end_time)