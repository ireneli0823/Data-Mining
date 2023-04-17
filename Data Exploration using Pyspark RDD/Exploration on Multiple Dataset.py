from pyspark import SparkContext
import os
import json
import time
import sys
os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'
sc = SparkContext('local[*]','task3')
sc.setLogLevel("WARN")
review_file = sys.argv[1]
business_file= sys.argv[2]

time_1 = time.time()

reviewRDD = sc.textFile(review_file).map(lambda x: json.loads(x)).map(lambda x: (x["business_id"],x["stars"]))
businessRDD = sc.textFile(business_file).map(lambda x: json.loads(x)).map(lambda x: (x["business_id"],x["city"]))
joinRDD = reviewRDD.leftOuterJoin(businessRDD).map(lambda x: (x[1][1],x[1][0]))
avg_city = joinRDD.groupByKey().mapValues(lambda x: sum(x)/len(x))

time_2 = time.time() - time_1

avg_sorted_city = avg_city.sortBy(lambda x: [-x[1], x[0]]).collect()
title = [("city","stars")]

file = open(sys.argv[3],'w')
for i in title:
    file.write(str(i[0])+','+str(i[1]) +'\n')
for i in avg_sorted_city:
    file.write(str(i[0]) + ',' + str(i[1]) + '\n')
file.close()


####task 3 B####
# M1
start_A = time.time()

avg_cityA = avg_city.collect()
avg_cityA_sorted = sorted(avg_cityA, key = lambda x: [-x[1], x[0]], reverse=False)

for x in avg_cityA_sorted[:10]:
    print(x[0])

end_A = time_2 + time.time() - start_A



# M2
start_B = time.time()

avg_cityB = avg_city.sortBy(lambda x: [-x[1], x[0]]).take(10)

for x in avg_cityB:
    print(x[0])

end_B = time_2 + time.time() - start_B



# ans_dict
ans_dict = {
    "m1": end_A,
    "m2": end_B,
    "reason": "Because MapReduce processes data in parallel in a list of chunks, but Python is just a single-threaded languague, which only process data sequentially in a single thread, which means when it process big data, Spark will faster than Python. However, when the dataset is not that big, Python maybe perform well than Spark."
}

json_obj = json.dumps(ans_dict, indent=2)
with open(sys.argv[4] ,"w") as f:
    f.write(json_obj)