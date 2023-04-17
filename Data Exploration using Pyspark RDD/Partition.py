from pyspark import SparkContext
import os
import json
import time
import sys

os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'
print('cmd entry:',sys.argv)
sc = SparkContext('local[*]','task2')
sc.setLogLevel("WARN")
file_path = sys.argv[1]

def count_num(iterator):
    yield sum(1 for x in iterator)

def count_num_customized(iterator):
    tmpsum = 0
    for x in iterator:
        tmpsum += len(x[1])
    yield tmpsum

def to_list(x):
    return [x]

def to_append(x,y):
    x.append(y)
    return x

def to_extend(x, y):
    x.extend(y)
    return x

def f(x):
    return len(x)

######default method
reviewRDD1 = sc.textFile(file_path).map(lambda x: json.loads(x))

start_default = time.time()

viewed_business1 = reviewRDD1.map(lambda x: (x["business_id"],1))

distinct_business = viewed_business1.reduceByKey(lambda x,y: x+y)

distinct_business_top10 = distinct_business.sortBy(lambda x: [-x[1], x[0]]).take(10)

end_default = time.time() - start_default


default_partition = reviewRDD1.getNumPartitions()
default_item =viewed_business1.mapPartitions(count_num).collect()
#####################

#####customized method_1
reviewRDD3 = sc.textFile(file_path).map(lambda x: json.loads(x))

start_customized1 = time.time()

viewed_business3 = reviewRDD3.map(lambda x: (x["business_id"],1))
num_partition_customized = int(sys.argv[3])
viewed_business3_customized = viewed_business3\
    .combineByKey(to_list, to_append, to_extend,num_partition_customized,lambda x: hash(x)%num_partition_customized)

viewed_business3_top10 = viewed_business3_customized.mapValues(f)\
    .sortBy(lambda x:[-x[1],x[0]]).take(10)

end_customized = time.time() - start_customized1


customized_partition = num_partition_customized
customized_item = viewed_business3_customized.mapPartitions(count_num_customized).collect()

#answer dict#
ans_dict = {
    "default":{
        "n_partition": default_partition,
        "n_items": default_item,
        "exe_time": end_default
    },
    "customized":{
        "n_partition": customized_partition,
        "n_items": customized_item,
        "exe_time": end_customized
    }
}

json_obj = json.dumps(ans_dict, indent=6)
with open(sys.argv[2],"w") as file:
    file.write(json_obj)