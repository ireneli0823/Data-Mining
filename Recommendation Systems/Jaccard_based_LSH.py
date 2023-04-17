from pyspark import SparkContext
import os
import time
import random

import sys
import csv

# start timing!
start = time.time()

os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'
print('cmd entry:',sys.argv)
file_path = sys.argv[1]
out_file_path = sys.argv[2]
#file_path = "small.csv"
sc = SparkContext('local[*]','HW3_task1')
sc.setLogLevel("WARN")

# define minhash function


#get params of hash fuctions:
def getHashParams(n):
    ans = list()
    a = [random.randint(1, 2147483647) for _ in range(n)]
    b = [random.randint(0, 2147483647) for _ in range(n)]
    for i in range(n):
        tmp = list();
        tmp.append(a[i])
        tmp.append(b[i])
        ans.append(tmp)
    return ans

num_hashfuctions = 30
num_bands = 15
rows_in_band = 2





RDD = sc.textFile(file_path)
header = RDD.first()
RDD = RDD.filter((lambda x: x != header))

# load rdd to (business_id,user_id) format
RDD = RDD.map(lambda x: x.split(",")).map(lambda x: (x[1], x[0]))

# get distinct business_id
business_idRDD = RDD.map(lambda x: x[0]).distinct().sortBy(lambda x: x).zipWithIndex()
business_id_dict = business_idRDD.collectAsMap()
business_id_dict_inverted = dict()
for key in business_id_dict:
    business_id_dict_inverted[business_id_dict[key]] = key

#print(business_id_dict)

# get distinct user_id
user_idRDD = RDD.map(lambda x: x[1]).distinct().sortBy(lambda x: x).zipWithIndex()
user_id_dict = user_idRDD.collectAsMap()
num_user = len(user_id_dict)
#print(user_idRDD.collect())

# group user_id by business_id
group_userRDD = RDD.groupByKey().mapValues(list).map(lambda x: (business_id_dict[x[0]],x[1]))\
    .mapValues(lambda x: [user_id_dict[i] for i in x])
#print(group_userRDD.collect())


#
hashParams = getHashParams(num_hashfuctions)
user_hashRDD = user_idRDD.map(lambda x:(x[1],[((i[0]*x[1]+i[1])%2147483647)%num_user for i in hashParams]))
#print(user_hashRDD.collect())
#print(num_user)
def minhash(x):
    # Create a random permutation of the indices
    res = list()
    for j in hashParams:
        tmp = list()
        for i in x:
            tmp.append( ((j[0] * i + j[1]) % 2147483647) % num_user )
        res.append(min(tmp))
    return res

def partitionMtobBands(x):
    res = list()
    for i in range(num_bands):
        tmp = list()
        for j in range(rows_in_band):
            tmp.append(x[i*rows_in_band+j])
        res.append(tmp)
    return res
# minhash
signature_matrixRDD = group_userRDD.mapValues(lambda x:minhash(x))
signature_matrix_with_partitionRDD = signature_matrixRDD.mapValues(lambda x:partitionMtobBands(x))
#print(signature_matrix_with_partitionRDD.collect())

def f(ite):
    signature_list = list(ite)
    bucket_list = list()
    for i in range(num_bands):
        bucket_list.append(dict())
    for signature in signature_list:
        business_id = signature[0]
        bands_list = signature[1]
        #print(business_id)
        #print(bands_list)
        for i in range(num_bands):
            if(tuple(bands_list[i]) in bucket_list[i]):
                bucket_list[i][tuple(bands_list[i])].append(business_id)
            else:
                tmplist = list()
                tmplist.append(business_id)
                bucket_list[i][tuple(bands_list[i])] = tmplist
    for bucket in bucket_list:
        #print(bucket)
        qqq = 1
    #print()
    yield bucket_list

def mergeList(x,y):
    x.extend(y)
    return x

def pair(list_ite):
    res = []
    for i in range(len(list_ite)):
        for j in range((i+1),len(list_ite)):
            if j < len(list_ite):
                res.append((list_ite[i],list_ite[j]))
    return res

for i in range(num_bands):
    bucketRDD = signature_matrix_with_partitionRDD.mapPartitions(f).map(lambda x:x[i])
    #print("bucket",list(bucketRDD.collect()))

    # convert the dictionary to a list of key-value tuples
    bucketRDD_list = list(bucketRDD.collect())
    my_list = list()
    for mydic in bucketRDD_list:
        my_list.append(list(mydic.items()))

    my_rdd = sc.parallelize(my_list).flatMap(lambda x:x).reduceByKey(mergeList).filter(lambda x:len(x[1])>1)
    candidate_rdd = my_rdd.mapValues(lambda x:pair(x)).map(lambda x: x[1]).flatMap(lambda x:x)
    if i==0:
        candidate_all_rdd = candidate_rdd
    else:
        candidate_all_rdd = candidate_all_rdd.union(candidate_rdd)
    # print the RDD contents
    #print(candidate_rdd.collect())
#print(candidate_all_rdd.collect())
candidate_set = set(candidate_all_rdd.collect())
group_user_dict = dict(group_userRDD.collect())
#print(candidate_set)
final_result = list()
def computejaccard(A, B):
    intersection_AB = A.intersection(B)
    union_AB = A.union(B)
    return len(intersection_AB)/len(union_AB)
for candidate_pair in candidate_set:
    set1 = set(group_user_dict[candidate_pair[0]])
    set2 = set(group_user_dict[candidate_pair[1]])
    similarity = computejaccard(set1,set2)
    if(similarity>=0.5):
        b1 = business_id_dict_inverted[candidate_pair[0]]
        b2 = business_id_dict_inverted[candidate_pair[1]]
        if b1>b2:
            tmp = b1
            b1 = b2
            b2 = tmp
        final_result.append([b1,b2,similarity])

final_result.sort(key = lambda x:(x[0],x[1]))

with open(out_file_path,'w') as f:
    writer = csv.writer(f)
    header = ['business_id_1','business_id_2','similarity']
    writer.writerow(header)
    writer.writerows(final_result)


end_time = time.time() - start

#print(end_time)