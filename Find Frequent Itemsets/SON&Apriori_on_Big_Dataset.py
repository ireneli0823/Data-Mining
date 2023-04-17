from pyspark import SparkContext
import os
import time
import sys
import csv

os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'
print('cmd entry:',sys.argv)
file_path = sys.argv[3]
sc = SparkContext('local[*]','task2')
sc.setLogLevel("WARN")

RDD = sc.textFile(file_path)

def remove_character(ite):
    #tmp = []
    for i in range(len(ite)):
        ite[i] = ite[i][1:-1]
    return ite
header = RDD.first()
RDD = RDD.filter(lambda x: x != header).map(lambda x: x.split(",")).map(remove_character)\
    .map(lambda x: (x[0]+"-"+str(int(x[1])), int(x[5])))

new_header = ["DATE-CUSTOMER_ID","PRODUCT_ID"]
res = RDD.collect()
res.insert(0,new_header)
with open('after_prepossess.csv','w') as f:
    writer = csv.writer(f)
    writer.writerows(res)







file_path = "after_prepossess.csv"

support_th = int(sys.argv[2])
filter_th = int(sys.argv[1])

def f(ite):
    partition_size = 0#sum(1 for x in ite)
    tmp_list = list()
    for x in ite:
        partition_size += 1
        tmp_list.append(x)
    tmp_support_th = (partition_size / total_num_user) * support_th
    #print(partition_size / total_num_user)


    yield apriori(tmp_list,tmp_support_th)

def apriori(baskets,support):
    tmp = dict()
    res = list()
    lk = list()
    for i in baskets:
        for x in i:
            if x not in tmp:
                tmp[x] = 1
            else:
                tmp[x] += 1
    for key in tmp:
        if tmp[key] >= support:
            tmptuple = (key,)
            lk.append(tmptuple)
            res.append(tmptuple)
    #ck = generate_ck(lk,1)
    ck = list()
    for i in range(len(lk)):
        for j in range(i+1,len(lk)):
            tmptuple = (min(lk[i][0],lk[j][0]),max(lk[i][0],lk[j][0]))
            ck.append(tmptuple)

    n = 2
    sum1 = 3
    while len(ck) > 0:
        lk = turly_filter(baskets,support,ck)
        for j in lk:
            res.append(j)
        ck = generate_ck(lk,sum1)
        n += 1
        sum1 += n
    return res


def turly_filter(baskets,support,ck):
    lk = list()
    keymap = dict()
    for basket in baskets:
        tmp_basket_set = set(basket)
        for itemtuple in ck:
            flag = True
            for item in itemtuple:
                if item not in tmp_basket_set:
                    flag = False
                    break;
            if flag:
                if itemtuple in keymap:
                    keymap[itemtuple] += 1
                else:
                    keymap[itemtuple]  = 1
    for key in keymap:
        if keymap[key]>=support:
            lk.append(key)
    return lk

def generate_ck(lk,sum1):
    ans_ck = list()
    tmp_dict = dict()
    for i in range(len(lk)):
        for j in range(i+1,len(lk)):
            tmp_tuple = tuple(sorted(set(lk[i]).union(set(lk[j]))))
            if (tmp_tuple in tmp_dict) and (len(tmp_tuple)==len(lk[i])+1):
                tmp_dict[tmp_tuple] += 1
            else:
                tmp_dict[tmp_tuple] = 1
    for x in tmp_dict:
        if tmp_dict[x] >= sum1:
            ans_ck.append(x)
    return ans_ck

def re_count(ite):
    ans = dict()
    for j in ite:
        for i in candidates_list_sorted:
            if set(i[0]).issubset(set(j)):
                if i[0] in ans:
                    ans[i[0]] += 1
                else:
                    ans[i[0]] = 1
    return ans.items()

start = time.time()  ##start timing!!
RDD = sc.textFile(file_path)
header = RDD.first()


RDD = RDD.filter((lambda x: x != header))


itemsetsRDD = RDD.map(lambda x: x.split(",")).map(lambda x: (x[0],x[1])).distinct().groupByKey().mapValues(list)\
    .filter(lambda x: len(x[1]) > filter_th).map(lambda x: x[1])

total_num_user = itemsetsRDD.count()
itemsetsRDD_1 = itemsetsRDD.mapPartitions(f).flatMap(lambda x:x).map(lambda x:(x,1)).reduceByKey(lambda x,y:x+y)


candidates_list = itemsetsRDD_1.collect()
candidates_list_sorted = sorted(candidates_list, key = lambda x: (len(x[0]),x))


with open(sys.argv[4],'w') as f:
    f.write("Candidates:")
    f.write('\n')
    res_str = ""
    prevLen = 1
    for j in candidates_list_sorted:
        i = j[0]
        if len(i) != prevLen:
            res_str = res_str[:-1]
            f.write(res_str)
            f.write('\n\n')
            prevLen = len(i)
            res_str = ""
        str1 = "("
        for j in i:
            str1 += '\'' + j + '\'' + ", "
        str1 = str1[:-2] + "),"
        res_str += str1
    res_str = res_str[:-1]
    f.write(res_str)
    f.write('\n\n')


itemsetsRDD_2 = itemsetsRDD.mapPartitions(re_count).reduceByKey(lambda x,y: x+y)\
    .filter(lambda x: x[1] >= support_th)

truly_frequent_list = sorted(itemsetsRDD_2.collect(), key = lambda x: (len(x[0]),x))



with open(sys.argv[4],'a') as f:
    f.write("Frequent Itemsets:")
    f.write('\n')

    res_str = ""
    prevLen = 1
    for j in truly_frequent_list:
        i = j[0]
        if len(i) != prevLen:
            res_str = res_str[:-1]
            f.write(res_str)
            f.write('\n\n')
            prevLen = len(i)
            res_str = ""
        str1 = "("
        for j in i:
            str1 += '\'' + j + '\'' + ", "
        str1 = str1[:-2] + "),"
        res_str += str1
    res_str = res_str[:-1]
    f.write(res_str)

end = time.time() - start  ##stop timing!!
print("Duration:",end)
