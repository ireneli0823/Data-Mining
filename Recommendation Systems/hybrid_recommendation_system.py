from math import sqrt
from pyspark import SparkContext
import os
import time
import xgboost as xgb
import sys
import csv
import json
import pandas as pd #111
#import numpy as np

# start timing!
start = time.time()

os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'
print('cmd entry:',sys.argv)
file_path = sys.argv[1]+"/yelp_train.csv"
json_path = sys.argv[1]+"/user.json"
business_json_path = sys.argv[1]+"/business.json"
out_file_path = sys.argv[3]
test_file_path = sys.argv[2]

sc = SparkContext('local[*]','HW3_task2_2')
sc.setLogLevel("WARN")

def getPearsonCorrelation(business_id1,business_id2):
    business_users_ratings_list1 = business_users_ratings_dict[business_id1]
    business_users_ratings_list2 = business_users_ratings_dict[business_id2]
    avg1_all = 0
    for user_rating in business_users_ratings_list1:
        avg1_all += user_rating[1]
    avg1_all = avg1_all/len(business_users_ratings_list1)
    avg2_all = 0
    for user_rating in business_users_ratings_list2:
        avg2_all += user_rating[1]
    avg2_all = avg2_all/len(business_users_ratings_list2)
    business_users_ratings_dict1 = dict(business_users_ratings_list1)
    business_users_ratings_dict2 = dict(business_users_ratings_list2)
    coUsers = set()
    for user in business_users_ratings_dict1:
        if user in business_users_ratings_dict2:
            coUsers.add(user)
    if len(coUsers)==0:
        res = avg2_all/avg1_all
        if res > 1:
            res = 1/res
        return res
    avg1_co = 0
    avg2_co = 0
    for user in coUsers:
        avg1_co += business_users_ratings_dict1[user]
        avg2_co += business_users_ratings_dict2[user]
    avg1_co = avg1_co/len(coUsers)
    avg2_co = avg2_co/len(coUsers)
    Numerator = 0
    Denominator1 = 0
    Denominator2 = 0
    for user in coUsers:
        r1 = business_users_ratings_dict1[user]
        r2 = business_users_ratings_dict2[user]
        diff1 = r1 - avg1_co
        diff2 = r2 - avg2_co
        Numerator += diff1*diff2
        Denominator1 += diff1*diff1
        Denominator2 += diff2*diff2
    Denominator = sqrt(Denominator1*Denominator2)
    if Denominator==0:
        return 1

    return Numerator/Denominator
# makePredict
def makePredict(ite):
    partial_list = list(ite)
    predict_result = list()
    for row in partial_list:
        user_name = row[0]
        if user_name in user_index_dict:
            user_id = user_index_dict[user_name]
            user_businesses_ratings_list = user_businesses_ratings_dict[user_id]
            avg_user_businesses = 0
            for user_business_rating in user_businesses_ratings_list:
                avg_user_businesses += user_business_rating[1]
            avg_user_businesses = avg_user_businesses/len(user_businesses_ratings_list)
        business_name = row[1]
        if business_name in business_index_dict:
            business_id = business_index_dict[business_name]
            business_users_ratings_list = business_users_ratings_dict[business_id]
            avg_business_users = 0
            for business_user_rating in business_users_ratings_list:
                avg_business_users += business_user_rating[1]
            avg_business_users = avg_business_users/len(business_users_ratings_list)
        #cold start
        if (user_name not in user_index_dict) and (business_name not in business_index_dict):
            p = 3
        elif user_name not in user_index_dict:
            p = avg_business_users
        elif business_name not in business_index_dict:
            p = avg_user_businesses
        else:
            #print(user_businesses_ratings_list)
            Numerator = 0
            Denominator = 0
            weight_rate_list = list()
            for user_business_rating in user_businesses_ratings_list:
                neighbor_id = user_business_rating[0]
                r = user_business_rating[1]
                w = getPearsonCorrelation(business_id,neighbor_id)
                if w>0:
                    weight_rate_list.append((w,r))
            weight_rate_list.sort(key=lambda x:x[0],reverse=True)
            N = 100
            for i in range(min(len(weight_rate_list),N)):
                w = weight_rate_list[i][0]
                r = weight_rate_list[i][1]
                Numerator += w*r
                Denominator += abs(w)

            if Denominator==0:
                p = avg_user_businesses
            else:
                p = Numerator/Denominator

        predict_result.append((user_name,business_name,p))

    yield predict_result

def average(ite):
    ite_sum = 0
    num = len(ite)
    for i in ite:
        ite_sum += i
    return (ite_sum/num)

def replce_iregular_id(ite):
    res = [ite[0],ite[1],0,0,0,0,0,0]
    if ite[0] in user_index_dict:
        res[0] = user_index_dict[ite[0]]
    else:
        res[0] = -1

    if ite[0] in user_js_dict:
        res[5] = user_js_dict[ite[0]][0]
        res[6] = user_js_dict[ite[0]][3]
        res[7] = user_js_dict[ite[0]][1]
    else:
        res[5] = avg_user_rated_number
        res[6] = avg_user_average_star
        res[7] = 0

    if ite[1] in business_index_dict:
        res[1] = business_index_dict[ite[1]]
    else:
        res[1] = -1

    if ite[1] in business_js_dict:
        res[3] = business_js_dict[ite[1]][1]
        res[4] = business_js_dict[ite[1]][0]
    else:
        res[3] = avg_business_rated_number
        res[4] = avg_business_average_star

    return res



# load csv data to RDD
RDD = sc.textFile(file_path)
header = RDD.first()
rawRDD = RDD.filter((lambda x: x != header)).map(lambda x: x.split(","))

# split to different RDD
business_num_rate_dict = rawRDD.map(lambda x: (x[1],x[2])).groupByKey().mapValues(len).collectAsMap()
user_num_rate_dict = rawRDD.map(lambda x: (x[0],x[2])).groupByKey().mapValues(len).collectAsMap()

business_avg_rate_dict = rawRDD.map(lambda x: (x[1],float(x[2]))).groupByKey().mapValues(average).collectAsMap()
user_avg_rate_dict = rawRDD.map(lambda x: (x[0],float(x[2]))).groupByKey().mapValues(average).collectAsMap()


user_index_dict = rawRDD.map(lambda x: x[0]).distinct().sortBy(lambda x: x).zipWithIndex().collectAsMap()
business_index_dict = rawRDD.map(lambda x: x[1]).distinct().sortBy(lambda x: x).zipWithIndex().collectAsMap()

user_business_RDD = rawRDD.map(lambda x:(x[0],(x[1],float(x[2]))))
business_user_RDD = rawRDD.map(lambda x:(x[1],(x[0],float(x[2]))))
user_business_RDD = user_business_RDD.map(lambda x:(user_index_dict[x[0]],x[1]))\
    .mapValues(lambda x:(business_index_dict[x[0]],x[1]))
business_user_RDD = business_user_RDD.map(lambda x:(business_index_dict[x[0]],x[1]))\
    .mapValues(lambda x:(user_index_dict[x[0]],x[1]))
user_businesses_ratings_RDD = user_business_RDD.groupByKey().mapValues(list)
user_businesses_ratings_dict = dict(user_businesses_ratings_RDD.collect())
business_users_ratings_RDD = business_user_RDD.groupByKey().mapValues(list)
business_users_ratings_dict = dict(business_users_ratings_RDD.collect())
#reverse_business_index_dict = dict()
#for key in business_index_dict.keys():
#    reverse_business_index_dict[business_index_dict[key]] = key
test_list = list()
with open(test_file_path,'r') as f2:
    reader2 = csv.reader(f2)
    next(reader2)
    for row in reader2:
        test_list.append((row[0],row[1]))

#get final result
test_RDD = sc.parallelize(test_list).mapPartitions(makePredict).flatMap(lambda x:x).map(lambda x:((x[0],x[1]),x[2]))
final_reults = dict(test_RDD.collect())

# load json data to RDD
user_js_RDD = sc.textFile(json_path).map(lambda x: json.loads(x))\
    .map(lambda x: ((x['user_id']), (x['review_count'], x['useful'], x['fans'], x['average_stars'])))

user_index_dict = user_js_RDD.map(lambda x: x[0]).distinct().sortBy(lambda x: x).zipWithIndex().collectAsMap()
#reverse_user_index_dict = dict()
#for key in user_index_dict:
#    reverse_user_index_dict[user_index_dict[key]] = key
user_js_dict = user_js_RDD.collectAsMap()
#print(reverse_user_index_dict)

business_js_RDD = sc.textFile(business_json_path).map(lambda x: json.loads(x)).map(lambda x: ((x["business_id"]),((x["stars"],x["review_count"]))))
business_js_dict = business_js_RDD.collectAsMap()


processed_data_RDD = rawRDD\
    .map(lambda x: (user_index_dict[x[0]],business_index_dict[x[1]],float(x[2]),business_num_rate_dict[x[1]],business_avg_rate_dict[x[1]],user_num_rate_dict[x[0]],user_avg_rate_dict[x[0]],user_js_dict[x[0]][1]))

train_dataset = processed_data_RDD.collect()

df = pd.DataFrame(train_dataset, columns = ['user_idx', 'business_idx','stars','business_rated_number','business_average_star','user_rated_number','user_average_star','user_review_useful'])


x_train = df.drop('stars',axis = 1)
y_train = df.stars.values



model = xgb.XGBRegressor(learning_rate=0.05)
model.fit(x_train, y_train)

# deal with outliers！
avg_business_rated_number = business_js_RDD.map(lambda x:x[1][1]).mean()
avg_user_rated_number = user_js_RDD.map(lambda x: x[1][0]).mean()
avg_business_average_star = business_js_RDD.map(lambda x: x[1][0]).mean()
avg_user_average_star = user_js_RDD.map(lambda x: x[1][-1]).mean()



# test RDD
test_RDD = sc.textFile(test_file_path)
test_header = test_RDD.first()
test_RDD = test_RDD.filter((lambda x: x != test_header)).map(lambda x: x.split(","))
test_RDD_iregular = test_RDD.map(replce_iregular_id)
test_RDD_list = test_RDD.collect()

test_dataset = test_RDD_iregular.collect()
test_df = pd.DataFrame(test_dataset, columns = ['user_idx', 'business_idx','stars','business_rated_number','business_average_star','user_rated_number','user_average_star','user_review_useful'])



x_test = test_df.drop(['stars'], axis=1)#.values
prediction = model.predict(x_test)

result = prediction.tolist()
all_result_list = []
for i in range(len(result)):
    tmp = [test_RDD_list[i][0], test_RDD_list[i][1], result[i]]
    #res = [reverse_user_index_dict[tmp[0]], reverse_business_index_dict[tmp[1]], tmp[2]]
    #print("tmp",tmp)
    #print("res",res)
    all_result_list.append(tmp)


alpha = 0.21
with open(out_file_path,'w') as f:
    writer = csv.writer(f)
    header = ['user_id','business_id','prediction']
    writer.writerow(header)
    for v in all_result_list:
        key = (v[0], v[1])
        pred1 = final_reults[key]
        pred2 = v[2]
        row = [v[0], v[1], alpha*pred1 + (1-alpha)*pred2]
        writer.writerow(row)

#print("RMSE : ", np.sqrt(mean_squared_error(test_df.stars.values, prediction)))

end_time = time.time() - start

#print(end_time)