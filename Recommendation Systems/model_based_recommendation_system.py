from pyspark import SparkContext
import os
import time
import xgboost as xgb
import sys
import csv
import json
import pandas as pd
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
#reverse_business_index_dict = dict()
#for key in business_index_dict.keys():
#    reverse_business_index_dict[business_index_dict[key]] = key


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

with open(out_file_path,'w') as f:
    writer = csv.writer(f)
    header = ['user_id','business_id','prediction']
    writer.writerow(header)
    for v in all_result_list:
        writer.writerow(v)

#print("RMSE : ", np.sqrt(mean_squared_error(test_df.stars.values, prediction)))

end_time = time.time() - start

#print(end_time)