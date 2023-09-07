"""
Method description
to improve the accuracy and efficiency for this recommendation system, I add lots of features. To decide which feature should be added into the XGBoost, I calculated their importance and add the feature with larger importance for this model. Besides, in order to avoid overfitting, I divided the train dataset into 5 groups and train the model with 4 groups and use the other one to test its accuracy. This process will be did 5 times to avoid accuracy. Then we used stacking to improve its accuracy.

Error Distribution:
>=0 and <1: 102250
>=1 and <2: 32835
>=2 and <3: 6149
>=3 and <4: 810
>=4: 0


RMSE:
0.9792574100552962

Execution Time:
982s

"""

import numpy as np
from math import sqrt
import pandas as pd
import time
import json
import os
import sys
from pyspark import SparkContext
import ast
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
import xgboost as xgb
from catboost import CatBoostRegressor
import csv

# start timing!
start = time.time()

os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

# print('cmd entry:',sys.argv)
train_file_path = sys.argv[1] + "/yelp_train.csv"
json_path = sys.argv[1] + "/user.json"
business_json_path = sys.argv[1] + "/business.json"
out_file_path = sys.argv[3]
test_file_path = sys.argv[2]

sc = SparkContext('local[*]', 'XGBoost_recommendation')
sc.setLogLevel("WARN")


# load user json data to RDD

def get_friends_num(ite):
    if ite == 'None':
        return 0
    else:
        return len(ite.split(","))


def yelp_since(ite):
    return hash(ite)


user_js_RDD = sc.textFile(json_path).map(lambda x: json.loads(x)) \
    .map(lambda x: (
    (x['user_id']),
    (x['review_count'],
     yelp_since(x["yelping_since"]),
     get_friends_num(x["friends"]),
     x['useful'],
     x["funny"],
     x["cool"],
     x['fans'],
     x['average_stars'],
     x["compliment_photos"]
     )
))

user_js_dict = user_js_RDD.collectAsMap()


# load business json data to RDD
def postal_code(ite):
    if ite is None:
        return 0
    else:
        return hash(ite)


def attribute_preprocess(ite):
    tmp = []
    if ite is not None:
        if "BusinessAcceptsCreditCards" in ite:
            if ite["BusinessAcceptsCreditCards"] == "True":
                tmp.append(1)
            else:
                tmp.append(0)
        else:
            tmp.append(0)

        if "BusinessParking" in ite:
            tmp_dict = ast.literal_eval(ite["BusinessParking"])
            if "street" in tmp_dict:
                tmp.append(int(tmp_dict["street"]))
            else:
                tmp.append(0)
            if "validated" in tmp_dict:
                tmp.append(int(tmp_dict["validated"]))
            else:
                tmp.append(0)
            if "lot" in tmp_dict:
                tmp.append(int(tmp_dict["lot"]))
            else:
                tmp.append(0)
            if "valet" in tmp_dict:
                tmp.append(int(tmp_dict["valet"]))
            else:
                tmp.append(0)
        else:
            tmp.append(0)
            tmp.append(0)
            tmp.append(0)
            tmp.append(0)

        if "DogsAllowed" in ite:
            if ite["DogsAllowed"] == "True":
                tmp.append(1)
            else:
                tmp.append(0)
        else:
            tmp.append(0)

        if "DriveThru" in ite:
            if ite["DriveThru"] == "True":
                tmp.append(1)
            else:
                tmp.append(0)
        else:
            tmp.append(0)

        if "GoodForKids" in ite:
            if ite["GoodForKids"] == "True":
                tmp.append(1)
            else:
                tmp.append(0)
        else:
            tmp.append(0)

        if "HasTV" in ite:
            if ite["HasTV"] == "True":
                tmp.append(1)
            else:
                tmp.append(0)
        else:
            tmp.append(0)

        if "OutdoorSeating" in ite:
            if ite["OutdoorSeating"] == "True":
                tmp.append(1)
            else:
                tmp.append(0)
        else:
            tmp.append(0)

        if "RestaurantsAttire" in ite:
            tmp.append(hash(ite["RestaurantsAttire"]))
        else:
            tmp.append(0)

        if "RestaurantsDelivery" in ite:
            if ite["RestaurantsDelivery"] == "True":
                tmp.append(1)
            else:
                tmp.append(0)
        else:
            tmp.append(0)

        if "RestaurantsGoodForGroups" in ite:
            if ite["RestaurantsGoodForGroups"] == "True":
                tmp.append(1)
            else:
                tmp.append(0)
        else:
            tmp.append(0)

        if "RestaurantsPriceRange2" in ite:
            tmp.append(int(ite["RestaurantsPriceRange2"]))
        else:
            tmp.append(0)

        if "RestaurantsReservations" in ite:
            if ite["RestaurantsReservations"] == "True":
                tmp.append(1)
            else:
                tmp.append(0)
        else:
            tmp.append(0)

        if "RestaurantsTableService" in ite:
            if ite["RestaurantsTableService"] == "True":
                tmp.append(1)
            else:
                tmp.append(0)
        else:
            tmp.append(0)

        if "RestaurantsTakeOut" in ite:
            if ite["RestaurantsTakeOut"] == "True":
                tmp.append(1)
            else:
                tmp.append(0)
        else:
            tmp.append(0)

        if "WheelchairAccessible" in ite:
            if ite["WheelchairAccessible"] == "True":
                tmp.append(1)
            else:
                tmp.append(0)
        else:
            tmp.append(0)

        if "WiFi" in ite:
            if ite["WiFi"] == "free":
                tmp.append(1)
            else:
                tmp.append(0)
        else:
            tmp.append(0)

        if "NoiseLevel" in ite:
            tmp.append(hash(ite["NoiseLevel"]))
        else:
            tmp.append(0)

        if "ByAppointmentOnly" in ite:
            if ite["ByAppointmentOnly"] == "True":
                tmp.append(1)
            else:
                tmp.append(0)

        if "Smoking" in ite:
            tmp.append(hash(ite["Smoking"]))
        else:
            tmp.append(0)

        if "HappyHour" in ite:
            if ite["HappyHour"] == "True":
                tmp.append(1)
            else:
                tmp.append(0)

        if "CoatCheck" in ite:
            if ite["CoatCheck"] == "True":
                tmp.append(1)
            else:
                tmp.append(0)

        if "Music" in ite:
            tmp.append(1)
        else:
            tmp.append(0)

        if "Caters" in ite:
            if ite["Caters"] == "True":
                tmp.append(1)
            else:
                tmp.append(0)
        if "Alcohol" in ite:
            tmp.append(hash(ite["Alcohol"]))
        else:
            tmp.append(0)

        if "GoodForMeal" in ite:
            tmp_dict = ast.literal_eval(ite["GoodForMeal"])
            if "dessert" in tmp_dict:
                tmp.append(int(tmp_dict["dessert"]))
            else:
                tmp.append(0)
            if "latenight" in tmp_dict:
                tmp.append(int(tmp_dict["latenight"]))
            else:
                tmp.append(0)
            if "lunch" in tmp_dict:
                tmp.append(int(tmp_dict["lunch"]))
            else:
                tmp.append(0)
            if "dinner" in tmp_dict:
                tmp.append(int(tmp_dict["dinner"]))
            else:
                tmp.append(0)
            if "brunch" in tmp_dict:
                tmp.append(int(tmp_dict["brunch"]))
            else:
                tmp.append(0)
            if "breakfast" in tmp_dict:
                tmp.append(int(tmp_dict["breakfast"]))
            else:
                tmp.append(0)

        else:
            tmp.append(0)
            tmp.append(0)
            tmp.append(0)
            tmp.append(0)
            tmp.append(0)
            tmp.append(0)

    return tmp


def category_preprocess(ite):
    if ite is not None:
        tmp = ite.split(",")
        if "Restaurants" in tmp:
            return 1
        elif "Food" in tmp:
            return 1
        elif "Health & Medical" in tmp:
            return 2
        else:
            return 0
    else:
        return 1


business_js_RDD = sc.textFile(business_json_path).map(lambda x: json.loads(x)) \
    .map(lambda x: (
    x["business_id"],
    (postal_code(x["postal_code"]),
     x["stars"],
     x["review_count"],
     x["is_open"],
     category_preprocess(x["categories"]),
     attribute_preprocess(x["attributes"])
     )))

business_js_dict = business_js_RDD.collectAsMap()


# print(business_js_dict["AjEbIBw6ZFfln7ePHha9PA"])

def changeToList(tup):
    res = list()
    res.append(tup[0])
    for i in tup[1]:
        res.append(i)
    for i in tup[2][:5]:
        res.append(i)
    for i in tup[2][5]:
        res.append(i)
    return res


# load train data to RDD
RDD = sc.textFile(train_file_path)
header = RDD.first()
trainRDD = RDD.filter((lambda x: x != header)).map(lambda x: x.split(","))
# print(trainRDD.take(5))
trainRDD = trainRDD.map(lambda x: (float(x[2]), user_js_dict[x[0]], business_js_dict[x[1]]))
# print(trainRDD.take(5))

# Apply the lambda function to the RDD using flatMap()
flattenedRDD = trainRDD.map(changeToList)
# print(flattenedRDD.take(5))


train_dataset = flattenedRDD.collect()
df = pd.DataFrame(
    train_dataset,
    columns=['stars',
             'review_count',
             'yelping_since',
             'friends',
             'useful',
             'funny',
             'cool',
             'fans',
             'average_stars',
             'compliment_photos',
             'bus_postal_code',
             'bus_avg_stars',
             'bus_review_count',
             'bus_is_open',
             'categories',
             'BusinessAcceptsCreditCards',
             'street',
             'validated',
             'lot',
             'valet',
             'DogsAllowed',
             'DriveThru',
             'GoodForKids',
             'HasTV',
             'OutdoorSeating',
             'RestaurantsAttire',
             'RestaurantsDelivery',
             'RestaurantsGoodForGroups',
             'RestaurantsPriceRange2',
             'RestaurantsReservations',
             'RestaurantsTableService',
             'RestaurantsTakeOut',
             'WheelchairAccessible',
             'WiFi',
             'NoiseLevel',
             "ByAppointmentOnly",
             "Smoking",
             "HappyHour",
             "CoatCheck",
             "Music",
             'Caters',
             'Alcohol',
             'dessert',
             'latenight',
             'lunch',
             'dinner',
             'brunch',
             'breakfast'
             ]
)

train_drop = []
train_df = df.drop(train_drop, axis=1)
# x_train = train_df.drop(train_drop,axis = 1)
# y_train = train_df.stars.values

# deal with outliers！
avg_bus_review_count = df.bus_review_count.mean()
avg_bus_avg_stars = df.bus_avg_stars.mean()
avg_review_count = df.review_count.mean()
avg_average_stars = df.average_stars.mean()


# load test data to RDD
def fill_missing_user(x):
    if x not in user_js_dict:
        return (avg_review_count, 0, 0, 0, 0, 0, 0, avg_average_stars, 0)
    else:
        return user_js_dict[x]


def fill_missing_business(x):
    if x not in business_js_dict:
        return (0, avg_bus_avg_stars, avg_bus_review_count, 0, 1, [0] * 33)
    else:
        return business_js_dict[x]


def combineToList(ite):
    tmp = list()
    for i in ite[0]:
        tmp.append(i)
    for i in ite[1]:
        tmp.append(i)
    return tmp


test_RDD = sc.textFile(test_file_path)
test_header = test_RDD.first()
test_RDD = test_RDD.filter((lambda x: x != test_header)).map(lambda x: x.split(","))
# print(test_RDD.take(5))
test_processed_RDD = test_RDD.map(lambda x: (float(x[2]), x[0], x[1]))
test_feature_RDD = test_processed_RDD \
    .map(lambda x: ((x[1], x[2]), (x[0], fill_missing_user(x[1]), fill_missing_business(x[2])))) \
    .map(lambda x: (x[0], changeToList(x[1]))).map(combineToList)
# print(test_processed_RDD.take(5))
# print(test_feature_RDD.take(5))

test_dataset = test_feature_RDD.collect()

test_df = pd.DataFrame(
    test_dataset,
    columns=['user_id',
             'business_id',
             'stars',
             'review_count',
             'yelping_since',
             'friends',
             'useful',
             'funny',
             'cool',
             'fans',
             'average_stars',
             'compliment_photos',
             'bus_postal_code',
             'bus_avg_stars',
             'bus_review_count',
             'bus_is_open',
             'categories',
             'BusinessAcceptsCreditCards',
             'street',
             'validated',
             'lot',
             'valet',
             'DogsAllowed',
             'DriveThru',
             'GoodForKids',
             'HasTV',
             'OutdoorSeating',
             'RestaurantsAttire',
             'RestaurantsDelivery',
             'RestaurantsGoodForGroups',
             'RestaurantsPriceRange2',
             'RestaurantsReservations',
             'RestaurantsTableService',
             'RestaurantsTakeOut',
             'WheelchairAccessible',
             'WiFi',
             'NoiseLevel',
             "ByAppointmentOnly",
             "Smoking",
             "HappyHour",
             "CoatCheck",
             "Music",
             'Caters',
             'Alcohol',
             'dessert',
             'latenight',
             'lunch',
             'dinner',
             'brunch',
             'breakfast'
             ]
)
# print(test_df.head())

test_drop = ['user_id', 'business_id']
test_df = test_df.drop(test_drop, axis=1)

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

base_models = [xgb.XGBRegressor(learning_rate=0.1, gamma=1, max_depth=5, n_estimators=200),
               CatBoostRegressor(iterations=100, depth=6, verbose=False, random_state=42)]

oof_train = np.zeros((train_df.shape[0], len(base_models)))
oof_test = np.zeros((test_df.shape[0], len(base_models)))

for idx, model in enumerate(base_models):
    print("Model {}".format(idx))
    print("idx", idx)
    oof_test_temp = np.zeros((test_df.shape[0], n_splits))
    # print("oof_test_temp shape[0]",oof_test_temp.shape[0])
    # print("oof_test_temp",oof_test_temp)
    for fold, (train_index, valid_index) in enumerate(kf.split(train_df)):
        print("Fold {}".format(fold + 1))
        X_train_fold = train_df.iloc[train_index].drop(columns='stars')
        y_train_fold = train_df.iloc[train_index]['stars']
        X_valid_fold = train_df.iloc[valid_index].drop(columns='stars')

        model.fit(X_train_fold, y_train_fold)
        # print("model.fit")
        oof_train[valid_index, idx] = model.predict(X_valid_fold)

        oof_test_temp[:, fold] = model.predict(test_df.drop(columns='stars'))

    oof_test[:, idx] = oof_test_temp.mean(axis=1)

# Train the meta-model using the OOF predictions from the base models
meta_model = LinearRegression()
meta_model.fit(oof_train, train_df['stars'])

stacked_preds = meta_model.predict(oof_test)

result = stacked_preds.tolist()
all_result_list = []
for i in range(len(result)):
    tmp = [test_dataset[i][0], test_dataset[i][1], result[i]]
    all_result_list.append(tmp)

with open(out_file_path, 'w') as f:
    writer = csv.writer(f)
    header = ['user_id', 'business_id', 'prediction']
    writer.writerow(header)
    for v in all_result_list:
        writer.writerow(v)

end_time = time.time() - start
print(end_time)