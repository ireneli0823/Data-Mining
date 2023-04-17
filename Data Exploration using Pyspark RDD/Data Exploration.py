from pyspark import SparkContext 
import os
import json
import sys
os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'
print('cmd entry:',sys.argv)

sc = SparkContext('local[*]','review_num')
sc.setLogLevel("WARN")
file_path = sys.argv[1]
reviewRDD = sc.textFile(file_path).map(lambda x: json.loads(x))


# The total number of reviews
review_num = reviewRDD.count()
ans = review_num


# The number of reviews in 2018
review_2018 = reviewRDD.filter(lambda x: '2018' in x['date'])
num_2018 = review_2018.count()
ans = num_2018


# The number of distinct users who wrote reviews
reviewRDD1 = reviewRDD.map(lambda x: (x['user_id'],1))
distinct_user = reviewRDD1.reduceByKey(lambda x,y: x+y)
ans_c = distinct_user.count()


# The top 10 users who wrote the largest numbers of reviews and the number of reviews they wrote
distinct_user_top10 = distinct_user.sortBy(lambda x: [-x[1], x[0]])
ans_d = distinct_user_top10.take(10)


# The number of distinct businesses that have been reviewed
viewed_business = reviewRDD.map(lambda x: (x["business_id"],1))
distinct_business = viewed_business.reduceByKey(lambda x,y: x+y)
ans_e = distinct_business.count()


# The top 10 businesses that had the largest numbers of reviews and the number of reviews they had
distinct_business_top10 = distinct_business.sortBy(lambda x: [-x[1], x[0]])
ans_f = distinct_business_top10.take(10)


ans_dict = {
    "n_review": review_num,
    "n_review_2018": num_2018,
    "n_user": ans_c,
    "top10_user": ans_d,
    "n_business":ans_e,
    "top10_business": ans_f
}
json_obj = json.dumps(ans_dict)
with open(sys.argv[2],"w") as f:
    f.write(json_obj)


