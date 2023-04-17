import csv
from math import sqrt

true_results = list()
with open('../resource/asnlib/publicdata/yelp_val.csv','r') as f1:
    reader1 = csv.reader(f1)
    next(reader1)

    for row in reader1:
        true_results.append(((row[0],row[1]),row[2]))

true_results = dict(true_results)

RMSE = 0
n = 0
minp = 10
maxp = -1
predict_results = list()
with open('task2_3_out.csv','r') as f2:
    reader2 = csv.reader(f2)
    next(reader2)
    for row in reader2:
        key = (row[0],row[1])
        Pred = float(row[2])
        Rate = float( true_results[key] )
        RMSE += (Pred-Rate)*(Pred-Rate)
        n+=1

'''
for i in range(len(predict_results)):

    Pred = float(predict_results[i][2])
    Rate = float(true_results[i][2])
    minp = min(minp, Pred)
    maxp = max(maxp, Pred)
    #print("diff:",(Pred-Rate),n)
    RMSE += (Pred-Rate)*(Pred-Rate)
    n += 1
'''
RMSE = sqrt(RMSE/n)
print("RMSE:",RMSE)
print("max",maxp)
print("min",minp)

www = "jskdKNDS"
print(www)
www = www.lower()
print(www)