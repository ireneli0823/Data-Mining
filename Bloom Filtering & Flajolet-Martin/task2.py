from blackbox import BlackBox
#from pyspark import SparkContext
import os
import sys
import time
import binascii
import  random

def myhashs(s):
    result = []
    int_s = int(binascii.hexlify(s.encode('utf8')),16)
    for param in hash_parameters:
        result.append(  ((param[0]*int_s + param[1])%p)%m )

    return result

def count_trailing_zeros(n):
    count = 0
    while n>0 and n%2==0:
        count += 1
        n = n//2

    return count

def get_median(numbers):
    numbers.sort()
    length = len(numbers)
    if length%2==1:
        return numbers[length//2]
    else:
        return (numbers[length//2-1] + numbers[length//2])/2

start = time.time()

inputfile = sys.argv[1]
stream_size = int(sys.argv[2])
num_of_asks = int(sys.argv[3])
outputfile = sys.argv[4]

number_of_hashes = 300
m = 2**32
p = 4294967311#69997
hash_parameters = []
for i in range(number_of_hashes):
    a = random.randint(0, m-1)
    b = random.randint(0, m-1)
    hash_parameters.append((a,b))


bx = BlackBox()
with open(outputfile, "w") as file:
    file.write("Time,Ground Truth,Estimation\n")
    sum_truth = 0
    sum_estimation = 0
    for k in range(num_of_asks):
        stream_users = bx.ask(inputfile,stream_size)
        users_hashes = []
        for user in stream_users:
            users_hashes.append(myhashs(user))
        sum = 0
        maxRs = []
        for j in range(number_of_hashes):
            R = 0
            for i in range(len(stream_users)):
                R = max(R,count_trailing_zeros(users_hashes[i][j]))
            sum += 2**R
            maxRs.append(2**R)
        #print(sum/number_of_hashes)
        averages = []
        index = 0
        group_size = 5
        #maxRs.sort()
        while index<number_of_hashes:
            sum1 = 0
            for i in range(group_size):
                sum1 += maxRs[index+i]
            averages.append(sum1/group_size)
            index += group_size

        truth = len(set(stream_users))
        estimation = round(get_median(averages))
        sum_truth += truth
        sum_estimation += estimation
        file.write(str(k)+","+str(truth)+","+str(estimation)+"\n")

file.close()

print("truth",sum_truth)
print("estimation",sum_estimation)

end_time = time.time() - start
print("Duration:",end_time)