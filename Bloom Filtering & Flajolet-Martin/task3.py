from blackbox import BlackBox
#from pyspark import SparkContext
import os
import sys
import time
import binascii
import  random


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

random.seed(553)


with open(outputfile, "w") as file:
    file.write("seqnum,0_id,20_id,40_id,60_id,80_id\n")
    bx = BlackBox()
    seqnum = 0
    stream_users = bx.ask(inputfile, stream_size)
    line = str(seqnum+100)+","+stream_users[0]+","+stream_users[20]+","+stream_users[40]+","+stream_users[60]+","+stream_users[80]
    file.write(line+"\n")
    seqnum += 100
    print(stream_users)
    for k in range(1,num_of_asks):
        stream_users = bx.ask(inputfile,stream_size)
        line = str(seqnum + 100) + "," + stream_users[0] + "," + stream_users[20] + "," + stream_users[40] + "," + \
               stream_users[60] + "," + stream_users[80]
        file.write(line + "\n")
        print(stream_users)
        seqnum += 100



end_time = time.time() - start
print("Duration:",end_time)