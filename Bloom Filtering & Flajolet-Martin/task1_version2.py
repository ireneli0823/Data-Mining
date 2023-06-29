import binascii
import random
import csv
from blackbox import BlackBox

bit_array_length = 69997
stream_size = 100
num_of_asks = 100
in_file_name = "users.txt"
out_file_name = "output.csv"


bx = BlackBox()


def user_id_to_number(uid):
    return int(binascii.hexlify(uid.encode('utf8')), 16)

a = [random.randint(1, bit_array_length) for _ in range(5)]
b = [random.randint(0, bit_array_length) for _ in range(5)]

def myhashs(uid):
    result = []
    user_id_num = user_id_to_number(uid)

    p = 999331  # prime number

    for i in range(3):
        result.append(((a[i] * user_id_num + b[i]) % p) % bit_array_length)
    return result

# get all streams in a list
all_stream = list()
for i in range(num_of_asks):
    stream_users = bx.ask(in_file_name, stream_size)
    all_stream.append(stream_users)

seen_user = set()
bit_array = bytearray(bit_array_length)
FPR_ans = list()
Time = 0
for stream in all_stream:
    FP = 0
    TN = 0
    TP = 0
    FN = 0
    size = stream_size
    tmp_stream_list = []
    for i in stream:
        tmp_stream_list.append((i,myhashs(i)))
    #print(tmp_stream_list)
    #print(len(tmp_stream_list))
    for (x, y) in tmp_stream_list:
        if all(bit_array[i] == 1 for i in y) and x not in seen_user:
            #print(all(bit_array[i] == 1 for i in y))
            FP += 1
        elif (all(bit_array[i] == 1 for i in y) == False) and x not in seen_user:
            #print(all(bit_array[i] == 1 for i in y))
            TN += 1
            for i in y:
                bit_array[i] = 1
        elif all(bit_array[i] == 1 for i in y) and x in seen_user:
            TP += 1
        elif (all(bit_array[i] == 1 for i in y) == False) and x in seen_user:
            FN += 1
        else:
            for i in y:
                bit_array[i] = 1
        seen_user.add(x)
    #print(FP," ",TN," ",FN," ",TP)
    FPR = float(FP / size)
    FPR_ans.append((Time, FPR))
    Time += 1
print(FPR_ans)




with open(out_file_name, "w", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(["Time", "FPR"])
    csv_writer.writerows(FPR_ans)
