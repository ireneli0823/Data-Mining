from pyspark import SparkContext
import os
import itertools
import sys
import time

start = time.time()
print('cmd entry:',sys.argv)
sc = SparkContext('local[*]', 'HW4_task2')
sc.setLogLevel("WARN")



#

threshold = int(sys.argv[1]) #7
input_file = sys.argv[2] #"ub_sample_data.csv"
output_file1 = sys.argv[3] #"out_task1.txt"
output_file2 = sys.argv[4]

RDD = sc.textFile(input_file)
header = RDD.first()
RDD = RDD.filter((lambda x: x != header))

# load rdd to (user_id,business_id) format
user_businesses_dict = RDD.map(lambda x: x.split(",")).map(lambda x: (x[0], x[1])).groupByKey().mapValues(list)\
    .map(lambda x: (x[0], sorted(x[1]))).collectAsMap()


#get vertices and edges
user_list = list(user_businesses_dict.keys())
vertices = set()
edges = list()
num_users = len(user_list)
for i in range(num_users):
    for j in range(i+1,num_users):
        set_0 = set(user_businesses_dict[user_list[i]])
        set_1 = set(user_businesses_dict[user_list[j]])
        num_intersect = set_1.intersection(set_0)
        if len(num_intersect) >= threshold:
            vertices.add(user_list[i])
            vertices.add(user_list[j])
            if user_list[i] < user_list[j]:
                edges.append((user_list[i], user_list[j]))
            else:
                edges.append((user_list[j], user_list[i]))

#costruct graph wit veritex and it's neighbor vertices
vertices_neighbors = dict()
for edge in edges:
    v0 = edge[0]
    v1 = edge[1]
    if v0 in vertices_neighbors:
        neighbors0 = vertices_neighbors[v0]
    else:
        neighbors0 = set()

    if v1 in vertices_neighbors:
        neighbors1 = vertices_neighbors[v1]
    else:
        neighbors1 = set()

    neighbors0.add(v1)
    neighbors1.add(v0)

    vertices_neighbors[v0] = neighbors0
    vertices_neighbors[v1] = neighbors1


#calculate edge betweeness
edge_betweeness = dict()
for root in sorted(vertices):

    #consturct tree with BFS form root node
    queue = []
    queue.append((root,0))
    ans = list()

    prevset = set()
    currset = set()
    currset.add(root)
    currlevel = -1

    while len(queue)!=0:
        curr = queue.pop(0)
        level = curr[1]
        curr = curr[0]
        if level!=currlevel:
            currlevel = level
            prevset = prevset.union(currset)
            currset.clear()
        if level>=len(ans):
            ans.append(dict())
        if curr not in ans[level]:
            tmplist = list()
            tmpprevset = set()
            tmpnextset = set()
            tmplist.append(tmpprevset)
            tmplist.append(tmpnextset)
            tmplist.append(1)
            tmplist.append(1)
            ans[level][curr] = tmplist

        if level == 0:
            ans[level][curr][3] == 1
        else:
            tmpsum = 0
            for prev in ans[level][curr][0]:
                tmpsum += ans[level-1][prev][3]
            ans[level][curr][3] = tmpsum


        neighbors = vertices_neighbors[curr]
        for neighbor in neighbors:
            if neighbor not in prevset:
                if level+1 >= len(ans):
                    ans.append(dict())

                if neighbor not in ans[level+1]:
                    tmplist = list()
                    tmpprevset = set()
                    tmpnextset = set()
                    tmplist.append(tmpprevset)
                    tmplist.append(tmpnextset)
                    tmplist.append(1)
                    tmplist.append(1)
                    ans[level+1][neighbor] = tmplist

                queue.append((neighbor,level+1))
                ans[level][curr][1].add(neighbor)
                ans[level+1][neighbor][0].add(curr)
                currset.add(neighbor)
    total_levels = len(ans)
    edges_bet_tmp = dict()
    #tree constructed

    #caculate betweenness with constructed tree
    for i in range(total_levels-1,0,-1):
        for currNode in ans[i]:
            credit = ans[i][currNode][2]
            prevNodes = ans[i][currNode][0]
            total_num_paths = 0

            for prevNode in prevNodes:
                total_num_paths += ans[i-1][prevNode][3]
            for prevNode in prevNodes:
                num_paths = ans[i-1][prevNode][3]
                if total_num_paths==0:
                    p = 1
                else:
                    p = num_paths/total_num_paths
                v1 = currNode
                v2 = prevNode
                if v1<v2:
                    edge = (v1,v2)
                else:
                    edge = (v2,v1)
                if edge not in edge_betweeness:
                    edge_betweeness[edge] = credit*p
                else:
                    edge_betweeness[edge] += credit * p
                ans[i-1][prevNode][2] += credit*p
    #print(edges_bet_tmp)
    #print(ans)
betweeness_list = sorted(list(edge_betweeness.items()),key=lambda x:(-x[1],x[0][0],x[0][1]))

with open(output_file1, "w") as file:
    for i in betweeness_list:
        line = str(i[0])+", "+str(round(i[1]/2,5))
        file.write(line + "\n")





end_time = time.time() - start

print(end_time)