from csv import reader
import math
from random import randint
from random import seed

data = []
centroid = []
cluster = dict()


def city_block(list1, list2):
    dist = 0
    try:
        for i in range(len(list1) - 1):
            dist += abs(float(list1[i]) - float(list2[i]))
    except:
        print("List1: ",list1)
        print("List2: ",list2)
        print(centroid)
    return dist


def euclidean(list1, list2):
    dist = 0.0
    for i in range(len(list1) - 1):
        dist += (float(list1[i]) - float(list2[i])) ** 2
    return math.sqrt(dist)


def num_attributes_child(temp_data):
    dict = {}
    count = 0
    for index in temp_data:
        count += 1
        label = data[index][-1]
        if label not in dict:
            dict[label] = 1
        else:
            dict[label] = dict[label] + 1

    return dict, count

def num_attributes(temp_data):
    dict = {}
    count = 0
    for index in range(1, len(temp_data)):
        count += 1
        label = data[index][-1]
        if label not in dict:
            dict[label] = 1
        else:
            dict[label] = dict[label] + 1

    return dict, count

def WSS(final, func):
    WSS = 0
    for value in final:
        c = centroid[value]
        for point in final[value]:
            WSS = WSS + func(data[point], c) ** 2
    return (WSS)

def info_gain(data, final, func, label_dict):
    info_gain_value = 0
    #Entropy of the Parent
    not_used, n_parent = num_attributes(data)

    parent_entropy = 0
    for ent in label_dict:
        parent_entropy += entropy(label_dict[ent], n_parent)

    info_gain_value = parent_entropy

    for cluster in final:
        temp = final[cluster]
        temp_dict, count = num_attributes_child(temp)
        for ent in temp_dict:
            temp_ent = entropy(temp_dict[ent], count )
            info_gain_value = info_gain_value - ((count / n_parent) * temp_ent)
    return info_gain_value


def entropy(c,n):
    if n > 0:
        fraction = c / n
    else:
        fraction = 0
    return (-1) * fraction * math.log2(fraction)

def BSS(final, func):
    BSS_Value = 0
    count = 1
    res = [-1]* len(data[1])
    for index in range(2, len(data)):
        temp_frame = data[index]
        for val in range(0, len(temp_frame) - 1):
            new = float(res[val]) + float(temp_frame[val])
            res[val] = new
        count += 1
    for i in range(len(res) - 1):
        res[i] = res[i] / count

    value = 0
    for cent_array in centroid:
        BSS_Value = BSS_Value + (len(final[value])) * func(cent_array, res) ** 2
        value += 1

    return BSS_Value

def tableau(data, final):
    res = [-1] * (len(data))
    for index in final:
        for value in final[index]:
            res[value] = index
    res = res[-(len(res)-1):]
    return res


def main():

    file = input("Path To Data (csv): ")
    centroid_change = False
    # open file in read mode
    with open(file, 'r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            data.append(row)
    read_obj.close()

    label_dict, num  = num_attributes(data)
    K = len(label_dict)

    for i in range(1,4):
       cluster_body(i*K, centroid_change, label_dict)

def cluster_body(K, centroid_change, label_dict):
    orig_k = K
    print("K: ",K)
    seed()
    centroid.clear()
    for num in range(orig_k):
        value = randint(1, (len(data) - 1))
        centroid.append(data[value][:-1])
    # print("Centroid: ", centroid)
    sum_WSS = 0
    sum_BSS = 0
    sum_infoGain = 0
    for j in range(1):
        wss, bss, info = create_cluster(centroid_change, orig_k, city_block, label_dict)
        sum_WSS += wss
        sum_BSS += bss
        sum_infoGain += info
    print("Average WSS: {} Average BSS: {} Average Info Gain: {}".format(sum_WSS / 10, sum_BSS / 10, sum_infoGain / 10))

    seed()
    centroid.clear()
    for num in range(orig_k):
        value = randint(1, (len(data) - 1))
        centroid.append(data[value][:-1])
    # print("Centroid: ", centroid)
    sum_WSS = 0
    sum_BSS = 0
    sum_infoGain = 0
    for j in range(1):
        wss,bss, info = create_cluster(centroid_change, orig_k, euclidean, label_dict)
        sum_WSS += wss
        sum_BSS += bss
        sum_infoGain += info
    print("Average WSS: {} Average BSS: {} Average Info Gain: {}".format(sum_WSS / 10, sum_BSS / 10, sum_infoGain / 10))

def create_cluster(centroid_change, k, func, label_dict):
    print(func)
    while not centroid_change:
        cent_dict = dict()
        for val in range(k):
            cent_dict[val] = []

        for index in range(1, len(data)):
            temp = dict()
            for val in range(k):
                temp[val] = []
            for num in range(len(centroid)):
                temp[num].append(func(data[index], centroid[num]))
            closest = sorted(temp, key=temp.get)[0]
            cent_dict[closest].append(index)

        for val in range(len(cent_dict)):
            new = recompute(cent_dict[val])
            centroid_change = (new == centroid[val])
            centroid[val] = new

    print(cent_dict)
    return WSS(cent_dict, func), BSS(cent_dict, func), info_gain(data, cent_dict, func, label_dict)
    # print(cent_dict)
    # print("WSS: ", WSS(cent_dict, func))
    # print("BSS: ", BSS(cent_dict, func))
    # print("Info Gain: ", info_gain(data,cent_dict, func, label_dict))
    # print()


def recompute(cluster):
    new_cent = []

    if len(cluster) == 0:
        seed()
        cluster.append(randint(1,len(data) -1))
    try:
        for att in range(len(data[int(cluster[0])]) - 1):
            sum = 0
            for index in cluster:
                current = data[index]
                sum += float(current[att])
            new_cent.append(sum / len(cluster))

    except:
        print("Cluster had no values")
    return new_cent
main()
