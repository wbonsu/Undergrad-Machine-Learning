import pandas as pd
from scipy.io import arff
import math

def createModel(dict, path):
    data = arff.loadarff(path)
    df = pd.DataFrame(data[0])
    count = 0
    num_rows = int(df.count()[1])
    for x in range(num_rows):
        temp = []
        for key in df.keys():
           str = df.get(key)
           temp.append(str[count])
        dict[count] = temp
        count += 1

def euclidean(list1, list2):
    dist = 0.0
    for i in range(len(list1) - 1):
        dist += (list1[i] - list2[i])**2
    return math.sqrt(dist)

def chebyshev(list1, list2):
    max = -1
    for i in range(len(list1) - 1):
        dist = abs(list1[i] - list2[i])
        if dist > max:
            max = dist
    return max

def city_block(list1, list2):
    dist = 0
    for i in range(len(list1) - 1):
        dist += abs(list1[i] - list2[i])
    return dist

def cosine_sim(list1, list2):
    prod = 0
    x = 0
    y = 0
    val = len(list1) - 1
    for i in range(val):
        prod += (list1[i] * list2[i])
    for i in range(val):
        x += (list1[i])**2
    for i in range(val):
        y += (list2[i]) ** 2
    return prod / math.sqrt(x*y)

def main():
    train = input("Training Data: ")
    test = input("Testing Data: ")
    genes = {}
    test_info = {}
    createModel(genes, train)
    createModel(test_info, test)

   #K-Values
    k = [3, 5, 7, 9, 11]
    for value in k:
        eucl_confusion = {}
        cheb_confusion = {}
        city_confusion = {}
        sim_confusion = {}

        print("For K = {}".format(value))
        count = 0
        for test_row in test_info.values():

            count += 1
            eucl, cheb, city, cos = evaluate(test_row, genes, value)
            eucl_voting = voting_dist(eucl, genes)
            cheb_voting = voting_dist(cheb, genes)
            city_voting = voting_dist(city, genes)
            sim_voting = sim(cos, genes)

            if test_row[-1] not in eucl_confusion:
                eucl_confusion[test_row[-1]] = [0,0]
            if test_row[-1] == eucl_voting:
                eucl_confusion[test_row[-1]][0] += 1
            else:
                eucl_confusion[test_row[-1]][1] += 1

            if test_row[-1] not in cheb_confusion:
                cheb_confusion[test_row[-1]] = [0,0]
            if test_row[-1] == cheb_voting:
                cheb_confusion[test_row[-1]][0] += 1
            else:
                cheb_confusion[test_row[-1]][1] += 1

            if test_row[-1] not in city_confusion:
                city_confusion[test_row[-1]] = [0, 0]
            if test_row[-1] == city_voting:
                city_confusion[test_row[-1]][0] += 1
            else:
                city_confusion[test_row[-1]][1] += 1

            if test_row[-1] not in sim_confusion:
                sim_confusion[test_row[-1]] = [0, 0]
            if test_row[-1] == sim_voting:
                sim_confusion[test_row[-1]][0] += 1
            else:
                sim_confusion[test_row[-1]][1] += 1

        pred = pred_val(eucl_confusion, list(eucl_confusion)[0], list(eucl_confusion)[1])
        print("Euclidean {} Precision: {} Recall: {} F1: {}".format(list(eucl_confusion)[0], pred[0], pred[1], pred[2]))
        pred = pred_val(eucl_confusion, list(eucl_confusion)[1], list(eucl_confusion)[0])
        print("Euclidean {} Precision: {} Recall {} F1: {}".format(list(eucl_confusion)[1], pred[0], pred[1], pred[2]))

        pred = pred_val(cheb_confusion, list(cheb_confusion)[0], list(eucl_confusion)[1])
        print("Chebyshev {} Precision: {} Recall: {} F1: {}".format(list(cheb_confusion)[0], pred[0], pred[1], pred[2]))
        pred = pred_val(cheb_confusion, list(cheb_confusion)[1], list(cheb_confusion)[0])
        print("Chebyshev {} Precision: {} Recall {} F1: {}".format(list(cheb_confusion)[1], pred[0], pred[1], pred[2]))

        pred = pred_val(city_confusion, list(city_confusion)[0], list(city_confusion)[1])
        print("City {} Precision: {} Recall: {} F1: {}".format(list(city_confusion)[0], pred[0], pred[1], pred[2]))
        pred = pred_val(city_confusion, list(city_confusion)[1], list(city_confusion)[0])
        print("City {} Precision: {} Recall {} F1: {}".format(list(city_confusion)[1], pred[0], pred[1], pred[2]))

        pred = pred_val(sim_confusion, list(sim_confusion)[0], list(sim_confusion)[1])
        print("Similarity {} Precision: {} Recall: {} F1: {}".format(list(sim_confusion)[0], pred[0], pred[1], pred[2]))
        pred = pred_val(sim_confusion, list(sim_confusion)[1], list(sim_confusion)[0])
        print("Similarity {} Precision: {} Recall {} F1: {}".format(list(sim_confusion)[1], pred[0], pred[1], pred[2]))

        eucl_confusion[list(eucl_confusion)[1]].reverse()
        cheb_confusion[list(cheb_confusion)[1]].reverse()
        city_confusion[list(city_confusion)[1]].reverse()
        sim_confusion[list(sim_confusion)[1]].reverse()

        print("Euclidean: {}".format(eucl_confusion))
        print("Cheb: {}".format(cheb_confusion))
        print("City: {}".format(city_confusion))
        print("Similarity: {}".format(sim_confusion))

def pred_val(conf, label, second):
    TP = conf[label][0]
    FN = conf[label][1]
    FP = conf[second][1]

    d1 = 1 if (TP + FP) == 0 else (TP + FP)
    d2 = 1 if (TP + FN) == 0 else (TP + FN)
    d3 = 1 if (2*TP + FP + FN) == 0 else (2*TP + FP + FN)

    return TP/d1, TP / d2, (2*TP) / d3

#Each neighbor votes weighted by sim**2
def sim(eval_list, genes):
    voting = {}
    for entry in eval_list:
        vote = genes[entry[1]][-1]
        sim_value = entry[0]
        if vote in voting:
            voting[vote] += sim_value**2
        else:
            voting[vote] = sim_value**2
    return sorted(voting, key=voting.get)[-1]

#Each neighbor votes weighted by 1 / (dist**2 + 1)
def voting_dist(eval_list, genes):
    voting = {}
    for entry in eval_list:
        vote = genes[entry[1]][-1]
        dist_weight = 1 / (entry[0] ** 2 + 1)
        if vote in voting:
            voting[vote] += dist_weight
        else:
            voting[vote] = dist_weight
    return sorted(voting, key=voting.get)[-1]

def evaluate(list, dict, k):
    index = 0

    euclid = []
    cheb = []
    city = []
    cos = []

    for train_row in dict.values():
        euclid.append( (euclidean(list, train_row), index))
        cheb.append( (chebyshev(list, train_row), index) )
        city.append( (city_block(list, train_row), index) )
        cos.append( (cosine_sim(list, train_row), index) )
        index += 1

    return sorted(euclid, key=lambda x: x[0])[:k], sorted(cheb, key=lambda x: x[0])[:k], sorted(city, key=lambda x: x[0])[:k], sorted(cos, key=lambda x: x[0], reverse= True)[:k]

main()