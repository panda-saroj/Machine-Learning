import csv
import numpy as np


'''
Loads the Mushroom Data File and Divides the data into two separate
List of Lists. One for Training and the other for testing
'''
def loadDataFile(filename):
    file = csv.reader(open(filename, "r"))

#    for row in file:
    dataset = list(file)
#    print(dataset)

    train_dataset = dataset[0:4000]
    test_dataset = dataset[4000:]

#    print("type(train_dataset)", type(train_dataset))
#    print("type(test_dataset)", type(test_dataset))

    train_dataset = np.array(train_dataset)
    test_dataset = np.array(test_dataset)

    return train_dataset, test_dataset

'''
This Function Calculates priori probability
of poisounous and edible mushrooms
'''
def calculate_priori_prob(train_dataset):

    p_count = 0
    e_count = 0

    feature_list_len = len(train_dataset[0])
    train_dataset_len = len(train_dataset)

#For Each feature type and edibility type create Blank dictionaries and count each feature type.
    dictlist_p = [dict() for x in range(feature_list_len - 1)]
    dictlist_e = [dict() for x in range(feature_list_len -1)]

#    feature_list_len = len(train_dataset[0])
    for i in range(train_dataset_len):
#If the edibility is p
        if (train_dataset[i][0] == 'p'):
            p_count += 1
            for j in range (1, feature_list_len):

                #print(type(dictlist_p[j-1]))
                #temp = dictlist_p[j-1]

#If the feature value does not exist in the dictionary then initialize its value with 1 else increment the count
                if (train_dataset[i][j] not in (dictlist_p[j-1]).keys()):

                #if ((dictlist_p[j-1]).get([train_dataset[i][j]]) == None):
                    (dictlist_p[j - 1])[train_dataset[i][j]] = 1
                else:
                    (dictlist_p[j - 1])[train_dataset[i][j]] += 1

#If the edibility is e
        if (train_dataset[i][0] == 'e'):
            e_count += 1
            for j in range (1, feature_list_len):
                if (train_dataset[i][j] not in (dictlist_e[j-1]).keys()):

                #if ((dictlist_p[j-1]).get([train_dataset[i][j]]) == None):
                    (dictlist_e[j - 1])[train_dataset[i][j]] = 1
                else:
                    (dictlist_e[j - 1])[train_dataset[i][j]] += 1


    priori_prob_p = p_count / train_dataset_len
    priori_prob_e = e_count / train_dataset_len

    print("p_count", p_count)
    print("e_count", e_count)


    return priori_prob_p, priori_prob_e, dictlist_p, dictlist_e, p_count, e_count


'''
This Function calculates the class for each of the test records and computes the accuracy of prediction.
'''
def predict_category(train_dataset_len, test_dataset, priori_prob_p, priori_prob_e, dictlist_p, dictlist_e, p_count, e_count):
    length = len(test_dataset)

    correct_predict_count = 0


    feature_list_len = len(test_dataset[0])

    conf_array = np.array([[0,0], [0,0]])

    for i in range(len(test_dataset)):
    #for i in range(2000):

#Initialize each of the probability as 1 for each record
        prob_p = 1
        prob_e = 1

        for j in range(1, feature_list_len):

 #           total_feature_values_p = sum(dictlist_p[j - 1].values())
  #          total_feature_values_e = sum(dictlist_e[j - 1].values())

#If the feature in the test data is not seen in the Train data, to avoid 0 probability leading to whole probability as 0
#put a fractional probability for both types of edibility p and e

            if (test_dataset[i][j] not in (dictlist_p[j - 1]).keys()):

                #prob_p *= 1 / (total_feature_values_p + 1)
                prob_p = prob_p * (1 / (train_dataset_len + 1))
            else:
                #prob_p *= (dictlist_p[j - 1])[test_dataset[i][j]] / total_feature_values_p
                #prob_p = prob_p * ((dictlist_p[j - 1])[test_dataset[i][j]] / train_dataset_len)
                prob_p = prob_p * ((dictlist_p[j - 1])[test_dataset[i][j]] / p_count)


            if (test_dataset[i][j] not in (dictlist_e[j - 1]).keys()):

                    #prob_e *= 1 / (total_feature_values_e + 1)
                prob_e = prob_e * (1 / (train_dataset_len + 1))
            else:
                    #prob_e *= (dictlist_e[j - 1])[test_dataset[i][j]] / total_feature_values_e
                #prob_e = prob_e * ((dictlist_e[j - 1])[test_dataset[i][j]] / train_dataset_len)
                prob_e = prob_e * ((dictlist_e[j - 1])[test_dataset[i][j]] / e_count)

#After each feature probability is calculated, multiply it by prior probability of each category.
        prob_p = prob_p * priori_prob_p
        prob_e = prob_e * priori_prob_e

#        print(test_dataset[i])
#        print("prob_p", prob_p)
#        print("prob_e", prob_e)

#Compute whether the higher probability counted


        if (prob_e >= prob_p):
            if (test_dataset[i][0] == 'e'):
                correct_predict_count += 1
                conf_array[0][0] +=1
            else:
                conf_array[0][1] += 1
        else:
            if (test_dataset[i][0] == 'p'):
                correct_predict_count += 1
                conf_array[1][1] += 1
            else:
                conf_array[1][0] += 1



    print("correct_predict_count", correct_predict_count)
    print("Total Test Data", len(test_dataset))

    print("Accuracy % =", (correct_predict_count/len(test_dataset)) * 100)

    print()


    print("contingency table")
    print("=================")

    print()

    print("\t\t  e \t    p")
    print('e\t\t {} \t {}'.format(conf_array[0][0], conf_array[0][1]))
    print('p\t\t{} \t     {}'.format(conf_array[1][0], conf_array[1][1]))





filename = 'agaricus-lepiota.csv'
train_dataset, test_dataset = loadDataFile(filename)

#print("train_dataset")
#print(train_dataset)

#print("test_dataset")
#print(test_dataset)

print("len(train_dataset)", len(train_dataset))
print("len(test_dataset)", len(test_dataset))

#print("train_dataset[0][0]", train_dataset[0][0])

priori_prob_p, priori_prob_e, dictlist_p, dictlist_e, p_count, e_count = calculate_priori_prob(train_dataset)

print("priori_prob_p", priori_prob_p)
print("priori_prob_e", priori_prob_e)

#print("dictlist_p", dictlist_p)
#print("dictlist_e", dictlist_e)

predict_category(len(train_dataset), test_dataset, priori_prob_p, priori_prob_e, dictlist_p, dictlist_e, p_count, e_count)







