# %%
import csv
import math
import numpy as np
import pandas as pd
import random

train_data = pd.read_csv('train_modified_v5.csv')
test_data = pd.read_csv('test_modified_v5.csv')

# %%
# Data Preprocess


def data_process(data):
    # data['AgeLabel'] = data['AgeLabel'].replace(['Young'], 1)
    # data['AgeLabel'] = data['AgeLabel'].replace(['Older'], 2)
    data['Title'] = data['Title'].replace(['Master'], 1)
    data['Title'] = data['Title'].replace(['Mr'], 2)
    data['Title'] = data['Title'].replace(['Capt'], 2)
    data['Title'] = data['Title'].replace(['Col'], 2)
    data['Title'] = data['Title'].replace(['Jonkheer'], 2)
    data['Title'] = data['Title'].replace(['Major'], 2)
    data['Title'] = data['Title'].replace(['Rev'], 2)
    data['Title'] = data['Title'].replace(['Sir'], 2)
    data['Title'] = data['Title'].replace(['Miss'], 3)
    data['Title'] = data['Title'].replace(['Mlle'], 3)
    data['Title'] = data['Title'].replace(['Mme'], 3)
    data['Title'] = data['Title'].replace(['Mrs'], 4)
    data['Title'] = data['Title'].replace(['Ms'], 4)


data_process(train_data)
data_process(test_data)

# %%
# Variable Declaration

_survived_numfamily = np.empty(11)
_died_numfamily = np.empty(11)
_survived_numage = np.empty(12)
_died_numage = np.empty(12)
_survived_title = np.empty(5)
_died_title = np.empty(5)
_survived_ppp = 0
_died_ppp = 0
_survived_class1 = 0
_survived_class2 = 0
_survived_class3 = 0
_died_class1 = 0
_died_class2 = 0
_died_class3 = 0
_survived_age_mean = 0
_survived_age_var = 0
_died_age_mean = 0
_died_age_var = 0
_survived_male = 0
_survived_female = 0
_died_male = 0
_died_female = 0
_survived_Emb_Q = 0
_survived_Emb_S = 0
_survived_Emb_C = 0
_died_Emb_Q = 0
_died_Emb_S = 0
_died_Emb_C = 0
_survived_logfare_mean = 0
_survived_logfare_var = 0
_died_logfare_mean = 0
_died_logfare_var = 0

# %%
# data processing before using Naive Bayes


def dataTraining(_train):
    global _survived_ppp
    global _died_ppp
    global _survived_class1
    global _survived_class2
    global _survived_class3
    global _died_class1
    global _died_class2
    global _died_class3
    global _survived_age_mean
    global _survived_age_var
    global _died_age_mean
    global _died_age_var
    global _survived_male
    global _survived_female
    global _died_male
    global _died_female
    global _survived_Emb_Q
    global _survived_Emb_S
    global _survived_Emb_C
    global _died_Emb_Q
    global _died_Emb_S
    global _died_Emb_C
    global _survived_logfare_mean
    global _survived_logfare_var
    global _died_logfare_mean
    global _died_logfare_var

    _train_survived = _train['Survived'] == 1
    _train_survived = _train[_train_survived]
    _train_died = _train.drop(_train_survived.index)

    _survived_ppp = len(_train_survived) / \
        (len(_train_survived)+len(_train_died))
    _died_ppp = len(_train_died)/(len(_train_survived)+len(_train_died))

    # Pclass
    _survived_class1 = _train_survived['Pclass'] == 1
    _survived_class1 = len(
        _train_survived[_survived_class1])/len(_train_survived)
    _survived_class2 = _train_survived['Pclass'] == 2
    _survived_class2 = len(
        _train_survived[_survived_class2])/len(_train_survived)
    _survived_class3 = _train_survived['Pclass'] == 3
    _survived_class3 = len(
        _train_survived[_survived_class3])/len(_train_survived)

    _died_class1 = _train_died['Pclass'] == 1
    _died_class1 = len(_train_died[_died_class1])/len(_train_died)
    _died_class2 = _train_died['Pclass'] == 2
    _died_class2 = len(_train_died[_died_class2])/len(_train_died)
    _died_class3 = _train_died['Pclass'] == 3
    _died_class3 = len(_train_died[_died_class3])/len(_train_died)

    # AgeLabel
    for i in range(0, 12):
        _survived_age = _train_survived['AgeLabel'] == i
        _survived_numage[i] = len(
            _train_survived[_survived_age])/len(_train_survived)
        _died_age = _train_died['AgeLabel'] == i
        _died_numage[i] = len(_train_died[_died_age])/len(_train_died)

    _survived_young = _train_survived['AgeLabel'] == 1
    _survived_young = len(
        _train_survived[_survived_young])/len(_train_survived)
    _survived_older = _train_survived['AgeLabel'] == 2
    _survived_older = len(
        _train_survived[_survived_older])/len(_train_survived)

    _died_young = _train_died['AgeLabel'] == 1
    _died_young = len(_train_died[_died_young])/len(_train_died)
    _died_older = _train_died['AgeLabel'] == 2
    _died_older = len(_train_died[_died_older])/len(_train_died)

    # Age
    _survived_age_mean = _train_survived['Age_byMean'].mean()
    _survived_age_var = _train_survived['Age_byMean'].var()

    _died_age_mean = _train_died['Age_byMean'].mean()
    _died_age_var = _train_died['Age_byMean'].var()

    # NumFamily
    for i in range(0, 11):
        _survived_family = _train_survived['NumFamily'] == i
        _survived_numfamily[i] = len(
            _train_survived[_survived_family])/len(_train_survived)
        _died_family = _train_died['NumFamily'] == i
        _died_numfamily[i] = len(_train_died[_died_family])/len(_train_died)

    # SexValue
    _survived_male = _train_survived['SexValue'] == 0
    _survived_male = len(_train_survived[_survived_male])/len(_train_survived)
    _survived_female = _train_survived['SexValue'] == 1
    _survived_female = len(
        _train_survived[_survived_female])/len(_train_survived)

    _died_male = _train_died['SexValue'] == 0
    _died_male = len(_train_died[_died_male])/len(_train_died)
    _died_female = _train_died['SexValue'] == 1
    _died_female = len(_train_died[_died_female])/len(_train_died)

    # EmbValue
    _survived_Emb_Q = _train_survived['EmbValue'] == 0
    _survived_Emb_Q = len(
        _train_survived[_survived_Emb_Q])/len(_train_survived)
    _survived_Emb_S = _train_survived['EmbValue'] == 1
    _survived_Emb_S = len(
        _train_survived[_survived_Emb_S])/len(_train_survived)
    _survived_Emb_C = _train_survived['EmbValue'] == 2
    _survived_Emb_C = len(
        _train_survived[_survived_Emb_C])/len(_train_survived)

    _died_Emb_Q = _train_died['EmbValue'] == 0
    _died_Emb_Q = len(_train_died[_died_Emb_Q])/len(_train_died)
    _died_Emb_S = _train_died['EmbValue'] == 1
    _died_Emb_S = len(_train_died[_died_Emb_S])/len(_train_died)
    _died_Emb_C = _train_died['EmbValue'] == 2
    _died_Emb_C = len(_train_died[_died_Emb_C])/len(_train_died)

    # LogFare
    _survived_logfare_mean = _train_survived['LogFare'].mean()
    _survived_logfare_var = _train_survived['LogFare'].var()

    _died_logfare_mean = _train_died['LogFare'].mean()
    _died_logfare_var = _train_died['LogFare'].var()

    # Title
    stitle1 = _train_survived['Title'] == 1
    stitle1 = _train_survived[stitle1]
    stitle5 = _train_survived.drop(stitle1.index)
    stitle2 = _train_survived['Title'] == 2
    stitle2 = _train_survived[stitle2]
    stitle5 = stitle5.drop(stitle2.index)
    stitle3 = _train_survived['Title'] == 3
    stitle3 = _train_survived[stitle3]
    stitle5 = stitle5.drop(stitle3.index)
    stitle4 = _train_survived['Title'] == 4
    stitle4 = _train_survived[stitle4]
    stitle5 = stitle5.drop(stitle4.index)
    _survived_title[0] = len(stitle1)/len(_train_survived)
    _survived_title[1] = len(stitle2)/len(_train_survived)
    _survived_title[2] = len(stitle3)/len(_train_survived)
    _survived_title[3] = len(stitle4)/len(_train_survived)
    _survived_title[4] = len(stitle5)/len(_train_survived)

    dtitle1 = _train_died['Title'] == 1
    dtitle1 = _train_died[dtitle1]
    dtitle5 = _train_died.drop(dtitle1.index)
    dtitle2 = _train_died['Title'] == 2
    dtitle2 = _train_died[dtitle2]
    dtitle5 = dtitle5.drop(dtitle2.index)
    dtitle3 = _train_died['Title'] == 3
    dtitle3 = _train_died[dtitle3]
    dtitle5 = dtitle5.drop(dtitle3.index)
    dtitle4 = _train_died['Title'] == 4
    dtitle4 = _train_died[dtitle4]
    dtitle5 = dtitle5.drop(dtitle4.index)
    _died_title[0] = len(dtitle1)/len(_train_died)
    _died_title[1] = len(dtitle2)/len(_train_died)
    _died_title[2] = len(dtitle3)/len(_train_died)
    _died_title[3] = len(dtitle4)/len(_train_died)
    _died_title[4] = len(dtitle5)/len(_train_died)

# %%
# define logfare GaussianNB function


def gaussianNB(x, mean, var):
    a = 1/(2*math.pi*var)**0.5
    b = -((x-mean)**2)/(2*var)
    c = math.exp(b)
    return a*c


# %%
# define Naive Bayes function

def bayesTest(data, pr1, pr2, pr3, pr4, pr5, pr6):

    if data['SexValue'] == 0:
        s_sexvalue = _survived_male
        d_sexvalue = _died_male
    else:
        s_sexvalue = _survived_female
        d_sexvalue = _died_female

    if pr1 == 1:
        if data['Pclass'] == 1:
            s_pclass = _survived_class1
            d_pclass = _died_class1
        elif data['Pclass'] == 2:
            s_pclass = _survived_class2
            d_pclass = _died_class2
        else:
            s_pclass = _survived_class3
            d_pclass = _died_class3
    else:
        s_pclass = 1
        d_pclass = 1

    if pr2 == 1:
        # for i in range(0, 12):
        #     if data['AgeLabel'] == i:
        #         s_agelabel = _survived_numage[i]
        #         d_agelabel = _died_numage[i]
        s_agelabel = gaussianNB(
            data['Age_byMean'], _survived_age_mean, _survived_age_var)
        d_agelabel = gaussianNB(
            data['Age_byMean'], _died_age_mean, _died_age_var)
    else:
        s_agelabel = 1
        d_agelabel = 1

    if pr3 == 1:
        for i in range(0, 11):
            if data['NumFamily'] == i:
                s_numfamily = _survived_numfamily[i]
                d_numfamily = _died_numfamily[i]
    else:
        s_numfamily = 1
        d_numfamily = 1

    if pr4 == 1:
        for i in range(0, 4):
            if data['Title'] == i:
                s_title = _survived_title[i]
                d_title = _died_title[i]
                break
            else:
                s_title = _survived_title[4]
                d_title = _died_title[4]
    else:
        s_title = 1
        d_title = 1

    if pr5 == 1:
        if data['EmbValue'] == 0:
            s_embvalue = _survived_Emb_Q
            d_embvalue = _died_Emb_Q
        elif data['EmbValue'] == 1:
            s_embvalue = _survived_Emb_S
            d_embvalue = _died_Emb_S
        else:
            s_embvalue = _survived_Emb_C
            d_embvalue = _died_Emb_C
    else:
        s_embvalue = 1
        d_embvalue = 1

    if pr6 == 1:
        s_logfare = gaussianNB(
            data['LogFare'], _survived_logfare_mean, _survived_logfare_var)
        d_logfare = gaussianNB(
            data['LogFare'], _died_logfare_mean, _died_logfare_var)
    else:
        s_logfare = 1
        d_logfare = 1

    s = _survived_ppp*s_sexvalue * s_pclass*s_agelabel * \
        s_numfamily*s_title*s_embvalue*s_logfare
    d = _died_ppp*d_sexvalue * d_pclass*d_agelabel * \
        d_numfamily*d_title*d_embvalue*d_logfare
    #
    if s > d:
        return 1
    else:
        return 0


# %%
# Testing the Combinations

# listTesting_pr = [0]*63
# for a in range(1, 64):
#     total_correct = 0
#     total_precision = 0
#     total_recall = 0
#     print('========== Random [%d] ==========' % a)
#     _a = a
#     if a >= 32:
#         pr1 = 1
#         a -= 32
#     else:
#         pr1 = 0
#     if a >= 16:
#         pr2 = 1
#         a -= 16
#     else:
#         pr2 = 0
#     if a >= 8:
#         pr3 = 1
#         a -= 8
#     else:
#         pr3 = 0
#     if a >= 4:
#         pr4 = 1
#         a -= 4
#     else:
#         pr4 = 0
#     if a >= 2:
#         pr5 = 1
#         a -= 2
#     else:
#         pr5 = 0
#     if a == 1:
#         pr6 = 1
#     else:
#         pr6 = 0
#     # print('PR1 [%d] PR2 [%d] PR3 [%d] PR4 [%d] PR5 [%d] PR6 [%d]' %
#     #       (pr1, pr2, pr3, pr4, pr5, pr6))
#     for i in range(0, 100):
#         _train = train_data.sample(frac=0.8)
#         _test = train_data.drop(_train.index)
#         correct = 0
#         tp = 0
#         fn = 0
#         fp = 0
#         tn = 0
#         dataTraining(_train)
#         # print('Times[%d]' % (i+1))
#         for j in range(len(_test)):
#             countSur = 0
#             # countSur += bayesTest(_test.iloc[j], pr1, pr2, pr3, pr4, pr5, pr6)

#             # countSur += bayesTest(_test.iloc[j], 0, 1, 1, 0, 0, 0)
#             # countSur += bayesTest(_test.iloc[j], 1, 0, 1, 0, 1, 0)
#             # countSur += bayesTest(_test.iloc[j], 1, 1, 1, 0, 0, 0)
#             # countSur += bayesTest(_test.iloc[j], 1, 1, 1, 0, 1, 0)

#             if pr1 == 1:
#                 countSur += bayesTest(_test.iloc[j], 0, 1, 1, 0, 1, 0)
#             if pr2 == 1:
#                 countSur += bayesTest(_test.iloc[j], 0, 0, 1, 0, 1, 0)
#             if pr3 == 1:
#                 countSur += bayesTest(_test.iloc[j], 0, 1, 1, 0, 0, 0)
#             if pr4 == 1:
#                 countSur += bayesTest(_test.iloc[j], 1, 0, 1, 0, 1, 0)
#             if pr5 == 1:
#                 countSur += bayesTest(_test.iloc[j], 1, 1, 1, 0, 0, 0)
#             if pr6 == 1:
#                 countSur += bayesTest(_test.iloc[j], 1, 1, 1, 0, 1, 0)

#             if countSur/(pr1+pr2+pr3+pr4+pr5+pr6) > 0.5:
#                 # if countSur == 1:
#                 # if countSur/4 > 0.5:
#                 if _test.iloc[j]['Survived'] == 1:
#                     correct += 1
#                     tp += 1
#                 else:
#                     fp += 1
#             else:
#                 if _test.iloc[j]['Survived'] == 0:
#                     correct += 1
#                     tn += 1
#                 else:
#                     fn += 1
#             # else:
#             #     print(_test.iloc[j]['PassengerId'])
#         # print(correct/len(_test))
#         total_correct += (correct/len(_test))
#         total_precision += tp/(tp+fp)
#         total_recall += tp/(tp+fn)
#     # print('=====Final=====')
#     print('Correct Rate [%f]' % (total_correct/100))
#     print('Precision [%f]' % (total_precision/100))
#     print('Recall [%f]' % (total_recall/100))
#     print('F Score [%f]' % (2*(total_recall/100)*(total_precision /
#           100)/((total_precision/100)+(total_recall/100))))
#     listTesting_pr[_a-1] = [(_a), (total_correct/100),
#                             (total_precision/100), (total_recall/100)]
# print('\n-----------------High Points-----------------')
# for b in range(0, 63):
#     if listTesting_pr[b][1] > 0.79:
#         print('[%d]' % listTesting_pr[b][0])
#         print(listTesting_pr[b][1])

# %%
# Test data Predict

dataTraining(train_data)
PassengerId = []
Survived = []
dict = {'PassengerId': PassengerId, 'Survived': Survived}
for i in range(len(test_data)):
    PassengerId.append(892+i)
    countSur = 0
    countSur += bayesTest(test_data.iloc[i], 0, 1, 1, 0, 1, 0)
    countSur += bayesTest(test_data.iloc[i], 0, 0, 1, 0, 1, 0)
    countSur += bayesTest(test_data.iloc[i], 0, 1, 1, 0, 0, 0)
    countSur += bayesTest(test_data.iloc[i], 1, 0, 1, 0, 1, 0)
    countSur += bayesTest(test_data.iloc[i], 1, 1, 1, 0, 0, 0)
    countSur += bayesTest(test_data.iloc[i], 1, 1, 1, 0, 1, 0)
    if countSur/2 > 0.5:
        Survived.append(1)
    else:
        Survived.append(0)
print('Finish Predicting')
test_0629 = pd.DataFrame(dict)
test_0629.to_csv('Team_09.csv', index=None)


# %%
