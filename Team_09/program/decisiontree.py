# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 19:44:01 2021

@author: User
"""

import pandas as pd
import numpy as np
import collections as cc
import math



def generate_dataset(file_name):
    '''
    資料調整
    '''
    #讀取資料
    data_read = pd.read_csv(file_name, index_col=0)
    
    
    if(file_name==r'train_modified_v5.csv'):#訓練資料------訓練資料名稱
        data_read=data_read[['Survived','SexValue','NumFamily','AgeLabel']]#送入模型的特徵
        test_data=data_read.sample(frac=0.2)#20%當作測試資料
        train_data=data_read[~data_read.index.isin(test_data.index)]#80%作為訓練資料
        #把x,y分開
        ytest = test_data['Survived'].values
        test_data.drop('Survived', axis=1, inplace=True)
        xtest = test_data.values
        #把x,y分開
        ytrain = train_data['Survived'].values
        train_data.drop('Survived', axis=1, inplace=True)
        xtrain = train_data.values
        return xtrain, xtest, ytrain, ytest
    else:#預測資料
        data_read=data_read[['SexValue','NumFamily','AgeLabel']]#送入模型的特徵
        x = data_read.values
        return x
           
    
def createtree(xtrain, ytrain, eps):
    '''
    創造決策樹
    '''
    #判斷全活或死
    Class = np.unique(ytrain)
    if len(Class) == 1:#假如全部都一樣，那麼長度就會是1
        return Class[0]
    # 如果已經没有特徵可以分類，就看結果死多活多
    elif len(xtrain[0]) == 0:
        return findmajorclass(ytrain)
    # 其它情况下，需要對結點分類，分類方法:計算信息增益
    else:
        # 計算信息增益(不純度減少的程度)以及得到信息增益最大的特徵
        feature_best, g_max = G(xtrain,ytrain)
        # 如果最佳特徵的信息增益<域值，就看現在哪類多回傳，調MODEL可以用的參數
        if g_max < eps:
            return findmajorclass(ytrain)  
        treeDict = {feature_best:{}}
        # 樹生成後，對數據集根據最佳特徵進行劃分
        sub_datasets, sub_labelsets = splitdataset(xtrain,ytrain, feature_best)
        # 子集合的數量
        sets_num = len(sub_labelsets)
        # 每個子集合再往下長，用遞迴做
        for i in range(sets_num):
            treeDict[feature_best][i] = createtree(sub_datasets[i], sub_labelsets[i], eps)
        return treeDict

def findmajorclass(label):
    '''
    段落說明:找出最多的類別(判斷這群人是活多還是死多)
    '''
    # .most_common(1)，從Counter中提出現數目最多的一組#第一個數字是最多的類別，第二個數字是該類別有多少
    Major_Class = cc.Counter(label).most_common(1)[0][0]#所以我們選擇返回第一項
    #print(cc.Counter(label).most_common(1)[0][0])
    return Major_Class

def G(data, label):
    '''
    得到信息增益g ( D ,A )
    '''
    # 算出當前結點的經驗熵H ( D )
    H = E(label)
    # 算出當前結點的條件熵conditional entropy H ( D ∣ A )、H ( D ∣ B )...用陣列存
    EH= get_E(data, label)
    # 得到最大的信息增益g ( D ,A )
    g = [H - num for num in EH]
    g_max = max(g)
    print(g_max)#幫助參數調整
    # 得到最大信息增益對應之特徵
    feature_best = g.index(g_max)
    print(feature_best)#幫助參數調整
    return feature_best, g_max

def E(label):
    
    '''
    算出當前結點的熵 entropy H ( D )
    '''
    # 所有的類(死活)
    Class = np.unique(label)
    # 統計出死活各多少
    Class_num = cc.Counter(label)
    # 看看類有幾個(2)
    Class_len = len(label)
    #套公式囉
    H = 0
    for c in Class:
        H += -1 *(Class_num[c] / Class_len)* math.log((Class_num[c] / Class_len),2)
    return H

def get_E(data,label):
    '''
    算出當前結點的條件熵H ( D ∣ A )、H ( D ∣ B )...用陣列存
    '''
    #看有幾個feature
    feature_num = len(data[0])
    #看有幾筆data
    data_num = len(data)
    #條件熵
    f_H_Arr = []
    for f in range(feature_num):
    # 這個特徵下所有data有哪些值(.T簡單轉置矩陣)
        f_data = list(np.array(data).T[f])
    # 特徵f有哪些值可取
        f_value = np.unique(f_data)
    # 特徵f的條件熵
        all_H = 0
    # 走過特徵f的每一個可取的值(d1,d2...)
        for f_v in f_value:
            # 得到f_data中值是f_v的所有index
            index = np.argwhere(f_data == f_v)      
            # 值是f_v的所有標記
            f_label = []
            for i in index:
                # 得到此特徵下滿足值是f_v對應的所有標記
                f_label.append(label[i[0]])
            # 計算f_label的熵
            f_H = E(f_label)
            # 得到在該特徵下，值是f_v的機率
            f_P = len(f_label) /data_num
            # 計算出條件熵
            all_H += f_P * f_H
    # 丟到陣列裡準備回傳
        f_H_Arr.append(all_H)
    return f_H_Arr

def splitdataset(data, label, feature):
    '''
    分枝數據集
    '''

    # 轉置data
    data_T = np.transpose(data)
    #獲得想要分枝特徵的可取值
    feature_value = np.unique(data_T[feature])
    # 準備兩個列表，用來放分割之後的子集合
    datasets = []
    labelsets = []
    # 走過最佳特徵的每個data取值
    for f in feature_value:

        datasets_sub = []
        labelsets_sub = []
        # 使用 enumerate() 函式來同時輸出索引與元素，才不會亂掉
        for Index, num in enumerate(data_T[feature]):
            # 當data set中某筆data的e該特徵f時，得到它的index
            if num == f:
                # 將這個特徵去掉，才不會下次又考慮這個特徵
                data_temp = data[Index]
                del data_temp[feature]
                # 存儲劃分後的子集
                # 此時得到的僅為最佳特徵的一個取值下的子集
                datasets_sub.append(data_temp)
                labelsets_sub.append(label[Index])
        # 存儲根據最佳特徵的不同取值劃分的子集
        datasets.append(datasets_sub)
        labelsets.append(labelsets_sub)
    return datasets, labelsets


################################################################

def predict(data, tree):
    '''
    Survived值預測
    '''
    # 初始化Class
    Class = -1
    
    # 當Class被賦予了新值，也就是說該樣本點被分類，則停止循環
    while Class == -1:
        
        # 獲得當前結點的key和value
        # key代表結點中需要對哪一個特徵進行判斷
        # value代表結點的可取值
        (key, value), = tree.items()
        # 該樣本在結點所需判斷的特徵的值
        feature_value = data[key]
        if feature_value in value :
        # 如果判斷下來，其值還是字典
        # 那麼就說明還在內部結點，要繼續往下分
            if type(value[feature_value]).__name__ == 'dict':

                # 將該內部結點及其子樹設為新的樹
                tree = value[feature_value]

            # 如果判斷下來是不是字典了，說明到達葉節點
            if type(value[feature_value]).__name__ != 'dict':
                # 則返回葉結點對應的分類
                Class = value[feature_value]
                #print(Class)
        else : 
            keyy = value.keys()
            thisvalue = [999,999]
            for i in keyy:
                temp = abs(i-feature_value)
                if (temp < thisvalue[0]):
                    thisvalue[0] = temp
                    thisvalue[1] = i
            feature_value = thisvalue[1]
            if type(value[feature_value]).__name__ == 'dict':
                # 將該內部結點及其子樹設為新的樹
                tree = value[feature_value]
            # 如果判斷下來是不是字典了，說明到達葉節點
            if type(value[feature_value]).__name__ != 'dict':
                # 則返回葉結點對應的分類
                Class = value[feature_value]
                
    return Class


def classify(xtest, ytest, tree):
    '''
    測試集測試樣本數量
    '''
    testnum = len(ytest)
    
    #答錯的次數
    errortimes = 0
    #走過每個data確認是否正確
    for i in range(testnum):
            Class = predict(xtest[i], tree)
            if Class != ytest[i]:
                errortimes += 1
    #正確率
    acc = 1 - (errortimes / testnum)
    return acc


def forecast(test_data_list:list, tree):
    '''
    預測資料
    '''
    
    PassengerId=[]
    Survived=[]
    dict = {'PassengerId':PassengerId, 'Survived': Survived}#行名稱
    for i, data in enumerate(test_data_list):#一筆一筆送進去預測
        Class = predict(data, tree)#進行預測
        PassengerId.append(892+i)#從892號開始
        Survived.append(Class)
    df = pd.DataFrame(dict)
    df.to_csv('Forecasted.csv',index = False)
    return df
    
        



#%%

    
if __name__ == "__main__":
    accurate=0
    while(accurate<0.8):#測試資料最小容許準確度
        #讀取訓練資料csv，並完成資料預處理
        file_name =r'train_modified_v5.csv' 
        xtrain, xtest, ytrain, ytest= generate_dataset(file_name)
        #讀取預測資料csv，並完成資料預處理
        file_name2 =r'test_modified_v5.csv'
        test_data= generate_dataset(file_name2)

        #資料結構限制，轉list
        xtrain = xtrain.tolist()
        xtest = xtest.tolist()
        
        tree = createtree(xtrain,ytrain,0.1)#參數為最小信息增益
        accurate = classify(xtest, ytest, tree)#測試準確度，避免過擬和
              
        result_1=forecast(test_data, tree)#預測新資料
        #accurate=1
        

        
    