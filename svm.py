#svm
import numpy as np
import pandas as pd
import re
import os
import json
import csv
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split



def load_data(filename):
    data = np.genfromtxt(filename, delimiter=',')
    x = data[:, 2:]  # 数据特征
    X = []
    Y = []
    y = data[:, 1]
    print(len(y))
    st1 = y[2500]
    st2 = y[-2501]
    print(st1,st2)
    for i in range(len(y)):
        if y[i] >st2:
            Y.append(1)
            X.append(x[i])
        else:
            Y.append(-1)
            X.append(x[i])

    scaler = preprocessing.StandardScaler().fit(X) 
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.1)
    return x_train, x_test, y_train, y_test,scaler




def train_data(filename):
    data = np.genfromtxt(filename, delimiter=',')
    x = data[:, 2:]  # 数据特征
    X = []
    Y = []
    y = data[:, 1]
    print(len(y))
    st1 = y[2500]
    st2 = y[-2501]
    print(st1,st2)
    for i in range(len(y)):
        if y[i] >st2:
            Y.append(1)
            X.append(x[i])
        elif y[i]<st1:
            Y.append(-1)
            X.append(x[i])
    x_data = np.array(x)
    y_label = np.array(Y)
    scaler = preprocessing.StandardScaler().fit(x_data)
    return x_data, y_label,scaler


def test_data(i,svc):
    datadict = dict()
    candi = []
    filename = rootpath + '/processed_data/query' + str(i) + '.csv'
        #data = np.genfromtxt(filename, delimiter=',')
    with open(filename) as f:
        reader = csv.reader(f)
        result = list(reader)

    for line in result:
        #f = re.sub('|\”|\[|\]', '', f)
        id1,id2,word,feature = line[0],line[1],line[2],json.loads(line[3])
        #print(feature)
        result = svc.predict([feature])
        p = svc.predict_proba([feature])

        #print(result)
        if result[0] == 1:
            datadict[word] = p
    
        
    return datadict


data_path = 'train_data5.csv'
x_train,y_train,scaler = train_data(data_path)
# svc = SVC(kernel='rbf', class_weight='balanced',C = 10,probability=True)
svc = SVC(kernel='rbf',class_weight='balanced',C = 10,probability=True)
svc.fit(x_train, y_train)


# svr = SVR(kernel='rbf')
# svr.fit(x_train, y_train)
# svr = make_pipeline(StandardScaler(), SVR(C=5.0, epsilon=0.2))
# svr.fit(x_train, y_train)
#
#pred2 = svc.predict(x_train)
pred2 = svc.predict(x_test)
#
# jindu = svr.score(x_test, y_test)
# print('测试精度为%s' % jindu)
# jindu = svr.score(x_train, y_train)
# print('训练精度为%s' % jindu)

# jindu = np.sum(np.array(y_test) == pred2, axis=0) / len(y_test)
# print('测试精度为%s' % jindu)
# jindu = np.sum(np.array(y_train) == pred1, axis=0) / len(y_train)
# print('训练精度为%s' % jindu)
jingdu = 0
l = 0
for i in range(len(pred2)):
    if pred2[i] == 1:
        l+=1
        if y_test[i] ==1:
            jingdu+=1
jingdu = jingdu/l
print(jingdu)
datapath = '/processed_data'

for i in range(51):
    allfeature = []
    allword = []
    datadict = dict()
    candi = []
    filename = rootpath + '/processed_data2/query' + str(i) + '.csv'
        #data = np.genfromtxt(filename, delimiter=',')
    with open(filename) as f:
        reader = csv.reader(f)
        result = list(reader)

    for line in result:
        #f = re.sub('|\”|\[|\]', '', f)
        #id1,id2,word,feature = line[0],line[1],line[2],json.loads(line[3])
        id1,word,feature = line[0],line[1],line[2:]
        allword.append(word)
        allfeature.append(feature)
    scaler.transform(allfeature)
    #print(allfeature)
        #print(feature)
    result = svc.predict(allfeature)
    p = svc.predict_proba(allfeature)
    #print(type(p))
    #print(p)
    tmp = [j[1] for j in p]
    tmp1= tmp.copy()
    tmp1.sort()
    std = tmp1[-30]
    for k in range(len(allword)):
        if tmp[k] > std and result[k] == 1:
            candi.append(allword[k])
    writtenfile = rootpath+'/added_word2/' + str(i) + '.csv'
    
    final = pd.DataFrame(np.array(candi))
    final.to_csv(writtenfile,mode='a', header=False,encoding="utf-8")
#         if result[0] == 1:
#             datadict[word] = p
   
#     print(len(datadict))