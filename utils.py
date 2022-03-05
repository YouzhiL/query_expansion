import math
import pandas as pd
from sklearn import preprocessing #标准化数据
import numpy as np
from sklearn import svm
from sklearn.svm import SVR
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer

def read_file(file):
    fp = open(file)
    output = dict()
    for line in fp.readlines():
        id, text = line.strip().split('\t')
        output[id] = text
    return output

def read_stopword(file):
    fp = open(file)
    output = []
    for line in fp.readlines():
        output.append(line.strip())
    return output
                      
def read_label(file):
    fp = open(file)
    output = dict()
    for line in fp.readlines():
        i, id, text = line.strip().split('\t')
        output[id] = text
    return output

def example_list2dict(input):
    output = dict()
    for word in input.split():
        if output.get(word) is None:
            output[word] = 0
        output[word] += 1
    return output

def cal_idf(doc_dict):
    doc_num = len(doc_dict)
    idf = dict()
    for doc_id in doc_dict:
        doc_text = list(set(doc_dict[doc_id].split()))
        for word in doc_text:
            if idf.get(word) is None:
                idf[word] = 0
            idf[word] += 1
    for word in idf:
        idf[word] = math.log((doc_num - idf[word] + 0.5) / (idf[word] + 0.5))
    return idf

def cal_tfidf(ordered):
    print('calculate tfidf')
    texts = []
    for doc in ordered:
        texts.append(ordered[doc])
    print(texts)
    vt = TfidfVectorizer()  
    result = vt.fit_transform(texts)
    print(result.toarray())
    print(vt.get_feature_names())

    # Get the TF-IDF value for each word/vocabulary
    scores = zip(vt.get_feature_names(), np.asarray(result.sum(axis=0)).ravel())

    # Sort terms based on highest TF-IDF value
    top_term_with_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Return top n terms with the highest TF-IDF value
    # return top_term_with_scores[:50]
    word = []
    for i in range(60):
        word.append(top_term_with_scores[i][0])
    return word


def bm25(query, doc, idf, avg_doc_len=374):
    k1 = 2.0
    k2 = 1.0
    b = 0.75
    score = 0.0
    for word in query:
        if doc.get(word) == None:
            continue
        W_i = idf[word]
        f_i = doc[word]
        qf_i = query[word]
        doc_len = sum(doc.values())
        K = k1 * (1 - b + b * doc_len / avg_doc_len)
        R1 = f_i * (k1 + 1) / (f_i + K)
        R2 = qf_i * (k2 + 1) / (qf_i + k2)
        R = R1 * R2
        score += W_i * R
    return score

def get_expanded1(doc_dict,query,ordered,max_expan = 300):
    querylist = list(query.keys())
    keyword = dict()
    for doc_index in ordered:
        doc = example_list2dict(doc_dict[doc_index])
        for content in doc:
            
            if content not in querylist:
                if not keyword.__contains__(content):
                    keyword[content] = doc[content]
                else:
                    keyword[content] += doc[content]
        
    ordered_freq = sorted(keyword.items(), key=lambda item:item[1], reverse=True)
    freq = [x[1] for x in ordered_freq]
    freq = freq[:max_expan]
    ordered_freq = [x[0] for x in ordered_freq]
    ordered_freq = ordered_freq[:max_expan]
    #order_freq 存储频率大于等于3的可拓展query
    expan_query = []
    for num in range(max_expan):
        if freq[num]>2:
            expan_query.append(ordered_freq[num])

    return(expan_query)


def get_expanded(doc_dict,query,ordered,max_expan = 300):
    querylist = list(query.keys())
    keyword = dict()
    for doc_index in ordered:
        doc = example_list2dict(doc_dict[doc_index])
        for content in doc:
            
            if content not in querylist:
                if not keyword.__contains__(content):
                    keyword[content] = doc[content]
                else:
                    keyword[content] += doc[content]
        
    words = list(keyword.items())
    print(words)
    expan_query = []
    for i in range(len(words)):
        if words[i][1]>2:
            expan_query.append(words[i][0])

    return(expan_query)

      
def get_subscores(query,doc_dict,idf,i):
    subscores = dict()
    for m in range(300):
        doc_index = "%d_%d"% (i, m)
        if(doc_dict.__contains__(doc_index)):
        
            doc = example_list2dict(doc_dict.get(doc_index))
            
        else:
            break
        
        
        score = bm25(query, doc, idf)
        subscores[doc_index] = score

    
    d_order = sorted(subscores.items(), key=lambda item:item[1], reverse=True)
    ordered_docIndex = [x[0] for x in d_order]
    return ordered_docIndex, d_order

def cal_tf(doc_dict,query):
    sum2 = 0#原始query出现的频率
    for doc_index in doc_dict:
        #print('In doc_index:',doc_index)
        doc = example_list2dict(doc_dict[doc_index])
        for word in query:
            #print('original query word',word,'appears',doc.get(word,0),'times')
            sum2+=doc.get(word,0)
    print('tf: ',sum2)
    return sum2


    
def normalize(alist):
    
    maxval = max(alist)
    minval = min(alist)
    if(maxval == minval):
        return [0]*len(alist)
    tmp = np.array(alist)
    tmp = (tmp - minval)/(maxval - minval)
    return list(tmp)

def distri_f(ordered,expan_query,query,tf1):
    #记录在feedback document中，候选词相对原始query出现的频率
    feature1list = []
    #print('\n','在feedback document中，候选词相对原始query出现的频率...')
    #print(ordered)
    for newitem in expan_query:
        sum1 = 0#候选词出现的频率
        for doc_index in ordered:
            #print('In doc_index:',doc_index)
            doc = example_list2dict(ordered[doc_index])
            #print('候选词',newitem,'出现了',doc.get(newitem,0),'次')
            sum1 += doc.get(newitem,0)
        if sum1 == 0:
            print('\n !!!!candidate freq is 0!!!!')
        feature1 =  math.log(sum1/tf1)
        feature1list.append(feature1)

    l1 = normalize(feature1list)
    print('feature1list:',l1)


    return feature1list
    
    
def feature12(doc_dict,ordered_index,expan_query,query,tf1,tf2):
    #记录在all document中，候选词相对原始query出现的频率
    #print(doc_dict)
    feature1list = []#记录在ordered document中，候选词相对原始query出现的频率
    feature2list = []#记录在all document中，候选词相对原始query出现的频率
    #print('\n','在all document中，候选词相对原始query出现的频率...')

    for newitem in expan_query:
        sum1 = 0#在ordered document中，候选词出现的频率
        sum2 = 0#all document中,候选词出现的频率
        for doc_index in doc_dict:
            #print('In doc_index:',doc_index)
            doc = example_list2dict(doc_dict[doc_index])
            #print('候选词',newitem,'出现了',doc.get(newitem,0),'次')
            sum2 += doc.get(newitem,0)
            if doc_index in ordered_index:
                sum1 += doc.get(newitem,0)
        if sum2 == 0:
            print('\n !!!!candidate freq is 0!!!!')
        feature1 =  math.log(sum1/tf1)
        feature2 =  math.log(sum2/tf2)
        feature1list.append(feature1)
        feature2list.append(feature2)
    feature1list = normalize(feature1list)
    feature2list = normalize(feature2list)
    return feature1list,feature2list

def cooccure_f(ordered_doc_dict,expan_query,query,tf):
    
    #print(query)
    feature3list = []
    feature5list = []
    feature7list = []
    feature9list = []
    for newitem in expan_query:

        sum2 = 0#record single occurance
        sum3 = 0#record double occurance
        sum4 = 0
        sum5 = 0
        sum6 = 0#record the times appears with all query words
        b = ordered_doc_dict.values()
        u = 0
        mindis = dict()
        freq = dict()
        for items in b:
        
            i = 0#记录这是第i个单词
            u = 0#记录单词前/后u个单词
            items = items.strip().split(' ')
            #print('\n',items)
            length= len(items)
            for word in items:
                #print('current word is: ',word)
                k = 0
                if word == newitem:
                    #print('current word is: ',word)
                    for q in query:
                        #print('current word in query is: ',q)
                        for u in range(1,8):
                            if i >= 8:
                                if items[i-u] == q:
                                    dis = u
                                    #print('distance is: -',dis)
                                    sum2+=1
                                    mindis[items[i-u]]= min(dis,mindis.get(items[i-u],16))
                                    if freq.get(items[i-u]) is None:
                                        freq[items[i-u]] = 0
                                    freq[items[i-u]] += 1
                                
                            if i <= length-8:
                                if items[i+u]==q:
                                    dis = u
                                    #print('distance is: +',dis)
                                    sum2+=1
                                    mindis[items[i+u]]= min(dis,mindis.get(items[i+u],16))
                                    if freq.get(items[i+u]) is None:
                                        freq[items[i+u]] = 0
                                    freq[items[i+u]] += 1
                                
                        for v in range(1,12):
                            if i >= 12:
                                if items[i-v] == q:
                                    k+=1
                            if i<= length -12:
                                if items[i+v] == q:
                                    k+=1
                    
                    if k>=2:
                        sum3+=1
                    if k>=len(query)-1:
                        sum6+=1
                i+=1
        for item in freq:
            sum4 += freq[item]
            sum5 += freq[item]*mindis[item]
        #print(' single occurance: ',sum2)
        #print('double occurance: ',sum3)
        #print('freq: ',sum4)
        #print('freq*mindis: ',sum5)
        #print(' all occurance: ',sum6)
        if(sum2 == 0):
            feature3 =  math.log(1.0/170000.0)
        else:
            feature3 = math.log(1.0*sum2/tf/len(query))
        if(sum3 == 0):
            feature5 = math.log(1.0/170000.0)
        else:
            feature5 = math.log(1.0*sum3/tf/len(query))

        feature7 = math.log(1.0*sum5/(sum4+1)+1/170000 )
        feature9 = math.log(1.0*sum6+0.5)
        feature3list.append(feature3)
        feature5list.append(feature5)
        feature7list.append(feature7)
        feature9list.append(feature9)
    feature3list = normalize(feature3list)
    feature5list = normalize(feature5list)
    feature7list = normalize(feature7list)
    feature9list = normalize(feature9list)

    
    #print('feature3,5,7,9normalized:' ,feature3list,feature5list,feature7list,feature9list)
    return feature3list,feature5list,feature7list,feature9list



def cooccure_colle(doc_dict,expan_query,query,tf):


    #print(query)
    feature3list = []
    feature5list = []
    feature7list = []
    feature9list = []
    for newitem in expan_query:

        sum2 = 0#record single occurance
        sum3 = 0#record double occurance
        sum4 = 0
        sum5 = 0
        sum6 = 0#record the times appears with all query words
        b = doc_dict.values()
        u = 0#记录query单词前/后u个单词
        mindis = dict()
        freq = dict()
        for items in b:
        
            i = 0#记录这是第i个单词
        
            items = items.strip().split(' ')
            #print('\n',items)
            length= len(items)
            for word in items:
                #print('current word is: ',word)
                k = 0
                if word == newitem:
                    #print('current word is: ',word)
                    for q in query:
                        #print('current word in query is: ',q)
                        for u in range(1,8):
                            if i >= 8:
                                if items[i-u] == q:
                                    dis = u
                                    #print('distance is: -',dis)
                                    sum2+=1
                                    mindis[items[i-u]]= min(dis,mindis.get(items[i-u],16))
                                    if freq.get(items[i-u]) is None:
                                        freq[items[i-u]] = 0
                                    freq[items[i-u]] += 1
                                
                            if i <= length-8:
                                if items[i+u]==q:
                                    dis = u
                                    #print('distance is: +',dis)
                                    sum2+=1
                                    mindis[items[i+u]]= min(dis,mindis.get(items[i+u],16))
                                    if freq.get(items[i+u]) is None:
                                        freq[items[i+u]] = 0
                                    freq[items[i+u]] += 1
                                
                        for v in range(1,12):
                            if i >= 12:
                                if items[i-v] == q:
                                    k+=1
                            if i<= length -12:
                                if items[i+v] == q:
                                    k+=1
                    
                    if k>=2:
                        sum3+=1
                    if k>=len(query)-1:
                        sum6+=1
                i+=1
        for item in freq:
            sum4 += freq[item]
            sum5 += freq[item]*mindis[item]
        #print(' single occurance: ',sum2)
        #print('double occurance: ',sum3)
        #print('freq: ',sum4)
        #print('freq*mindis: ',sum5)
        #print(' all occurance: ',sum6)
        if(sum2 == 0):
            feature3 =  math.log(1.0/170000.0)
        else:
            feature3 = math.log(1.0*sum2/tf/len(query))
        if(sum3 == 0):
            feature5 = math.log(1.0/170000.0)
        else:
            feature5 = math.log(1.0*sum3/tf/len(query))
        feature7 = math.log(1.0*sum5/(sum4+1)+1/170000 )
        feature9 = math.log(1.0*sum6+0.5)
        feature3list.append(feature3)
        feature5list.append(feature5)
        feature7list.append(feature7)
        feature9list.append(feature9)
    feature3list = normalize(feature3list)
    feature5list = normalize(feature5list)
    feature7list = normalize(feature7list)
    feature9list = normalize(feature9list)

    
    #print('feature4,6,8,10 normalized:' ,feature3list,feature5list,feature7list,feature9list)
    return feature3list,feature5list,feature7list,feature9list



def getmAP(query1,rele_order):
    rele = []
    for item in rele_order:
        rele.append(int(item))
    rele = np.array(rele)
    total_rele = sum(rele)
    count = 0
    mAP = 0
    for i in range(len(rele)):
        if(rele[i] == 1):
            count+=1
            mAP += count/(i+1)
            if(count==total_rele):
                return mAP/total_rele





