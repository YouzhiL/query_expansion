import utils
query_dict = read_file('query.txt')
doc_dict = read_file('doc.txt')
stop_word = read_stopword('stopword.txt')
rele_label = read_label('rele.txt') 
idf = cal_idf(doc_dict=doc_dict)


totalscores = []

trainfeature = []
trainlabel = []

i = 50
for query_index in query_dict:
    #对训练集训练模型
    if int(query_index) >=50:
        write_tocsv= []
        query = example_list2dict(query_dict[query_index])
        print('\n')
        print('current query is: ')
        print(query_index,query,'\n')

        #对该query进行相关文档降序排序，返回[文档号列表]；[文档号,bm25]
        ordered_docIndex,d_order = get_subscores(query,doc_dict,idf,i)
        #print(ordered_docIndex)
        #计算原来元素的标签mAP
        rele_order = []
        for item in ordered_docIndex:
            rele_order.append(rele_label[item])
        orig_mAP = getmAP(query,rele_order)
        #print('releorder: ', rele_order)
        #print('original map: ',orig_mAP)

        
        ordered_doc_dict = dict()
        for index in ordered_docIndex[:25]:
            ordered_doc_dict[index] = doc_dict[index]
        #print(ordered_docIndex)
    
        #对该项查询进行query-expansion,返回仅含有new query的列表
        expan_query = get_expanded(doc_dict,query,ordered_docIndex[:25],max_expan = 300)
        
        #print('expanded query is: ',expan_query,'\n')
        
        
        tf1 = cal_tf(ordered_doc_dict,query)
        tf2 = cal_tf(doc_dict,query)
        tmp_feature = []
        querylabel = dict()
        for newitem in expan_query:
            #print('200newitem',newitem)
            query1 = query.copy()
            query1[newitem] =1
            #对新加的每一个元素后的query进行文档排序
            ordered_docIndex1,d_order1 = get_subscores(query1,doc_dict,idf,i)
            
            #计算新元素的标签mAP
            rele_order = []
            for item in ordered_docIndex1:
                rele_order.append(rele_label[item])
            #print(rele_order)
            mAP = getmAP(query1,rele_order)
            #print('after expansion,the map is: ',mAP)
            #trainlabel.append((mAP-orig_mAP)/orig_mAP)
            tmp = (mAP - orig_mAP)/orig_mAP
            querylabel[newitem] = tmp
        print(querylabel)
            
        candidate = dict()  
        #print('newitem map:',querylabel)
        
        #选择使map变化最大的candidate words
        ordered_mAP = sorted(querylabel.items(), key=lambda item: item[1], reverse=True)
        tmpvalue = [x[1] for x in ordered_mAP]
        maxstd = tmpvalue[30]
        minstd = tmpvalue[-31]
        for item in querylabel:
            if querylabel[item] >= maxstd or querylabel[item] <= minstd:
                candidate[item] = querylabel[item]
        candi_word = list(candidate.keys())
        print(candidate)
        
        #抽取特征
        feature1list,feature2list = feature12(doc_dict,ordered_docIndex[:25],candi_word,query,tf1,tf2)
        feature3list,feature5list,feature7list,feature9list = cooccure_f(ordered_doc_dict,candi_word,query,tf1)
        feature4list,feature6list,feature8list,feature10list = cooccure_colle(doc_dict,candi_word,query,tf2)
        for m in range(len(candi_word)):
            #trainfeature.append([feature1list[m],feature2list[m],feature3list[m],feature4list[m],feature5list[m],feature6list[m],feature7list[m],feature8list[m],feature9list[m],feature10list[m]])
            
            trainfeature.append([feature1list[m],feature2list[m],feature3list[m],feature4list[m],feature5list[m],feature6list[m],feature9list[m],feature10list[m]])
            tmp=[candidate[candi_word[m]],feature1list[m],feature2list[m],feature3list[m],feature4list[m],feature5list[m],feature6list[m],feature9list[m],feature10list[m]]
            write_tocsv.append(tmp)
            trainlabel.append(candidate[candi_word[m]])
            print(candi_word[m],'feature: ', feature1list[m],feature2list[m],feature3list[m],feature4list[m],feature5list[m],feature6list[m],feature9list[m],feature10list[m],'\n')
        print('current label length:', len(trainlabel),'feature length: ', len(trainfeature),'\n\n')    
        i+=1
        
        final = pd.DataFrame(write_tocsv)
        final.to_csv('train_data5.csv',mode='a', header=False,encoding="utf-8")    
print('train label is: ',  '\n',trainlabel, '\n')
print('train feature is: ',  '\n',trainfeature, '\n')