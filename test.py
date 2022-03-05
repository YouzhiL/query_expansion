
import os
query_dict = read_file('query.txt')
doc_dict = read_file('doc.txt')
#stop_word = read_stopword('stopword.txt')

idf = cal_idf(doc_dict=doc_dict)

j = 0
i = 0
testfeature = []
testpredict = []
totalscores = []

for query_index in query_dict:
    # 对训练集训练模型
    if int(query_index) <= 50:
        write_tocsv = []
        query = example_list2dict(query_dict[query_index])
        final_query = query.copy()
        print('\n')
        print('current query is: ')
        print(query_index, query, '\n')

        # 对该query进行相关文档降序排序，返回[文档号列表]；[文档号,bm25]
        ordered_docIndex, d_order = get_subscores(query, doc_dict, idf, i)


        ordered_doc_dict = dict()
        for index in ordered_docIndex[:25]:
            ordered_doc_dict[index] = doc_dict[index]
        # print(ordered_docIndex)

        #candidate words for expansion
        candi_word = cal_tfidf(ordered_doc_dict)

        tf1 = cal_tf(ordered_doc_dict, query)
        tf2 = cal_tf(doc_dict, query)

        # 抽取特征
        feature1list, feature2list = feature12(doc_dict, ordered_docIndex[:25], candi_word, query, tf1, tf2)
        feature3list, feature5list, feature7list, feature9list = cooccure_f(ordered_doc_dict, candi_word, query, tf1)
        feature4list, feature6list, feature8list, feature10list = cooccure_colle(doc_dict, candi_word, query, tf2)
        for m in range(len(candi_word)):
            # trainfeature.append([feature1list[m],feature2list[m],feature3list[m],feature4list[m],feature5list[m],feature6list[m],feature7list[m],feature8list[m],feature9list[m],feature10list[m]])
            feature = [feature1list[m], feature2list[m], feature3list[m], feature4list[m], feature5list[m], feature6list[m],
                 feature9list[m], feature10list[m]]
            testfeature.append(feature)
            tmp = [candi_word[m],feature1list[m], feature2list[m], feature3list[m], feature4list[m], feature5list[m], feature6list[m],
                 feature9list[m], feature10list[m]]
            write_tocsv.append(tmp)
        print( 'current length: ', len(testfeature), '\n\n')
        
        file = rootpath + '/processed_data2/query' + str(i) + '.csv'
        final = pd.DataFrame(write_tocsv)
        final.to_csv(file, mode='a', header=False, encoding="utf-8")
        i += 1
print('test feature is: ', '\n', testfeature, '\n')
