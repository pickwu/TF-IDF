from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import jieba
import numpy as np

d1 = "我爱北京天安门，北京"
d2 = "天安门上太阳升"
d3 = "明月几时有？把酒问青天。"
d4 = "海上生明月，天涯共此时。"
def preprocess(*ds):
    # ds是多个文档的内容，每个文档切割后，以空格分开
    temp = []
    for i in ds:
        temp.append(" ".join(jieba.lcut(i)))
    return temp

corpus = preprocess(d1,d2,d3,d4)
print("输入文档需要用空格分词：\n",corpus)

print("开始手搓IF-IDF\n",10*"*")
# TfidfVectorizer没有生成TF的方法，借助CountVectorizer生成
print("第一步：去除停用词，计算词袋模型TF")
count_vectorizer = CountVectorizer()
tf_matrix = count_vectorizer.fit_transform(corpus)
tf = tf_matrix.toarray()
print("TF矩阵为:\n", tf)

# IDF: ln(总文档数+1/包含该词的文档+1）+1
count_docs = tf.shape[0] * np.ones(shape=(tf.shape[1])) + 1
docs_with_word = np.sum((tf != 0),axis=0) + 1
idf = np.log(count_docs/docs_with_word) + 1
print("第二步：计算词汇在整体的概率idf\n","idf矩阵为:\n",idf)

# TF-IDF
raw_tf_idf = tf * idf
print("第三步：计算TF*IDF\n","TF-IDF矩阵为：\n",raw_tf_idf)

# TF-IDF规范化
norm = np.linalg.norm(raw_tf_idf,ord=2,axis=1)
norm_tf_idf = raw_tf_idf / norm[:,np.newaxis]
print('第四步：对计算结果进行规范化：\n',"规范化后的TF—IDF矩阵为：\n",norm_tf_idf)
print("结束手搓IF-IDF\n",10*"*",'\n')


# # 借助TfidfVectorizer生成IDF
vector = TfidfVectorizer()
vector.fit(corpus) # 学习词汇表和idf

print("词袋列表：\n",vector.get_feature_names_out())
print("词汇表：\n",vector.vocabulary_)
print('IDF: ',vector.idf_)

# 实际中，vector.fit_transform(corpus),更加便捷
tf_idf =vector.transform(corpus)
print("sklearn的tf_idf:\n",tf_idf.toarray())