from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import jieba
path = Path("./secBlog").glob("**/*.txt")

def my_dataset(path=path):
    poem_content = [];label=[]
    for i in path:
        temp = open(i,'r',encoding='utf-8').read()
        temp = ' '.join(jieba.lcut(temp))
        poem_content.append(temp)
        if "moon" in str(i):
            label.append(1)
        else:
            label.append(0)
    return poem_content,label

# 训练数据
data,label = my_dataset()
# 停用词是一个列表
stop_words = list(set(i for i in jieba.cut(open('./stopwords.txt','r',encoding='utf-8').read())))

# 创建TF-IDF向量化器
vectorizer = TfidfVectorizer(stop_words=stop_words)
# 计算TF-IDF特征向量
tfidf_matrix = vectorizer.fit_transform(data)
# 获取词汇表
feature_names = vectorizer.get_feature_names_out()
# 提取关键词
keywords = [];temp=[]
for doc in range(len(data)):
    feature_index = tfidf_matrix[doc,:].nonzero()[1]
    tfidf_scores = zip(feature_index, [tfidf_matrix[doc, x] for x in feature_index])
    sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
    keywords.append([feature_names[i] for i, score in sorted_scores[:5]])
    temp.extend([feature_names[i] for i, score in sorted_scores[:5]])
print('关键词：', keywords)
from collections import Counter
print(Counter(temp))