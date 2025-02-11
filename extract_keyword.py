from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import jieba
path = Path("./secBlog").glob("**/*.txt")

def docs(path=path):
    documents = []
    for i in path:
        temp = open(i,'r',encoding='utf-8').read()
        temp = ' '.join(jieba.lcut(temp)) # 分词
        documents.append(temp)
    return documents

documents = docs(path)
print("文档示例:\t",documents[:1])

stop_words = list(set(i for i in jieba.cut(open('./stopwords.txt','r',encoding='utf-8').read())))
print("停用词列表:\t",stop_words[:5])

vectorizer = TfidfVectorizer(stop_words=stop_words)
print("创建TF-IDF向量化器")

tfidf_matrix = vectorizer.fit_transform(documents)
print("计算TF-IDF特征向量:")

feature_names = vectorizer.get_feature_names_out()
print("词汇表:\t",feature_names)

keywords = [];temp=[]
for doc in range(len(documents)):
    feature_index = tfidf_matrix[doc,:].nonzero()[1]
    tfidf_scores = zip(feature_index, [tfidf_matrix[doc, x] for x in feature_index])
    sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
    keywords.append([feature_names[i] for i, score in sorted_scores[:5]])
    temp.extend([feature_names[i] for i, score in sorted_scores[:5]])
print('每个文档保留前5个关键词：', keywords[:2])
from collections import Counter
print("前20个关键词",Counter(temp).most_common(20))