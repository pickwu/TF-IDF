from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from pathlib import Path
import jieba
path = Path("./poem").glob("**/*.txt")

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
x_train,x_test,y_train,y_test = train_test_split(data,label,test_size=0.1,random_state=4)

# 创建TF-IDF向量化器
vectorizer = TfidfVectorizer()
# 创建逻辑回归分类器
classifier = LogisticRegression()
# 创建逻辑朴素贝叶斯分类器
classifier_2 = MultinomialNB()
# 构建管道
model = make_pipeline(vectorizer, classifier_2)
# 训练模型
model.fit(x_train, y_train)
# 预测新文本的情感倾向
prediction = model.predict(x_test)
print('预测结果：', prediction)
print('标签结果：',y_test)