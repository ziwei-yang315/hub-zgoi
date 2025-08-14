import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

#数据处理
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))
vector = TfidfVectorizer()
input_feature = vector.fit_transform(input_sentence.values)
output = dataset[1].values
test_query = "帮我播放一下郭德纲的小品"
test_sentence = " ".join(jieba.lcut(test_query))

#贝叶斯模型
bs_model = MultinomialNB()
bs_model.fit(input_feature, output)

#贝叶斯验证
test_feature = vector.transform([test_sentence])
print("待预测的文本", test_query)
print("贝叶斯模型预测结果: ", bs_model.predict(test_feature))

#SVM模型
vector = CountVectorizer()
input_feature = vector.fit_transform(input_sentence.values)
output = dataset[1].values

#SVM验证
svm_model = SVC(kernel="linear", C=1.0, probability=True, class_weight="balanced")
svm_model.fit(input_feature, output)
print("svm模型预测结果: ", svm_model.predict(test_feature))