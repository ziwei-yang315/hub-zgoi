import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

#导入数据集

dataset = pd.read_csv('dataset.csv',sep='\t',header=None)
print(type(dataset))  #  读取进来后会成为 DataFrame  ,是二维数据
print(dataset.columns) # Index([0, 1], dtype='int64')
print(dataset.shape) #(12100, 2)
print('--')
print(dataset.iloc[0].head(5))
print('####')
print(dataset[0].head(5)) ##第0列的前5个值
print(type(dataset[0].head(5))) ##第0列的前5个值 从 DataFrame抽取第0 列, 返回的数据是 一个Series对象(一列，是一维)

##提取文本特征

input_sentence = dataset[0].apply(lambda x: ' '.join(jieba.lcut(x)))  ##对DF 第 0 列的字符串进行jieba 的lcut 精确分割
print(type(input_sentence))  # 一个Series对象
print(input_sentence.head(5)) ##第0列的前5个值
print('$$$$')
print(input_sentence.values)

# a= input_sentence.values
# print(type(a)) <class 'numpy.ndarray'>
# print(a[0])
# print(a[1])
# print(a.shape)  #(12100,)  # 1维数组
vector = CountVectorizer()  ## 初始化词频统计器    #对文本进行提取特征，默认是使用表的符合分词 我们用了空格
vector.fit(input_sentence.values) # # 学习词汇表 默认情况下，CountVectorizer 会按空格分割文本
# print (vector.vocabulary_)  # 查看词汇表
input_feature =vector.transform(input_sentence.values)   # # 转换为词频矩阵 #等价于 vector.transform(input_sentence).toarray()
# print(type(input_feature))  # <class 'scipy.sparse._csr.csr_matrix'>
# print(input_feature.shape)  # 查看词频矩阵形状 (12100, 10088)
# print(vector.get_feature_names_out()[:200])  # 打印前20个特征词
# print('$$')
# print(input_feature[:20])
# input_sentence = vector.fit_transform()

model = KNeighborsClassifier()
model.fit(input_feature,dataset[1].values)
print(model)

test_query= '帮我播放一下郭德纲的小品'
test_sentence = ' '.join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])

prediction = model.predict(test_feature)

print('待预测的文本',test_query)
print('KNN模型预测结果',prediction)

model=LogisticRegression()
model.fit(input_feature,dataset[1].values)
print(model)

test_query= '国庆取哪里玩好玩'
test_sentence = ' '.join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])

prediction = model.predict(test_feature)

print('待预测的文本',test_query)
print('逻辑回归模型预测结果',model.predict(test_feature))

model = tree.DecisionTreeClassifier()
model.fit(input_feature,dataset[1].values)
print(model)

test_query = '今天天气怎么样'
test_sentence = ' '.join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])


print('待预测的文本',test_query)
print('决策树模型预测结果',model.predict(test_feature))



