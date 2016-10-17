# -*- coding: utf-8 -*-
"""
Created on Thu Sep 01 16:36:42 2016

@author: xuefliang
"""
import glob
import urllib2
import cStringIO
import numpy as np
from PIL import Image
from sklearn import svm
from sklearn import tree
from sklearn import metrics
from sklearn import neighbors
from sklearn import cross_validation
from NeuralNetwork import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer

for i in range(1,50):
    url="http://10.249.11.1/rand.jsp?tSessionId="
    time=str(np.random.random())
    getCode_url = url+time
    request = urllib2.Request(getCode_url)
    res = urllib2.urlopen(request).read()
    image = Image.open(cStringIO.StringIO(res))
    
    pic_path="/home/xuefliang/VerificationCode/"
    pic_name=str(i)
    #image=Image.open(pic_path+pic_name)
    img_grey = image.convert('L')  # 转化为灰度图
    #img_grey.save(pic_path+pic_name)
    #二值化处理
    threshold = 140
    table = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)
    img_out= img_grey.point(table, '1')
    #img_out.save(pic_path+pic_name+".jpg")
    
    child_img_list = []
    for j in range(4):
        x = 7 + j * (11 + 1)  # 左边距、单个图片宽度、间隔
        y = 1
        child_img = img_out.crop((x, y, x + 11, y + 20)) #单个图片的宽度和高度
        child_img_list.append(child_img)
        child_img_list[j].save(pic_path+pic_name+str(j)+".jpg")

x=[]
y=[]
for i in range(0,10):
    pic_path = '/home/xuefliang/VerificationCode/'
    pic_name=str(i)
    for name in glob.glob(pic_path+pic_name+'/*.jpg'):
        img = Image.open(name)
        x.append(np.array(img).flatten())
        y.append(i)
X=np.vstack(x)
y=np.vstack(y)

#随机抽取生成训练集和测试集，其中训练集的比例为80%，测试集20%
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=1)
#查看训练集的行数
X_train.shape,y_train.shape

#生成SVM分类模型,核函数 
svm_clf = svm.SVC(C=1.0,kernel='poly',degree=3, gamma='auto', coef0=0.0)
#使用训练集对svm分类模型进行训练
svm_clf.fit(X_train, y_train.ravel())
#使用测试集衡量分类模型准确率
svm_clf.score(X_test, y_test)
#对测试集数据进行预测
predicted=svm_clf.predict(X_test)

svm_clf.predict(X_test)
#查看测试集中的真实结果
expected=y_test
#生成准确率的混淆矩阵(Confusion matrix)
print metrics.confusion_matrix(expected, predicted)
print metrics.classification_report(y_test, predicted)

#Dtree
Dtree_clf=tree.DecisionTreeClassifier(criterion='gini')
Dtree_clf.fit(X_train, y_train)
Dtree_clf.score(X_test, y_test)
predicted=Dtree_clf.predict(X_test)
expected=y_test
print metrics.confusion_matrix(expected, predicted)
print metrics.classification_report(y_test, predicted)

#KNN
knn_clf = neighbors.KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
knn_clf.score(X_test, y_test)
predicted=knn_clf.predict(X_test)
expected=y_test
print metrics.confusion_matrix(expected, predicted)
print metrics.classification_report(y_test, predicted)

#NeuralNetwork
X -= X.min()  # normalize the values to bring them into the range 0-1
X /= X.max()
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=1)
nn_clf=NeuralNetwork([220,500,10],'logistic')
labels_train = LabelBinarizer().fit_transform(y_train)
labels_test = LabelBinarizer().fit_transform(y_test)
nn_clf.fit(X_train, labels_train, epochs=5000)
predicted = []
for i in range(X_test.shape[0]):
    o = nn_clf.predict(X_test[i])
    predicted.append(np.argmax(o))
print metrics.confusion_matrix(y_test, predicted)
print metrics.classification_report(y_test, predicted)


url="http://10.249.11.1/rand.jsp"
request = urllib2.Request(url)
res = urllib2.urlopen(request).read()
image = Image.open(cStringIO.StringIO(res))
#image.save('/home/xuefliang/VerificationCode/imgcode.jpg')
img_grey = image.convert('L')
threshold = 140
table = []
for i in range(256):
    if i < threshold:
        table.append(0)
    else:
        table.append(1)
image= img_grey.point(table, '1')

child_img_list = []
number=[]
for j in range(4):
    x = 7 + j * (11 + 1)  # 左边距、单个图片宽度、间隔
    y = 1
    child_img = image.crop((x, y, x + 11, y + 20)) #单个图片的宽度和高度
    child_img_list.append(child_img)
    child_img_list[j].save('code.jpg')
    number.append(svm_clf.predict(np.array(Image.open('code.jpg')).flatten())[0])
np.asarray(number)