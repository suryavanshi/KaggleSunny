import csv
import numpy
import nltk
import json
import sys
import string
import pickle
import time
import datetime
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from sklearn.linear_model import BayesianRidge
from sklearn import svm
from sklearn import neighbors
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pandas as pd1
t2 = pd1.read_csv('train.csv')
#Total 77946 rows in train
per80 = int(0.8*len(t2))
train = t2[0:per80]
cv = t2[per80+1:]
vec_s1=0
vecp_cv=0
tfidfc=0
print "done read",datetime.datetime.now().time()

tfidf = TfidfVectorizer(min_df=10,max_features=None, strip_accents='unicode', analyzer='word')
tfidf.fit(train['tweet'])
X = tfidf.transform(train['tweet'])
#y = np.array(t1.ix[:,4:])
vec_s1 = X
vecp_cv = tfidf.transform(cv['tweet'])

print "done transform for",datetime.datetime.now().time()
    
mind={}
alpd={}
minidf={}
def calmetric(icv_p,pred,str1,alp):
    mae = metrics.mean_absolute_error(icv_p,pred)
    print "Param: %s, MAE:%f (alp): %f" % (str1,mae,alp)
    if str1 in mind.keys():
        if mae<mind[str1]:
            mind[str1]=mae
            alpd[str1]=alp
            
    else:
        mind[str1]=mae
        alpd[str1]=alp
        
    return mae
y_s1 = list(train.s1[0:])
y_s2 = list(train.s2[0:])
y_s3 = list(train.s3[0:])
y_s4 = list(train.s4[0:])
y_s5 = list(train.s5[0:])
y_w1 = list(train.w1[0:])
y_w2 = list(train.w2[0:])
y_w3 = list(train.w3[0:])
y_w4 = list(train.w4[0:])
y_k1 = list(train.k1[0:])
y_k2 = list(train.k2[0:])
y_k3 = list(train.k3[0:])
y_k4 = list(train.k4[0:])
y_k5 = list(train.k5[0:])
y_k6 = list(train.k6[0:])
y_k7 = list(train.k7[0:])
y_k8 = list(train.k8[0:])
y_k9 = list(train.k9[0:])
y_k10 = list(train.k10[0:])
y_k11 = list(train.k11[0:])
y_k12 = list(train.k12[0:])
y_k13 = list(train.k13[0:])
y_k14 = list(train.k14[0:])
y_k15 = list(train.k15[0:])
t2=0
X = 0
start_time = time.time()
##m_s1=0;m_s2=0;m_s3=0;m_s4=0;m_s5=0;
##m_w1=0;m_w2=0;m_w3=0;m_w4=0;
##m_k1
minp=[100]*16
def modelfit(alp):
    clf_s1 = linear_model.ElasticNet(alpha = alp)
    clf_s1.fit(vec_s1,y_s1)
    print "Done s1->",datetime.datetime.now().time()
    clf_s2 = linear_model.ElasticNet(alpha = alp)
    clf_s2.fit(vec_s1,y_s2)
    print "Done s2->",datetime.datetime.now().time()
    clf_s3 = linear_model.ElasticNet(alpha = alp)
    clf_s3.fit(vec_s1,y_s3)
    clf_s4 = linear_model.ElasticNet(alpha = alp)
    clf_s4.fit(vec_s1,y_s4)
    clf_s5 = linear_model.ElasticNet(alpha = alp)
    clf_s5.fit(vec_s1,y_s5)
    clf_w1 = linear_model.ElasticNet(alpha = alp)
    clf_w1.fit(vec_s1,y_w1)
    clf_w2 = linear_model.ElasticNet(alpha = alp)
    clf_w2.fit(vec_s1,y_w2)
    clf_w3 = linear_model.ElasticNet(alpha = alp)
    clf_w3.fit(vec_s1,y_w3)
    clf_w4 = linear_model.ElasticNet(alpha = alp)
    clf_w4.fit(vec_s1,y_w4)
    print "Done w4->",datetime.datetime.now().time()
    clf_k1 = linear_model.ElasticNet(alpha = alp)
    clf_k1.fit(vec_s1,y_k1)
    clf_k2 = linear_model.ElasticNet(alpha = alp)
    clf_k2.fit(vec_s1,y_k2)
    clf_k3 = linear_model.ElasticNet(alpha = alp)
    clf_k3.fit(vec_s1,y_k3)
    clf_k4 = linear_model.ElasticNet(alpha = alp)
    clf_k4.fit(vec_s1,y_k4)
    clf_k5 = linear_model.ElasticNet(alpha = alp)
    clf_k5.fit(vec_s1,y_k5)
    print "Done k5->",datetime.datetime.now().time()
    clf_k6 = linear_model.ElasticNet(alpha = alp)
    clf_k6.fit(vec_s1,y_k6)
    clf_k7 = linear_model.ElasticNet(alpha = alp)
    clf_k7.fit(vec_s1,y_k7)
    clf_k8 = linear_model.ElasticNet(alpha = alp)
    clf_k8.fit(vec_s1,y_k8)
    clf_k9 = linear_model.ElasticNet(alpha = alp)
    clf_k9.fit(vec_s1,y_k9)
    clf_k10 = linear_model.ElasticNet(alpha = alp)
    clf_k10.fit(vec_s1,y_k10)
    print "Done k10->",datetime.datetime.now().time()
    clf_k11 = linear_model.ElasticNet(alpha = alp)
    clf_k11.fit(vec_s1,y_k11)
    clf_k12 = linear_model.ElasticNet(alpha = alp)
    clf_k12.fit(vec_s1,y_k12)
    clf_k13 = linear_model.ElasticNet(alpha = alp)
    clf_k13.fit(vec_s1,y_k13)
    clf_k14 = linear_model.ElasticNet(alpha = alp)
    clf_k14.fit(vec_s1,y_k14)
    clf_k15 = linear_model.ElasticNet(alpha = alp)
    clf_k15.fit(vec_s1,y_k15)
    print "Done k15->",datetime.datetime.now().time()
    
    s1 = list(clf_s1.predict(vecp_cv))
    maer=calmetric(cv.s1[0:],s1,'s1',alp)
    s2 = list(clf_s2.predict(vecp_cv))
    maer=calmetric(cv.s2[0:],s2,'s2',alp)
    s3 = list(clf_s3.predict(vecp_cv))
    maer=calmetric(cv.s3[0:],s3,'s3',alp)
    s4 = list(clf_s4.predict(vecp_cv))
    maer=calmetric(cv.s4[0:],s4,'s4',alp)
    s5 = list(clf_s5.predict(vecp_cv))
    maer=calmetric(cv.s5[0:],s5,'s5',alp)
    w1 = list(clf_w1.predict(vecp_cv))
    maer=calmetric(cv.w1[0:],w1,'w1',alp)
    w2 = list(clf_w2.predict(vecp_cv))
    maer=calmetric(cv.w2[0:],w2,'w2',alp)
    w3 = list(clf_w3.predict(vecp_cv))
    maer=calmetric(cv.w3[0:],w3,'w3',alp)
    w4 = list(clf_w4.predict(vecp_cv))
    maer=calmetric(cv.w4[0:],w4,'w4',alp)
    k1 = list(clf_k1.predict(vecp_cv))
    maer=calmetric(cv.k1[0:],k1,'k1',alp)
    k2 = list(clf_k2.predict(vecp_cv))
    maer=calmetric(cv.k2[0:],k2,'k2',alp)
    k3 = list(clf_k3.predict(vecp_cv))
    maer=calmetric(cv.k3[0:],k3,'k3',alp)
    k4 = list(clf_k4.predict(vecp_cv))
    maer=calmetric(cv.k4[0:],k4,'k4',alp)
    k5 = list(clf_k5.predict(vecp_cv))
    maer=calmetric(cv.k5[0:],k5,'k5',alp)
    k6 = list(clf_k6.predict(vecp_cv))
    maer=calmetric(cv.k6[0:],k6,'k6',alp)
    k7 = list(clf_k7.predict(vecp_cv))
    maer=calmetric(cv.k7[0:],k7,'k7',alp)
    k8 = list(clf_k8.predict(vecp_cv))
    maer=calmetric(cv.k8[0:],k8,'k8',alp)
    k9 = list(clf_k9.predict(vecp_cv))
    maer=calmetric(cv.k9[0:],k9,'k9',alp)
    k10 = list(clf_k10.predict(vecp_cv))
    maer=calmetric(cv.k10[0:],k10,'k10',alp)
    k11 = list(clf_k11.predict(vecp_cv))
    maer=calmetric(cv.k11[0:],k11,'k11',alp)
    k12 = list(clf_k12.predict(vecp_cv))
    maer=calmetric(cv.k12[0:],k12,'k12',alp)
    k13 = list(clf_k13.predict(vecp_cv))
    maer=calmetric(cv.k13[0:],k13,'k13',alp)
    k14 = list(clf_k14.predict(vecp_cv))
    maer=calmetric(cv.k14[0:],k14,'k14',alp)
    k15 = list(clf_k15.predict(vecp_cv))
    maer=calmetric(cv.k15[0:],k15,'k15',alp)

alp = [1e-6,1e-5,1e-4,1e-3]
#alp=[1.0,2.0]

for a in alp:
    modelfit(a)
print "done training"
vec_s1 = 0
print "Time taken = ",time.time() - start_time



