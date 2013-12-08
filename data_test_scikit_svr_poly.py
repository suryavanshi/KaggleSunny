import numpy
import csv
import nltk
import json
import sys
import string
import pickle
import time
import datetime
from numpy import genfromtxt
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
import re
import pandas as pd1
#from nltk.corpus import wordnet as wn

#t = pd1.read_csv('Tags_Train_11Oct2013.csv',usecols=['Tag'])
#v_tag = list(t.Tag)
#print v_tag[0:10]
re1 = re.compile('\d+x\d+')
re2 = re.compile('\d+.\d*')
re3 = re.compile(r'\\')
re4 = re.compile(r'`|@')
re5 = re.compile(r'"+')
def remove_html_tags(data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)

def replace_chr(str1):
    chr1 = ['!','*',':','$','(',')','=',';','[',']','~','{','}','|','&','%','\xe2',
            '\x80','#']
    for e in chr1:
        str1 = str1.replace(e,' ')
    str1 = str1.replace('_',' ')
    str1 = str1.replace('/',' ')
    #str1 = str1.replace('.',' ')
    str1 = re1.sub('',str1)
    str1 = re2.sub('',str1)
    str1 = re3.sub(' ',str1)
    
    return str1
def remove_prepo(str_list):
    
    #str_list=list(set(str_list))
    for e in prepo:
        if e in str_list:
            str_list.remove(e)
    return str_list
striplist = [',','"','$','=',"'",'.','-','@','?','`','(',')',':',"//",'*']
smalltag = ['c','r','c#','qt','ip','3d','io','f#','hp','2d','go','gd','ls','sh',
            'vi','3g','tv','v8','rm','cp','ps','vm','tk','su','cd','su','pi','db']

long_tag = ['content-management-system','probability-distributions']
def strip_chr(str_list2):
    nlist4 = []
    
    for e in str_list2:
        #try:
        #    e.decode('utf-8')
        #except:
        #    print e, rown, str_list2
        #    str_list2.remove
        #eutf8 = filter(lambda x: x in string.printable, e)
        #e = re1.sub('',e)
        e = e.replace("'s",'')
        e = re4.sub('',e)
        e = re5.sub('',e)
        
        for c in striplist:
            if e!='.net': e = e.strip(c)
        if e.isdigit()==False:
            if(e in smalltag or len(e)>2):
                if(e in long_tag or len(e)<25):
                    nlist4.append(e)
    return nlist4

##def wordnetcheck(str_list2):
##    nlist4 = []
##    
##    for e in str_list2:
##        if (wn.synsets(e)): nlist4.append(e)
##    return nlist4

prepo = ['a','no','in','the','an','is','not','do','if','how','and','but','my',
             'thanks','of','will','at','have','this','so','on','only','can','any',
             'to','from','all','has','need','that','for','when','they','are','i','we',
             'want','it','use','able','using','am',"i'm",'seems','even',"they've",'into',
             'between','these','output','inside','getting','thus',"can't",'user','make',
             'get','writing','where','remove','like','find','why',"i've",'en','na','www',
             'what','by','just','or','does','as','you','there','eg','gives','without',
             'way','which','check','{','}','with','$','apart','=','==','-','+','me',"i'd",
             'I','look','often','still','run','simple','simply','problem',"that's",'lack',
             'fetch','help','work','assume','know','could','think','try','error','bad',
             'street','thank','right','change','new','really','see','guess','sure',
             'prevent','add','type','instead','machine','machines','requests','replace',
             'old','would','one','different','multiple','box','connecting','next','due'
             'receive','control','many','found','name','based','may','best','first',
             'makes','extensive','rules','process','somehow','comes','avoided','rewrites',
             'via','unfortunately','roughly','second','rewriting','application','product',
             'products','page','existing','rewrite','site','location','refresh','refreshing',
             'whole',"don't",'pull','information','create','creating','else','every','keep',
             'updating','single','taking','interested','foo','return','method','line','say',
             'end','title','include','generate','iterate','works','faster','someone','free',
             'sends','two','initiate','congratulations','reward','non','blvd','ave','white']
nltk_stop = nltk.corpus.stopwords.words('english') #stop words from nltk
stop_words = list(set(prepo+nltk_stop))


nr1 = 2013337
#nr2 = 5000
#nr2 =  2013337 - 3*nr1

#t = pd1.read_csv('train.csv',usecols=['tag_type']) #read 10 rows
t2 = pd1.read_csv('train.csv')

#t2 = pd1.read_csv('Test.csv',nrows=nr1)
#print t2
cnt = 0

start_time = time.time()
          
rown = 0
word_dict = {} #using words in the dictionary
max_cnt = 0
cnt=0

stm = nltk.stem.snowball.EnglishStemmer()
word_311=[]
sd = len(t2)
st1 = time.time()
for i in range(sd):
    try:
        texts = str(t2.tweet[i])
        #print "Here->>",texts,i
        #texts = str(texts)
        texts = texts.lower()
        texts = replace_chr(texts)
        texts = unicode(texts,errors='replace')
        #texts = filter(lambda x: x in string.printable, texts)
        word1 = texts.split()
        word1 = strip_chr(word1)
    except Exception as ex:
        print "here->",t2.description[i],texts,ex,i
        break
    word1 = [w for w in word1 if not w in nltk_stop]
    word1 = [stm.stem(w) for w in word1]
    for e in word1:
        word_311.append(e)
    #if i== 15: break
print "Time taken for all words->",time.time() - st1
all_words = nltk.FreqDist(w for w in word_311)
word_features = all_words.keys()[:12]
##word_features.remove("i'm")
##word_features.remove('get')
##word_features.remove('go')
word_features.remove("link")
word_features.remove("mention")
word_features.append(u'snow')
word_features.append(u'wind')
word_features.append(u'humid')
word_features.append(u'good')
word_features.append(u'cold')
word_features.append(u'warm')
print word_features

t2 = pd1.read_csv('train.csv',nrows=60000)
sd=len(t2)

train_s1=[]

try:
    #reader = csv.reader(f)
    for i in range(sd):
        #print row
        features={}
        texts = str(t2.tweet[i])
        texts = texts.lower()
        texts = replace_chr(texts)
        texts = unicode(texts,errors='replace')
        word1 = texts.split()
        word1 = strip_chr(word1)
        word1 = [w for w in word1 if not w in nltk_stop]
        word1 = [stm.stem(w) for w in word1]
##        if 'potholes' in word1:
##            word1.remove('potholes')
##            word1.append('pothole')
        for word in word_features:
            features['con(%s)' % word] = (word in word1)

        #features['state']=t2.state[i]
        train_s1.append(features)
                                     
        rown = rown + 1
        if (rown==10000 or rown==20000 or rown==30000 or rown==40000):
            print rown,"time=",datetime.datetime.now().time()
        #if rown==15: break
finally:
    print "Done feature"    

#print train_vote
vec = DictVectorizer()
vec_s1 = vec.fit_transform(train_s1).toarray()
clf_s1 = svm.SVR(kernel='rbf')
clf_s1.fit(vec_s1,t2.s1[0:])
clf_s2 = svm.SVR(kernel='rbf')
clf_s2.fit(vec_s1,t2.s2[0:])
clf_s3 = svm.SVR(kernel='rbf')
clf_s3.fit(vec_s1,t2.s3[0:])
clf_s4 = svm.SVR(kernel='rbf')
clf_s4.fit(vec_s1,t2.s4[0:])
clf_s5 = svm.SVR(kernel='rbf')
clf_s5.fit(vec_s1,t2.s5[0:])
print "done s5"
clf_w1 = svm.SVR(kernel='rbf')
clf_w1.fit(vec_s1,t2.w1[0:])
clf_w2 = svm.SVR(kernel='rbf')
clf_w2.fit(vec_s1,t2.w2[0:])
clf_w3 = svm.SVR(kernel='rbf')
clf_w3.fit(vec_s1,t2.w3[0:])
clf_w4 = svm.SVR(kernel='rbf')
clf_w4.fit(vec_s1,t2.w4[0:])
print "done w4"
clf_k1 = svm.SVR(kernel='rbf')
clf_k1.fit(vec_s1,t2.k1[0:])
clf_k2 = svm.SVR(kernel='rbf')
clf_k2.fit(vec_s1,t2.k2[0:])
clf_k3 = svm.SVR(kernel='rbf')
clf_k3.fit(vec_s1,t2.k3[0:])
clf_k4 = svm.SVR(kernel='rbf')
clf_k4.fit(vec_s1,t2.k4[0:])
clf_k5 = svm.SVR(kernel='rbf')
clf_k5.fit(vec_s1,t2.k5[0:])
print "done k5"
clf_k6 = svm.SVR(kernel='rbf')
clf_k6.fit(vec_s1,t2.k6[0:])
clf_k7 = svm.SVR(kernel='rbf')
clf_k7.fit(vec_s1,t2.k7[0:])
clf_k8 = svm.SVR(kernel='rbf')
clf_k8.fit(vec_s1,t2.k8[0:])
clf_k9 = svm.SVR(kernel='rbf')
clf_k9.fit(vec_s1,t2.k9[0:])
clf_k10 = svm.SVR(kernel='rbf')
clf_k10.fit(vec_s1,t2.k10[0:])
print "done k10"
clf_k11 = svm.SVR(kernel='rbf')
clf_k11.fit(vec_s1,t2.k11[0:])
clf_k12 = svm.SVR(kernel='rbf')
clf_k12.fit(vec_s1,t2.k12[0:])
clf_k13 = svm.SVR(kernel='rbf')
clf_k13.fit(vec_s1,t2.k13[0:])
clf_k14 = svm.SVR(kernel='rbf')
clf_k14.fit(vec_s1,t2.k14[0:])
clf_k15 = svm.SVR(kernel='rbf')
clf_k15.fit(vec_s1,t2.k15[0:])

print "done training"
print "Time taken = ",time.time() - start_time


rown=0
#ofile= open('test_autotag2.csv', 'wb')
of2 = open('weath_pred_sci_23Oct_rbf60k.csv','wb')
writer2 = csv.writer(of2, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
writer2.writerow(["id","s1","s2","s3",'s4','s5','w1','w2','w3','w4','k1','k2','k3',
                  'k4','k5','k6','k7','k8','k9','k10','k11','k12','k13','k14','k15'])

##writer = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
##writer.writerow(["id","latitude","longitude","summary","description","source",
##                 "created_time","tag_type"])
#f2 = open('test.csv','rb')
t2 = pd1.read_csv('test.csv')
sd = len(t2)
try:
    #reader = csv.reader(f2)
    for i in range(sd):
        #print row
        features={}
        texts = str(t2.tweet[i])
        texts = texts.lower()
        texts = replace_chr(texts)
        texts = unicode(texts,errors='replace')
        word1 = texts.split()
        word1 = strip_chr(word1)
        word1 = [w for w in word1 if not w in nltk_stop]
        word1 = [stm.stem(w) for w in word1]
##        if 'potholes' in word1:
##            word1.remove('potholes')
##            word1.append('pothole')
        for word in word_features:
            features['contains(%s)' % word] = (word in word1)

        vec_s2 = vec.fit_transform(features).toarray()
        #features['state']=t2.state[i]
        s1 = clf_s1.predict(vec_s2)[0]
        s2 = clf_s2.predict(vec_s2)[0]
        s3 = clf_s3.predict(vec_s2)[0]
        s4 = clf_s4.predict(vec_s2)[0]
        s5 = clf_s5.predict(vec_s2)[0]
        w1 = clf_w1.predict(vec_s2)[0]
        w2 = clf_w2.predict(vec_s2)[0]
        w3 = clf_w3.predict(vec_s2)[0]
        w4 = clf_w4.predict(vec_s2)[0]
        k1 = clf_k1.predict(vec_s2)[0]
        k2 = clf_k2.predict(vec_s2)[0]
        k3 = clf_k3.predict(vec_s2)[0]
        k4 = clf_k4.predict(vec_s2)[0]
        k5 = clf_k5.predict(vec_s2)[0]
        k6 = clf_k6.predict(vec_s2)[0]
        k7 = clf_k7.predict(vec_s2)[0]
        k8 = clf_k8.predict(vec_s2)[0]
        k9 = clf_k9.predict(vec_s2)[0]
        k10 = clf_k10.predict(vec_s2)[0]
        k11 = clf_k11.predict(vec_s2)[0]
        k12 = clf_k12.predict(vec_s2)[0]
        k13 = clf_k13.predict(vec_s2)[0]
        k14 = clf_k14.predict(vec_s2)[0]
        k15 = clf_k15.predict(vec_s2)[0]
        #print "s1->",s1
        writer2.writerow([int(t2.id[i]),s1,s2,s3,s4,s5,w1,w2,w3,w4,k1,k2,k3,k4,k5,k6,
                          k7,k8,k9,k10,k11,k12,k13,k14,k15])

        #print s1,s2,s3,s4,s5  
        rown = rown + 1
        if (rown==10000 or rown==20000 or rown==30000 or rown==40000):
            print rown,"time=",datetime.datetime.now().time()
        #if rown==15: break
finally:
    print "done"

#ofile.close()
of2.close()
print "done result"
            
##import pickle
##f = open('vote_class_30feat.pickle', 'wb')
##pickle.dump(cl_vote, f)
##f.close()
##f = open('view_class_30feat.pickle', 'wb')
##pickle.dump(cl_view, f)
##f.close()
##f = open('comm_class_30feat.pickle', 'wb')
##pickle.dump(cl_comm, f)
##f.close()
