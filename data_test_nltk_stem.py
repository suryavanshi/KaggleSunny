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
word_features = all_words.keys()[:43]
word_features.remove("i'm")
word_features.remove('get')
word_features.remove('go')
print word_features

train_s1=[]
train_s2=[]
train_s3=[]
train_s4=[];train_s5=[]
train_w1=[]
train_w2=[]
train_w3=[]
train_w4=[]
train_k1=[]
train_k2=[]
train_k3=[]
train_k4=[];train_k5=[];train_k6=[]
train_k7=[];train_k8=[];train_k9=[];train_k10=[];train_k11=[];train_k12=[];train_k13=[]
train_k14=[];train_k15=[]
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

        features['state']=t2.state[i]
        train_s1.append((features,t2.s1[i]*1000.0))
        train_s2.append((features,t2.s2[i]*1000.0))
        train_s3.append((features,t2.s3[i]*1000.0))
        train_s4.append((features,t2.s4[i]*1000.0))
        train_s5.append((features,t2.s5[i]*1000.0))
        train_w1.append((features,t2.w1[i]*1000.0))
        train_w2.append((features,t2.w2[i]*1000.0))
        train_w3.append((features,t2.w3[i]*1000.0))
        train_w4.append((features,t2.w4[i]*1000.0))
        train_k1.append((features,t2.k1[i]*1000.0))
        train_k2.append((features,t2.k2[i]*1000.0))
        train_k3.append((features,t2.k3[i]*1000.0))
        train_k4.append((features,t2.k4[i]*1000.0))
        train_k5.append((features,t2.k5[i]*1000.0))
        train_k6.append((features,t2.k6[i]*1000.0))
        train_k7.append((features,t2.k7[i]*1000.0))
        train_k8.append((features,t2.k8[i]*1000.0))
        train_k9.append((features,t2.k9[i]*1000.0))
        train_k10.append((features,t2.k10[i]*1000.0))
        train_k11.append((features,t2.k11[i]*1000.0))
        train_k12.append((features,t2.k12[i]*1000.0))
        train_k13.append((features,t2.k13[i]*1000.0))
        train_k14.append((features,t2.k14[i]*1000.0))
        train_k15.append((features,t2.k15[i]*1000.0))
                             
        rown = rown + 1
        if (rown==10000 or rown==20000 or rown==30000 or rown==40000):
            print rown,"time=",datetime.datetime.now().time()
        #if rown==15: break
finally:
    print "Done feature"    

#print train_vote
cl_s1 = nltk.NaiveBayesClassifier.train(train_s1)
cl_s2 = nltk.NaiveBayesClassifier.train(train_s2)
cl_s3 = nltk.NaiveBayesClassifier.train(train_s3)
cl_s4 = nltk.NaiveBayesClassifier.train(train_s4)
cl_s5 = nltk.NaiveBayesClassifier.train(train_s5)
cl_w1 = nltk.NaiveBayesClassifier.train(train_w1)
cl_w2 = nltk.NaiveBayesClassifier.train(train_w2)
cl_w3 = nltk.NaiveBayesClassifier.train(train_w3)
cl_w4 = nltk.NaiveBayesClassifier.train(train_w4)
cl_k1 = nltk.NaiveBayesClassifier.train(train_k1)
cl_k2 = nltk.NaiveBayesClassifier.train(train_k2)
cl_k3 = nltk.NaiveBayesClassifier.train(train_k3)
cl_k4 = nltk.NaiveBayesClassifier.train(train_k4)
cl_k5 = nltk.NaiveBayesClassifier.train(train_k5)
cl_k6 = nltk.NaiveBayesClassifier.train(train_k6)
cl_k7 = nltk.NaiveBayesClassifier.train(train_k7)
cl_k8 = nltk.NaiveBayesClassifier.train(train_k8)
cl_k9 = nltk.NaiveBayesClassifier.train(train_k9)
cl_k10 = nltk.NaiveBayesClassifier.train(train_k10)
cl_k11 = nltk.NaiveBayesClassifier.train(train_k11)
cl_k12 = nltk.NaiveBayesClassifier.train(train_k12)
cl_k13 = nltk.NaiveBayesClassifier.train(train_k13)
cl_k14 = nltk.NaiveBayesClassifier.train(train_k14)
cl_k15 = nltk.NaiveBayesClassifier.train(train_k15)
print "done training"
print "Time taken = ",time.time() - start_time


rown=0
#ofile= open('test_autotag2.csv', 'wb')
of2 = open('weath_pred_nltk_21Oct3_401k.csv','wb')
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

        features['state']=t2.state[i]
        s1 = float(cl_s1.classify(features))/1000.0
        s2 = float(cl_s2.classify(features))/1000.0
        s3 = float(cl_s3.classify(features))/1000.0
        s4 = float(cl_s4.classify(features))/1000.0
        s5 = float(cl_s5.classify(features))/1000.0
        w1 = float(cl_w1.classify(features))/1000.0
        w2 = float(cl_w2.classify(features))/1000.0
        w3 = float(cl_w3.classify(features))/1000.0
        w4 = float(cl_w4.classify(features))/1000.0
        k1 = float(cl_k1.classify(features))/1000.0
        k2 = float(cl_k2.classify(features))/1000.0
        k3 = float(cl_k3.classify(features))/1000.0
        k4 = float(cl_k4.classify(features))/1000.0
        k5 = float(cl_k5.classify(features))/1000.0
        k6 = float(cl_k6.classify(features))/1000.0
        k7 = float(cl_k7.classify(features))/1000.0
        k8 = float(cl_k8.classify(features))/1000.0
        k9 = float(cl_k9.classify(features))/1000.0
        k10 = float(cl_k10.classify(features))/1000.0
        k11 = float(cl_k11.classify(features))/1000.0
        k12 = float(cl_k12.classify(features))/1000.0
        k13 = float(cl_k13.classify(features))/1000.0
        k14 = float(cl_k14.classify(features))/1000.0
        k15 = float(cl_k15.classify(features))/1000.0
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
