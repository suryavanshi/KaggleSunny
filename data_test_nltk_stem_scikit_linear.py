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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm.sparse import LinearSVC
from sklearn.linear_model import SGDRegressor
from sklearn import linear_model
from sklearn import svm
from sklearn import neighbors
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize, sent_tokenize
import re
import pandas as pd1

vectorizer = CountVectorizer(min_df=1)
analyze = vectorizer.build_analyzer()

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

state_dict = {'AK': 'Alaska','AL': 'Alabama','AR': 'Arkansas','AS': 'American Samoa',
        'AZ': 'Arizona','CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}
nltk_stop = nltk.corpus.stopwords.words('english') #stop words from nltk
#stop_words = list(set(prepo+nltk_stop))


nr1 = 2013337
#nr2 = 5000
#nr2 =  2013337 - 3*nr1

#t = pd1.read_csv('train.csv',usecols=['tag_type']) #read 10 rows
t2 = pd1.read_csv('train.csv')

cnt = 0

start_time = time.time()
          
rown = 0
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
    #if i==5: break
print "Time taken for all words->",time.time() - st1
all_words = nltk.FreqDist(w for w in word_311)
word_features = all_words.keys()[:500]
#print "45 keys->",all_words.keys()[:70]
top_words = [u'weather', u'mention', u'link', u'day', u'storm', u'sunni', u'hot', u'today',
             u'outsid', u'rain', u'like', u'sunshin', u'degre', u'thunderstorm', u'feel',
             u'humid', u'wind', u'cold', u'raini', u'mph', u'good', u'love', u'snow', u'sever',
             u'nice', u'may', u'warm', u'great', u'look', u'time', u'lol', u'beauti', u'hope',
             u'come', u'morn', u'freez', u'enjoy', u'make', u'warn', u'watch', u'windi',
             u'perfect', u'got', u'week', u'need', u'work', u'tonight', u'right', u'back',
             u'high', u'summer', u'forecast', u'tornado', u'one', u'weekend', u'still',
             u"don't", u'counti', u'see', u'know', u'realli', u'think', u'new', u'tomorrow',
             u'want', u'night', u'sun', u'bad', u'even', u'thank', u'fuck', u'way', u'better',
             u'chilli', u'last', u'home', u'take']
#word_features = top_words[0:60]
#word_features.remove('like')
#word_features.remove('mention')
#word_features.remove('get')
#word_features.remove('link')
#word_features.remove('today')
#word_features.remove('outsid')
#word_features.remove('mph')
#word_features.remove("i'm")
#word_features.remove('go')
#word_features.remove('feel')
#word_features.remove('day')
#word_features.remove('lol')
#word_features.append(u'snow')
#word_features.append(u'good')

#print word_features
print "done word count"
state = ['wyoming', 'north dakota', 'nebraska', 'washington', 'rhode island', 'alaska',
         'iowa', 'nevada', 'maine', 'tennessee', 'colorado', 'mississippi',
         'south dakota', 'new jersey', 'oklahoma', 'delaware', 'minnesota',
         'north carolina', 'illinois', 'new york', 'arkansas', 'indiana', 'maryland',
         'louisiana', 'texas', 'arizona', 'wisconsin', 'michigan', 'kansas', 'utah',
         'virginia', 'oregon', 'connecticut', 'montana', 'california', 'new mexico',
         'vermont', 'georgia', 'pennsylvania', 'florida', 'hawaii', 'kentucky',
         'missouri', 'district of columbia', 'new hampshire', 'idaho', 'west virginia',
         'south carolina', 'ohio', 'alabama', 'massachusetts','nan']

#t2 = pd1.read_csv('train.csv')
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

        for word in word_features:
            features['con(%s)' % word] = (word in word1)
##        for st in state:
##            features['%s' % st] = st in t2.state[i]
        #features['state']=t2.state[i]
        train_s1.append(features)

                             
        rown = rown + 1
        if (rown==10000 or rown==20000 or rown==30000 or rown==40000):
            print rown,"time=",datetime.datetime.now().time()
        #if rown==100: break
finally:
    print "Done feature"    

vec = DictVectorizer()
vec_s1 = vec.fit_transform(train_s1).toarray() #convert features to numarray
#1 if feature is present 0 if not present
#print vec_s1.shape #[n_samples,n_features]
print vec.get_feature_names() #prints the features
print "Shape of feature->",vec_s1.shape

train_s1 = 0
y_s1 = list(t2.s1[0:])
y_s2 = list(t2.s2[0:])
y_s3 = list(t2.s3[0:])
y_s4 = list(t2.s4[0:])
y_s5 = list(t2.s5[0:])
y_w1 = list(t2.w1[0:])
y_w2 = list(t2.w2[0:])
y_w3 = list(t2.w3[0:])
y_w4 = list(t2.w4[0:])
y_k1 = list(t2.k1[0:])
y_k2 = list(t2.k2[0:])
y_k3 = list(t2.k3[0:])
y_k4 = list(t2.k4[0:])
y_k5 = list(t2.k5[0:])
y_k6 = list(t2.k6[0:])
y_k7 = list(t2.k7[0:])
y_k8 = list(t2.k8[0:])
y_k9 = list(t2.k9[0:])
y_k10 = list(t2.k10[0:])
y_k11 = list(t2.k11[0:])
y_k12 = list(t2.k12[0:])
y_k13 = list(t2.k13[0:])
y_k14 = list(t2.k14[0:])
y_k15 = list(t2.k15[0:])

clf_s1 = linear_model.Ridge(alpha = .3)
clf_s1.fit(vec_s1,y_s1)
print "Done s1->",datetime.datetime.now().time()
clf_s2 = linear_model.Ridge(alpha = .3)
clf_s2.fit(vec_s1,y_s2)
print "Done s2->",datetime.datetime.now().time()
clf_s3 = linear_model.Ridge(alpha = .3)
clf_s3.fit(vec_s1,y_s3)
clf_s4 = linear_model.Ridge(alpha = .3)
clf_s4.fit(vec_s1,y_s4)
clf_s5 = linear_model.Ridge(alpha = .3)
clf_s5.fit(vec_s1,y_s5)
clf_w1 = linear_model.Ridge(alpha = .3)
clf_w1.fit(vec_s1,y_w1)
clf_w2 = linear_model.Ridge(alpha = .3)
clf_w2.fit(vec_s1,y_w2)
clf_w3 = linear_model.Ridge(alpha = .3)
clf_w3.fit(vec_s1,y_w3)
clf_w4 = linear_model.Ridge(alpha = .3)
clf_w4.fit(vec_s1,y_w4)
print "Done w4->",datetime.datetime.now().time()
clf_k1 = linear_model.Ridge(alpha = .3)
clf_k1.fit(vec_s1,y_k1)
clf_k2 = linear_model.Ridge(alpha = .3)
clf_k2.fit(vec_s1,y_k2)
clf_k3 = linear_model.Ridge(alpha = .3)
clf_k3.fit(vec_s1,y_k3)
clf_k4 = linear_model.Ridge(alpha = .3)
clf_k4.fit(vec_s1,y_k4)
clf_k5 = linear_model.Ridge(alpha = .3)
clf_k5.fit(vec_s1,y_k5)
print "Done k5->",datetime.datetime.now().time()
clf_k6 = linear_model.Ridge(alpha = .3)
clf_k6.fit(vec_s1,y_k6)
clf_k7 = linear_model.Ridge(alpha = .3)
clf_k7.fit(vec_s1,y_k7)
clf_k8 = linear_model.Ridge(alpha = .3)
clf_k8.fit(vec_s1,y_k8)
clf_k9 = linear_model.Ridge(alpha = .3)
clf_k9.fit(vec_s1,y_k9)
clf_k10 = linear_model.Ridge(alpha = .3)
clf_k10.fit(vec_s1,y_k10)
print "Done k10->",datetime.datetime.now().time()
clf_k11 = linear_model.Ridge(alpha = .3)
clf_k11.fit(vec_s1,y_k11)
clf_k12 = linear_model.Ridge(alpha = .3)
clf_k12.fit(vec_s1,y_k12)
clf_k13 = linear_model.Ridge(alpha = .3)
clf_k13.fit(vec_s1,y_k13)
clf_k14 = linear_model.Ridge(alpha = .3)
clf_k14.fit(vec_s1,y_k14)
clf_k15 = linear_model.Ridge(alpha = .3)
clf_k15.fit(vec_s1,y_k15)
print "Done k15->",datetime.datetime.now().time()

print "done training"
print "Time taken = ",time.time() - start_time


rown=0
#ofile= open('test_autotag2.csv', 'wb')
of2 = open('test_pred_sci_11Nov1_ridge500f.csv','wb')
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
##        for st in state:
##            features['%s' % st] = st in str(t2.state[i])
        vecp_s1 = vec.fit_transform([features]).toarray()
        #print vecp_s1
        s1 = clf_s1.predict(vecp_s1)[0]
        s2 = clf_s2.predict(vecp_s1)[0]
        s3 = clf_s3.predict(vecp_s1)[0]
        s4 = clf_s4.predict(vecp_s1)[0]
        s5 = clf_s5.predict(vecp_s1)[0]
        w1 = clf_w1.predict(vecp_s1)[0]
        w2 = clf_w2.predict(vecp_s1)[0]
        w3 = clf_w3.predict(vecp_s1)[0]
        w4 = clf_w4.predict(vecp_s1)[0]
        k1 = clf_k1.predict(vecp_s1)[0]
        k2 = clf_k2.predict(vecp_s1)[0]
        k3 = clf_k3.predict(vecp_s1)[0]
        k4 = clf_k4.predict(vecp_s1)[0]
        k5 = clf_k5.predict(vecp_s1)[0]
        k6 = clf_k6.predict(vecp_s1)[0]
        k7 = clf_k7.predict(vecp_s1)[0]
        k8 = clf_k8.predict(vecp_s1)[0]
        k9 = clf_k9.predict(vecp_s1)[0]
        k10 = clf_k10.predict(vecp_s1)[0]
        k11 = clf_k11.predict(vecp_s1)[0]
        k12 = clf_k12.predict(vecp_s1)[0]
        k13 = clf_k13.predict(vecp_s1)[0]
        k14 = clf_k14.predict(vecp_s1)[0]
        k15 = clf_k15.predict(vecp_s1)[0]
        
        #print "s1->",s1[0]
        #features['state']=t2.state[i]
       
        writer2.writerow([int(t2.id[i]),s1,s2,s3,s4,s5,w1,w2,w3,w4,k1,k2,k3,k4,k5,k6,
                          k7,k8,k9,k10,k11,k12,k13,k14,k15])

        #print s1,s2,s3,s4,s5  
        rown = rown + 1
        if (rown==10000 or rown==20000 or rown==30000 or rown==40000):
            print rown,"time=",datetime.datetime.now().time()
        #if rown==20: break
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
