import numpy as np
from laughing import laughing
from embedding import *
import ipdb
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import re
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet
from nltk.corpus import sentiwordnet as swn
from collections import Counter



datapath = 'SemEval2018-T4-train-taskA.txt'

def parse_dataset(fp):
    y = [] # label
    corpus = [] # tweet
    with open(fp, 'rt') as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                label = int(line.split("\t")[1])
                tweet = line.split("\t")[2]
                y.append(label)
                corpus.append(tweet)
    return corpus, y

def bow_features(corpus):

    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True).tokenize
    vectorizer = TfidfVectorizer(strip_accents="unicode", analyzer="word", tokenizer=tokenizer, stop_words="english")
    X = vectorizer.fit_transform(corpus)
    X = X.toarray()
    return X


corp, label = parse_dataset(datapath)
pattern = '(http.+(\s)?)|((@\w*\d*(\s)?))'
corp1 = []

# wipe out url in the tweet
for string in corp:
    str1 = re.sub(pattern, '', string, flags=re.MULTILINE)
    corp1.append(str1)

X = bow_features(corp) # Get the TF-IDF features
class_counts = np.asarray(np.unique(label, return_counts=True)).T.tolist()

train1 = X[0:3067,:]
test1 =X[3067:3834,:]

data = 'ANC-token-word.txt'
tokenizer = RegexpTokenizer(r'\w+')
pattern = '(http.+(\s)?)|((@\w*\d*(\s)?))'
corp1 = []
lemma = WordNetLemmatizer()
global freqdata
freqdata = defaultdict(int)

with open(data,encoding ='ISO-8859-1') as file:
        for l,i in zip(file,range(239208)):
            token, count, _ = l.split('\t')
            freqdata[token] = count

# freqdata = np.reshape(freqdata,(239208,2))
# freqdata = {}

def frequency(target):
    if(target.isdigit()):
        return float('inf')
    else:
        return freqdata[target]
        
        # start = 0
        # end = len(freqdata) - 1
        # while start <= end:
        #     middle = (start + end)// 2
        #     midpoint = freqdata[middle][0]
        #     if midpoint > target:
        #         end = middle - 1
        #     elif midpoint < target:
        #         start = middle + 1
        #     else:
        #         return int(freqdata[middle][1])
        # return int(0)

for string in corp:
    str1 = re.sub(pattern, '', string, flags=re.MULTILINE)
    corp1.append(str1)

imbalance = []
rare = []
averagefreq = []
count = 0
# j=0

# for tweet,o in zip(corp1,range(20)):
for tweet in corp1:
    # print(j)
    token = tokenizer.tokenize(tweet)
    token = [word for word in token if word not in stopwords.words('english')]
    token = [lemma.lemmatize(word) for word in token]
    # print(token)
    freqoftokens = []
    for each in token:
        each = each.lower()
        if(each.isdigit()==False):
            freqoftokens.append(frequency(each))

    # print(token,freqoftokens)

    if len(freqoftokens)!=0:
        imbalance.append(max(freqoftokens)-min(freqoftokens))
        averagefreq.append(sum(freqoftokens)/len(freqoftokens))
        rare.append(min(freqoftokens))
    else:
        count+=1
        imbalance.append(0)
        averagefreq.append(0)
        rare.append(0)

    # j+=1

print(len(imbalance),len(rare),len(averagefreq),count)

with open('freq.txt','w') as file:
    for i in range(len(imbalance)):
        file.write("%f\t%f\t%f\n" %(imbalance[i],rare[i],averagefreq[i]))




pattern = '(http.+(\s)?)|((@\w*\d*(\s)?))'
corp1 = []

for tweet in corp:
    str1 = re.sub(pattern, '', tweet, flags=re.MULTILINE)
    corp1.append(str1)


pat = '([aA]*[hH][Aa]+[Hh][HhAa]*|[Oo]?[Ll]+[Oo]+[Ll]+[OolL]*|[Rr][oO]+[Ff]+[lL]+|[Ll][Mm][Aa]+[oO]+).'
string = 'Lolll HAha Rofl lolipop LMAOOO Hahaah..'
laughing = []

for a in corp1:
    b = re.findall(pat,a)
    laughing.append(len(b))

laughing = np.reshape(laughing,(3834,-1))




datapath = 'SemEval2018-T4-train-taskA.txt'
data = 'ANC-all-count.txt'
punc = []
ellips = []
pat2 = '[(\.\.\.)]+'

for a in corp1:
    print(a)
    punc.append(a.count('!')+a.count('?')+a.count(','))
    ellips.append(len(re.findall(pat2,a)))

with open('punctuation.txt', 'w') as data_in:
    for i in range(0,len(corp)):
        data_in.write("%f\t%f\n" %(punc[i],ellips[i]))


data = 'ANC-all-count.txt'
lemma = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')
pattern = '(http.+(\s)?)|((@\w*\d*(\s)?))'
corp1 = []
for tweet in corp:
    str1 = re.sub(pattern, '', tweet, flags=re.MULTILINE)
    corp1.append(str1)

a,b=0,0
syn = wordnet.synsets('sweet')
for sy in syn:
    senti = swn.senti_synset(sy.name())
    a+=senti.pos_score()
    b+=senti.neg_score()
a = a/len(syn)
b=b/len(syn)
print('Posscore:',a,'Negscore:',b)

positive_sum = []
negative_sum = []
averagesenti = []
imbalance = []
posgap = []
neggap = []
i=0
for line in corp1:
    print(i)
    token = tokenizer.tokenize(line)
    token = [word for word in token if word not in stopwords.words('english')] # Get token without stopword
    token = [lemma.lemmatize(word) for word in token] # Lemmatize
    poseachtweet=[]
    negeachtweet=[]
    for lem in token:
        a,b=0,0
        syn = list(swn.senti_synsets(lem))

        for sy in syn:
            a+=sy.pos_score()
            b+=sy.neg_score()

        if(len(syn)!=0):
            a = a/len(syn)
            b = b/len(syn)


        poseachtweet.append(a)
        negeachtweet.append(b)
    if(len(poseachtweet)!=0):
        max_pos =max(poseachtweet)
        max_neg = max(negeachtweet)
        pos_sum = sum(poseachtweet)
        neg_sum = sum(negeachtweet)
        senti_avg = (pos_sum-neg_sum)/len(token)

    else:
        max_pos = 0
        max_neg =0
        pos_sum = 0
        neg_sum =0
        senti_avg =0

    imbal = pos_sum - neg_sum

    positive_gap = max_pos - senti_avg
    negative_gap = max_neg - senti_avg

    positive_sum.append(pos_sum)
    negative_sum.append(neg_sum)
    averagesenti.append(senti_avg)
    imbalance.append(imbal)
    posgap.append(positive_gap)
    neggap.append(negative_gap)
    i+=1



with open('senti.txt', 'w') as data_in:
    for i in range(0,len(corp)):
        data_in.write("%f\t%f\t%f\t%f\t%f\t%f\n" %(positive_sum[i],negative_sum[i],averagesenti[i],imbalance[i],posgap[i],neggap[i]))



datapath = 'SemEval2018-T4-train-taskA.txt'
char =[] # length of tweet
word = [] # no. of tokens of tweet
wordmean = []
noun =[]
verb = []
adverb = []
adjective = []
nounrat = []
verbrat = []
adverbrat =[]
adjectiverat = []



with open(datapath, 'rt') as data_in:
    for line in data_in:
        if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
            line = line.rstrip()
            char.append(len(line.split('\t')[2]))
            words = line.split('\t')[2].split()
            word.append(len(words))
            avg = [] # record length of each token in tweet
            for i in words:
                avg.append(len(i))
            wordmean.append(sum(avg)/len(avg)) # record average length of token in each tweet

tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True).tokenize

for line in corp: # for each tweet
    token = tokenizer(line)
    a = nltk.Text(token)
    tags=nltk.pos_tag(a)
    counts = Counter(tag for word,tag in tags)
    noun.append(counts['NN']+counts['NNS']+counts['NNP']+counts['NNPS'])
    verb.append(counts['VB']+counts['VBD']+counts['VBG']+counts['VBN']+counts['VBP']+counts['VBZ'])
    adverb.append(counts['RB']+counts['RBR']+counts['RBS'])
    adjective.append(counts['JJ']+counts['JJS']+counts['JJR'])


for a,b,c,d,w in zip(noun,verb,adverb,adjective,word):
    nounrat.append(a/w)
    verbrat.append(b/w)
    adverbrat.append(c/w)
    adjectiverat.append(d/w)



with open('structure.txt','w') as file:
    for e,a,b,c,d,e,f,g,h,i,j in zip(char,word,wordmean,noun,verb,adjective,adverb,nounrat,verbrat,adverbrat,adjectiverat):
        file.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" %(e,a,b,c,d,e,f,g,h,i,j))



data = 'ANC-token-word.txt'
tokenizer = RegexpTokenizer(r'\w+')
pattern = '(http.+(\s)?)|((@\w*\d*(\s)?))'
corp1 = []
lemma1 = WordNetLemmatizer()
global freqdata
freqdata = []

with open(data,encoding ='ISO-8859-1') as file:
        for l,i in zip(file,range(239208)):
            freqdata.append(l.split('\t')[0])
            freqdata.append(l.split('\t')[1])

freqdata = np.reshape(freqdata,(239208,2))

def frequency(target):
    if(target.isdigit()):
        return float('inf')
    else:
        start = 0
        end = len(freqdata) - 1
        while start <= end:
            middle = (start + end)// 2
            midpoint = freqdata[middle][0]
            if midpoint > target:
                end = middle - 1
            elif midpoint < target:
                start = middle + 1
            else:
                return int(freqdata[middle][1])
        return int(0)

for str in corp:
    str1 = re.sub(pattern, '', str, flags=re.MULTILINE)
    corp1.append(str1)


synolower = []
synolowermean = []
synolowergap = []
synohighergap = []
u=0
synomean = []
maxsyno = []
synsetgap = []

for tweet in corp1:
    token = tokenizer.tokenize(tweet)
    token = [word for word in token if word not in stopwords.words('english')]
    token = [lemma1.lemmatize(word) for word in token]
    synolow = []
    synohigh = []
    syno = []
    c= 0
    print(u)
    for word in token:
        freqofsyn = []
        lemmas = []
        sy =[]
        fr = []
        ly = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                lemmas.append(lemma.name().lower())
            lemmas = list(set(lemmas))
        #print('done synset')

        for l,j in zip(lemmas,range(len(lemmas))):
            #print(frequency(l))
            if(frequency(word.lower())>frequency(l)):
                sy.append(l)
                fr.append(frequency(l))
            else:
                ly.append(l)
        #print('done finding')
        syno.append(len(lemmas))
        synolow.append(len(sy))
        synohigh.append(len(ly)-1)
        c= c+1
    if(len(synolow)!=0):
        lowavg = sum(synolow)/len(synolow)
        v= max(synolow)
        synolower.append(sum(synolow))

    else:
        lowavg = 0
        v=0
        synolower.append(0)

    if(len(synohigh)!=0):
        highavg = sum(synohigh)/len(synohigh)
        w= max(synohigh)
    else:
        highavg = 0
        w= 0

    synolowermean.append(lowavg)
    synolowergap.append(v-lowavg)
    synohighergap.append(w-highavg)

    if(len(syno)!=0):
        synomean.append(sum(syno)/len(syno))
        maxsyno.append(max(syno))
        synsetgap.append(max(syno)-min(syno))
    else:
        synomean.append(0)
        maxsyno.append(0)
        synsetgap.append(0)

    u+=1

print(len(synolower),len(synolowermean),len(synolowergap),len(synohighergap),len(synomean),len(maxsyno),len(synsetgap))

with open('frequency.txt','w') as file:
    for i in range(len(synolowergap)):
        file.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\n" %(synolower[i],synolowermean[i],synolowergap[i],synohighergap[i],synomean[i],maxsyno[i],synsetgap[i]))


embeddings = []
length_tweets = []

tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True).tokenize

for line in (corp): # for each tweet
	length_tweets.append([len(line)])

	tokens = tokenizer(line)
	# print(tokens)
	vector_now = [0]*dim

	count_valid_token = 0
	for token in tokens:
		if token in embeddings_index:
			vector_now += embeddings_index[token]
			count_valid_token += 1

	# Mean of the vector
	if count_valid_token > 0:
		for i in range(len(vector_now)):
			vector_now[i] /= count_valid_token

	embeddings.append(vector_now)


def readtext(filename):
	error =[]
	with open(filename) as f:
		for line in f:
			line = line.rstrip()
			a = line.split('\t')
			error.append([float(b) for b in a])
	return error

def Train_Test(b):

	K_FOLDS = 10

	CLF1 = RandomForestClassifier(max_depth=50, random_state=0)
	pred1 = cross_val_predict(CLF1, b, label, cv=K_FOLDS)
	scor1 = metrics.f1_score(label, pred1, pos_label=1)
	acc1= metrics.accuracy_score(label,pred1)
	prec1 = metrics.precision_score(label,pred1)
	reca1 = metrics.recall_score(label,pred1)
	# a = metrics.confusion_matrix(label,pred1)

	f_output.write("======== RandomForest ========\n")
	f_output.write("F1-score: " + str(scor1) + '\n')
	f_output.write("Accuracy: " + str(acc1) + '\n')
	f_output.write("Precision: " + str(prec1) + '\n')
	f_output.write("Recall: " + str(reca1) + '\n')
	# f_output.write("confusion matrix\n",a)


	CLF2 = GaussianNB()
	pred2 = cross_val_predict(CLF2,b , label, cv=K_FOLDS)
	scor2 = metrics.f1_score(label,pred2 , pos_label=1)
	acc2= metrics.accuracy_score(label,pred2)
	prec2 = metrics.precision_score(label,pred2)
	reca2 = metrics.recall_score(label,pred2)
	# a = metrics.confusion_matrix(label,pred2)
	# print("F1-score,Accuracy,precision and recall New features included- GaussianNB\n",scor2,acc2,prec2,reca2)

	f_output.write("======== GaussianNB ========\n")
	f_output.write("F1-score: " + str(scor2) + '\n')
	f_output.write("Accuracy: " + str(acc2) + '\n')
	f_output.write("Precision: " + str(prec2) + '\n')
	f_output.write("Recall: " + str(reca2) + '\n')
	# f_output.write("confusion matrix\n",a)

	CLF3 = LinearSVC()
	pred3 = cross_val_predict(CLF3 ,b , label, cv=K_FOLDS)
	scor3 = metrics.f1_score(label,pred3 , pos_label=1)
	acc3= metrics.accuracy_score(label,pred3)
	prec3 = metrics.precision_score(label,pred3)
	reca3 = metrics.recall_score(label,pred3)
	# a = metrics.confusion_matrix(label,pred3)
	# print("F1-score,Accuracy,precision and recall New features included- Linear SVM\n",scor3,acc3,prec3,reca3)

	f_output.write("======== LinearSVC ========\n")
	f_output.write("F1-score: " + str(scor3) + '\n')
	f_output.write("Accuracy: " + str(acc3) + '\n')
	f_output.write("Precision: " + str(prec3) + '\n')
	f_output.write("Recall: " + str(reca3) + '\n')
	# f_output.write("confusion matrix\n",a)

	CLF4 = DecisionTreeClassifier()
	pred4 = cross_val_predict(CLF4 ,b , label, cv=K_FOLDS)
	scor4 = metrics.f1_score(label,pred4 , pos_label=1)
	acc4= metrics.accuracy_score(label,pred4)
	prec4 = metrics.precision_score(label,pred4)
	reca4 = metrics.recall_score(label,pred4)
	# a = metrics.confusion_matrix(label,pred4)

	# print("F1-score,Accuracy,precision and recall New features included- Decision Tree Classifier\n",scor4,acc4,prec4,reca4)
	f_output.write("======== Decision Tree ========\n")
	f_output.write("F1-score: " + str(scor4) + '\n')
	f_output.write("Accuracy: " + str(acc4) + '\n')
	f_output.write("Precision: " + str(prec4) + '\n')
	f_output.write("Recall: " + str(reca4) + '\n')
	# f_output.write("confusion matrix\n",a)

pun = 'punctuation.txt'
fre = 'frequency.txt'
stru = 'structure.txt'
sent = 'senti.txt'
free = 'freq.txt'

punc = readtext(pun)
struc = readtext(stru)
syn = readtext(fre)
sentiment = readtext(sent)
frequen = readtext(free)
# b = np.hstack((X, embeddings, punc, struc, syn, sentiment, frequen))
# b = np.hstack((X, embeddings))
# b = embeddings
# print(np.shape(b))


f_output = open("results_test.txt", "w")

f_output.write("BoW | GloVe | Length\n")
b = np.hstack((X, embeddings, length_tweets))
Train_Test(b)
f_output.write("\n")

# f_output.write("BoW\n")
# b = X
# Train_Test(b)
# f_output.write("\n")

# f_output.write("BoW | GloVe\n")
# b = np.hstack((X, embeddings))
# Train_Test(b)
# f_output.write("\n")

# f_output.write("BoW | GloVe | Sentiment\n")
# b = np.hstack((X, embeddings, sentiment))
# Train_Test(b)
# f_output.write("\n")

# f_output.write("BoW | GloVe | Sentiment | Punctuations\n")
# b = np.hstack((X, embeddings, sentiment, punc))
# Train_Test(b)
# f_output.write("\n")

# f_output.write("BoW | GloVe | Sentiment | Punctuations | POS\n")
# b = np.hstack((X, embeddings, sentiment, punc, struc))
# Train_Test(b)
# f_output.write("\n")

# f_output.write("BoW | GloVe | Sentiment | Punctuations | POS | Synonym\n")
# b = np.hstack((X, embeddings, sentiment, punc, struc, syn))
# Train_Test(b)
# f_output.write("\n")

# f_output.write("BoW | GloVe | Sentiment | Punctuations | POS | Synonym | Frequent Words\n")
# b = np.hstack((X, embeddings, sentiment, punc, struc, syn, frequen))
# Train_Test(b)
# f_output.write("\n")