
"""
Yahia El Bsat
IEMS 308 - HW3
"""

import util

pre_dir = "/Users/Yahia/Desktop/IEMS308/HW3/articles_pre/"
corpus_dir = "/Users/Yahia/Desktop/IEMS308/HW3/articles/"


clean_corpus()

# Create an NLTK corpus reader to access text data.                
corpus = nltk.corpus.PlaintextCorpusReader(corpus_dir, '.*\.txt')

#get stats about corpus
words = nltk.FreqDist(corpus.words())
count = sum(words.values())
vocab = len(words)
sents = len(corpus.sents()) #696194 sentences


#### Companies & CEOs

#Importing training companies
t_comp =  pd.read_csv("/Users/Yahia/Desktop/IEMS308/HW3/training/companies.csv",header = None)
t_comp = t_comp.sort_values(by= 0)
t_comp = t_comp[0].unique().tolist()

t_ceo = pd.read_csv("/Users/Yahia/Desktop/IEMS308/HW3/training/ceo.csv",header = None)
t_ceo = t_ceo.fillna('')
t_ceo[2] = t_ceo[0] + ' ' + t_ceo[1]
t_ceo[2] = t_ceo[2].apply(lambda x: x.strip())
t_ceo.columns = ['a','b','name']
t_ceo = t_ceo.drop(['a','b'],axis = 1) 
t_ceo[1] = ''

#Find all sentences in corpus that include the company names in the training dataset
#tokeninize company names
temp = pd.Series(t_comp)
t_comp = pd.DataFrame([temp]).transpose()
t_comp[1] = ''
  

tokenizer = RegexpTokenizer("\w+") 
for i in range(len(t_comp)):
    t_comp.iloc[i,1] = tokenizer.tokenize(t_comp.iloc[i,0])     

for i in range(len(t_ceo)):
    t_ceo.iloc[i,1] = tokenizer.tokenize(t_ceo.iloc[i,0])   

comp_set = t_comp[1]
#comp_set = [set(x) for x in comp_set]

ceo_set = t_ceo[1]
#ceo_set = [set(x) for x in ceo_set]

# finding positive sample company sentences
comp_sent = []
ceo_sent = []
reg_sent = []

exp = "([A-Z][\w-]*(\s+[A-Z][\w-]*)+)"
pattern = re.compile(exp)


comp_keywords = {"Co", "Corp", "Corporation", "Company", "Group", "Inc",
            "Ltd", "Capital", "Financial", "Management"}

ceo_keywords = {"CEO", "chief",  "executive", "officer"}


sample = pd.DataFrame(columns = ['sent','word','type','sent_num','start','len','pos_before','pos_after','keyword','is_english',
                                      'is_location','num_capital','beg_sent','end_sent'])

dico = enchant.Dict("en_US")
    
def process_sentence(sent,word_l,typ,num):
    word = ' '.join(word_l)
    start = first_occur(word_l,sent)
    leng = len(word_l)
    pos = nltk.pos_tag(sent)
    pos_before = pos[start-1][1] if start != 0 else ''
    pos_after = pos[start+leng][1] if start + leng < len(sent) else ''    
    if typ == 'comp':
        keyword = 1 if len(set(sent).intersection(comp_keywords)) > 0 else 0
    elif typ == 'ceo':
        keyword = 1 if len(set(sent).intersection(ceo_keywords)) > 0 else 0
    else:
        keyword = 1 if len(set(sent).intersection(ceo_keywords.union(comp_keywords))) > 0 else 0
    is_english = all(dico.check(n) for n in word_l)
    is_english = 1 if is_english else 0
    places = GeoText(word)
    is_location = (len(places.cities) + len(places.countries)) != 0
    is_location = 1 if is_location else 0
    num_capital = sum(1 for c in word if c.isupper())
    beg_sent = 1 if start == 0 else 0
    end_sent = 1 if start + leng == len(sent) else 0
    r =  {'sent': sent, 'word':word, 'type':typ, 'sent_num':num, 'start':start, 
            'len':leng, 'pos_before':pos_before, 'pos_after':pos_after, 'keyword':keyword,
            'is_english': is_english, 'is_location': is_location, 'num_capital': num_capital,
            'beg_sent': beg_sent, 'end_sent': end_sent}
    return r
        

i = 0
for sent in corpus.sents()[:200000]:
    i = i+1
    if i % 2000 == 0: print(f"Processed {i} sent")
    if random.uniform(0,1) < 0.6:
        for name in comp_set:
            if set(name).issubset(sent):
                r = process_sentence(sent,name,'comp',i)
                sample = sample.append(r,ignore_index = True)
                break
        for name in ceo_set:
            if set(name).issubset(sent):
                r = process_sentence(sent,name,'ceo',i)
                sample = sample.append(r,ignore_index = True)
                break
    else:
        pats = re.findall(pattern, ' '.join(sent))
        for p in pats:
            tok = tokenizer.tokenize(p[0]) 
            r = process_sentence(sent,tok,'reg',i)
            sample = sample.append(r,ignore_index = True)


sample['pos_before'] = sample['pos_before'].astype('category')
sample['pos_after'] = sample['pos_after'].astype('category')


sample['is_ceo_keyword'] = 0
sample['is_comp_keyword'] = 0    

sample['is_ceo_keyword'] = sample.sent.apply(lambda sent: 1 if len(set(sent).intersection(ceo_keywords)) > 0 else 0)
sample['is_comp_keyword'] = sample.sent.apply(lambda sent: 1 if len(set(sent).intersection(comp_keywords)) > 0 else 0)       
sample['char_len'] = sample.word.map(len)

sample.to_csv('/Users/Yahia/Desktop/IEMS308/HW3/proper_sample.csv') 

#training and testing model
names_sample = sample[sample.type != 'reg']
names_sample['is_ceo'] = 0
names_sample['is_ceo'] = names_sample.type == 'ceo'
names_sample['is_comp'] = 0
names_sample['is_comp'] = names_sample.type == 'comp'

names_sample.is_ceo.value_counts()            
#==============================================================================
# False    17907
# True      5178
#==============================================================================


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier

x = names_sample.loc[:,['start','len','pos_before','pos_after','is_english',
                 'is_location','num_capital','beg_sent','end_sent','is_ceo_keyword','is_comp_keyword','char_len']]
x = pd.get_dummies(x)
y = names_sample['is_comp']
y_ceo = names_sample['is_ceo']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
x_train_ceo, x_test_ceo, y_train_ceo, y_test_ceo = train_test_split(x, y_ceo, test_size=0.25, random_state=0)

#logistic regression
logisticRegr = LogisticRegression()
lr_fit = logisticRegr.fit(x_train, y_train)
lr_pred_test = lr_fit.predict(x_test)
lr_score = lr_fit.score(x_test, y_test)
#accur is 0.80
lr_metrics = precision_recall_fscore_support(y_test,lr_pred_test,average='macro')
#precision: 0.71, #recall: 0.61, #fbeta: 0.63

#gaussian naive bayes
gnb = GaussianNB()
gnb_fit = gnb.fit(x_train, y_train)
gnb_pred_test = gnb_fit.predict(x_test)
gnb_score = gnb_fit.score(x_test, y_test)
#score 0.28
gnb_metrics = precision_recall_fscore_support(y_test,gnb_pred_test,average='macro')
#precision: 0.56, #recall: 0.52, #fbeta: 0.26

#random forest
rfc = RandomForestClassifier(n_estimators = 30)
rf_fit = rfc.fit(x_train, y_train)
rf_pred_test = rf_fit.predict(x_test)
rf_score = rf_fit.score(x_test, y_test)
#score: 0.89
rf_metrics = precision_recall_fscore_support(y_test,rf_pred_test,average='macro')
#precision: 0.86, #recall: 0.84, #fbeta: 0.85

#random forest for ceo
rfc_ceo = RandomForestClassifier(n_estimators = 30)
rf_ceo_fit = rfc_ceo.fit(x_train_ceo, y_train_ceo)
rf_pred_test_ceo = rf_ceo_fit.predict(x_test_ceo)
rf_score_ceo = rf_ceo_fit.score(x_test_ceo, y_test_ceo)
#score: 0.91
rf_metrics_ceo = precision_recall_fscore_support(y_test_ceo,rf_pred_test_ceo,average='macro')
#precision: 0.86, #recall: 0.84, #fbeta: 0.85



## Extracting all companies and ceos from remaining corpus
remain = pd.DataFrame(columns = ['sent','word','type','sent_num','start','len','pos_before','pos_after','keyword','is_english',
                                      'is_location','num_capital','beg_sent','end_sent'])


i = 200000
for sent in corpus.sents()[200000:]: #runned till 516000
    i = i + 1
    if i % 2000 == 0: print(f"Processed {i} sent")
    pats = re.findall(pattern, ' '.join(sent))
    for p in pats:
        tok = tokenizer.tokenize(p[0]) 
        r = process_sentence(sent,tok,'reg',i)
        remain = remain.append(r,ignore_index = True)


remain['is_ceo_keyword'] = 0
remain['is_comp_keyword'] = 0    

remain['is_ceo_keyword'] = remain.sent.apply(lambda sent: 1 if len(set(sent).intersection(ceo_keywords)) > 0 else 0)
remain['is_comp_keyword'] = remain.sent.apply(lambda sent: 1 if len(set(sent).intersection(comp_keywords)) > 0 else 0) 
remain['char_len'] = remain.word.map(len)


remain.to_csv('/Users/Yahia/Desktop/IEMS308/HW3/proper_remain.csv') 


# also append the reg from the <200000 range
all_to_mine = remain.append(sample, ignore_index=True)

#select the right columns from this, predict the comp rf, fit another model for ceo, predict again, output results
to_process = all_to_mine.loc[:,['start','len','pos_before','pos_after','is_english',
                 'is_location','num_capital','beg_sent','end_sent','is_ceo_keyword','is_comp_keyword','char_len']]

to_process = pd.get_dummies(to_process)
diff = list(set(to_process.columns) - set(x.columns))
to_process = to_process.drop(diff,axis = 1)

comp_pred = rf_fit.predict(to_process)
ceo_pred = rf_ceo_fit.predict(to_process)

all_to_mine['is_comp'] = comp_pred
all_to_mine['is_ceo'] = ceo_pred

all_to_mine.to_csv('/Users/Yahia/Desktop/IEMS308/HW3/result.csv')

out_ceo = all_to_mine.loc[(all_to_mine.is_ceo == True),'word']
out_ceo = out_ceo.unique().tolist()
out_ceo = pd.DataFrame({'ceo':out_ceo})

out_ceo.to_csv('/Users/Yahia/Desktop/IEMS308/HW3/out_ceo.csv')

out_comp = all_to_mine.loc[(all_to_mine.is_comp == True),'word']
out_comp = out_comp.unique().tolist()   
out_comp = pd.DataFrame({'comp':out_comp})

out_comp.to_csv('/Users/Yahia/Desktop/IEMS308/HW3/out_comp.csv')

### Percentages
t_perc = pd.read_csv("/Users/Yahia/Desktop/IEMS308/HW3/training/percentage.csv",header = None)           


found_percents = [ ]
pattern1 = re.compile("\d+(?:\.\d+)?(?:%| percent(?:age points)?)")

digits = "(?:one|two|three|four|five|six|seven|eight|nine)"
teens = "(?:eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen)"
tens = "(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"

exp = f"(?:{digits}|{teens}|{tens}|(?:{tens}-{digits})) percent(?:age points)?"
pattern2 = re.compile(exp)

data = []
for file in os.listdir('/Users/Yahia/Desktop/IEMS308/HW3/articles/'):
    data.append(open(os.path.join('/Users/Yahia/Desktop/IEMS308/HW3/articles/',file),'rb').read().decode('utf-8'))


for d in data:
    matches = re.findall(pattern1, d)
    found_percents.extend(matches)
    matches = re.findall(pattern2, d)
    found_percents.extend(matches)
      
found_percents = list(set(found_percents))   
found_percents = pd.DataFrame({'perc':found_percents})

found_percents.to_csv('/Users/Yahia/Desktop/IEMS308/HW3/out_percents.csv')








