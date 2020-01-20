'''
Loading Gensim and nltk libraries
'''
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from nltk import word_tokenize, pos_tag
import numpy as np
import re
import csv;
import numpy as np;
import pandas as pa;

def load(path):
    
    #76-86 87-96 97-01 02-06 07-12 13-15 16-19
    data7686 = [];
    data8796 = [];
    data9701 = [];
    data0206 = [];
    data0712 = [];
    data1315 = [];
    data1619 = [];
    
    a = pa.read_csv(path, error_bad_lines=False,encoding = "ISO-8859-1");
    data = [];
    i = 0;
    for i in range(len(a.text)):
        if(isinstance(a.date[i],str)):
            year = a.date[i].split('-')[0];
            doc = a.text[i];
            if (isinstance(doc, str)):
                if "[Via Interpreter]" in doc:
                    doc = doc.replace("[Via Interpreter]","");
                    
                if "[Applause]" in doc:
                    doc = doc.replace("[Applause]","");
                    
                if "[Booing]" in doc:
                    doc = doc.replace("[Booing]","");
                
                if "donald" in doc:
                    doc = doc.replace("donald","");
                  
                if "trump" in doc:
                    doc = doc.replace("trump","");  
       
                if "today" in doc:
                    doc = doc.replace("today","");
                
                if "obamacar" in doc:
                    doc = doc.replace("obamacar","");
                    
                if "yesterday" in doc:
                    doc = doc.replace("yesterday","");
                
                if "tomorrow" in doc:
                    doc = doc.replace("tomorrow",""); 
                
                if "want" in doc:
                    doc = doc.replace("want",""); 
                    
                if "[Off Microphone Question]" in doc:
                    doc = doc.replace("[Off Microphone Question]","");
                
                doc = re.sub('@\S*\s?','',doc);
                doc = re.sub('http\S*\s?','',doc);
                y = int(year);
                if(y>=1976 and y <=1986):
                    data7686.append(doc);
                if(y>=1987 and y <=1996):
                    data8796.append(doc);
                if(y>=1997 and y <=2001):
                    data9701.append(doc);
                if(y>=2002 and y <=2006):
                    data0206.append(doc);
                if(y>=2007 and y <=2012):
                    data0712.append(doc);
                if(y>=2013 and y <=2015):
                    data1315.append(doc);
                if(y>=2016 and y <=2019):
                    data1619.append(doc);
                
        i = i + 1;
        
    Data = [];
    Data.append(data7686);
    Data.append(data8796);
    Data.append(data9701);
    Data.append(data0206);
    Data.append(data0712);
    Data.append(data1315);
    Data.append(data1619);
    RE = [];
    for da in Data:
        newData = [];
        for Ndoc in da:
            if (Ndoc != ""):
                newData.append(Ndoc);
        RE.append(newData);
    
    return RE;

'''
Write a function to perform the pre processing steps on the entire dataset
'''

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# Tokenize and lemmatize
def preprocess(text):
    result=[]
    for token in word_tokenize(text) :
        tag = pos_tag([token])
        
        if tag[0][1] in {'NN','NNS','NNP','NNPS'}:
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(lemmatize_stemming(token))
            
    return result


#######################################################################################
#start
#######################################################################################

#data1 = load('fulltext.csv');
dataSet = load('datefulltext.csv');
#data = np.concatenate((data1,data2), axis = 0);
year = ['76-86', '87-96', '97-01', '02-06', '07-12', '13-15', '16-19'];
j = 0;
for data in dataSet:
    DNyear = year[j];
    j = j+1;
    stemmer = SnowballStemmer("english")
    
    
    print("preprocess start") 
    
    processed_docs = []
    for doc in data:
        processed_docs.append(preprocess(doc))
    print("preprocess done")    
    '''
    Create a dictionary from 'processed_docs' containing the number of times a word appears 
    in the training set using gensim.corpora.Dictionary and call it 'dictionary'
    '''
    dictionary = gensim.corpora.Dictionary(processed_docs)
    
    '''
    OPTIONAL STEP
    Remove very rare and very common words:
    
    - words appearing less than 15 times
    - words appearing in more than 10% of all documents
    '''
    dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n= 200000)
    
    
    '''
    Create the Bag-of-words model for each document i.e for each document we create a dictionary reporting how many
    words and how many times those words appear. Save this to 'bow_corpus'
    '''
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    
    '''
    Preview BOW for our sample preprocessed document
    '''
    document_num = 20
    bow_doc_x = bow_corpus[document_num]
    
    for i in range(len(bow_doc_x)):
        print("Word {} (\"{}\") appears {} time.".format(bow_doc_x[i][0], 
                                                         dictionary[bow_doc_x[i][0]], 
                                                         bow_doc_x[i][1]))
        
    # LDA mono-core -- fallback code in case LdaMulticore throws an error on your machine
    # lda_model = gensim.models.LdaModel(bow_corpus, 
    #                                    num_topics = 10, 
    #                                    id2word = dictionary,                                    
    #                                    passes = 50)
    
    # LDA multicore 
    '''
    Train your lda model using gensim.models.LdaMulticore and save it to 'lda_model'
    '''
    # TODO
    lda_model =  gensim.models.ldamodel.LdaModel(bow_corpus, 
                                       num_topics = 10, 
                                       id2word = dictionary,                                    
                                       passes = 50,
                                       )
    
    '''
    For each topic, we will explore the words occuring in that topic and its relative weight
    '''
    of = open(DNyear+".txt","w+");
    for idx, topic in lda_model.print_topics(-1):
        of.write("Topic: {} \nWords: {}".format(idx, topic ));
        of.write("\n");

    print(DNyear," Done");
    of.close();
    

