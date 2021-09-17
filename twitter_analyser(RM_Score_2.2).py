# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 11:24:22 2018

@author: AmirAbdulReda
"""

#Twitter data analyser

#Notes to self
    #The key difference in this file is that it both contains the "sign" activity
    #keyword AND registers user IDs at the same time. 

#=============================================================================#
#--------------------TWITTER RMT ANALYZER (mobilization only)-----------------#
#=============================================================================#

#For detecting and collecting RM tweets
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime
import re

#For topic modelling of the tweets
import numpy as np 
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
import math
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

os.chdir("G:\SOMOQPRO")

#---------------------------------------------------------------------------------#
#-------------------------I. Filtering tweets using the Lexicon ------------------#
#---------------------------------------------------------------------------------#

#Lexicon for Social Movement related upheavals
            #To be recorded in a separate file and imported in this script
            #when lexicon finalized
            
keyword_activities = ["sit-in", " protest", " rally ",  " picket ", " strike ", " boycott", "march", " fight",
                      " mobilization", " petition", " to demand ", " we demand ", "walkout", " walk out "]

keyword_organize =[" join ", " joined ", " stand with ", "come out", " come down ", " please come ",
                   " support ", " help the ",
                   "help us "," i am going to ", " i'm going", "im going" ," on my way ", " be there ", " sign"
                   #potentially add " sign"  too
                   ]

keyword_target = [" in ", " at ", " to ", " for ", "against", "outside", " with ", " PM ",
                  " AM "," here ", " tonight ", " today ", " tomorrow "]

keyword_negative = [#"in march", "march 1","march 2","march 3","march 4","march 5","march 6", "march 7","march 8","march 9","box", 
                    "game", "play","ufc", "wwe", "twitch", " mma ", "wrestl", " fan ",
                    " fans ", "marchmadness", "march madness","christmas parade", "viceroy",
                    "marchandiseur", "fitness", "nfl", "nfc", "nba"] #Notice three added keywords here
   
#Computing the RM score for the US based on our lexicon and storing its value by day in a pandas dataframe
total_tweets_US = 0
about_upheavals_US = 0
df_US = []   
protest_tweets_US=[]
tweet_type = []

with open('SOMOQPRO.txt') as f:
    for line in f:
        try:
            tweet = json.loads(line)
        except json.decoder.JSONDecodeError:
            pass
        if (('text' in tweet) and ('place' in tweet)):
            if tweet['place'] is not None and tweet['place']['country_code'] is not None:
                if tweet['place']['country_code']=='US':
                    total_tweets_US +=1
                    if (('text' in tweet) and ('place' in tweet)):
                        if tweet['place'] is not None and tweet['place']['country_code'] is not None:
                            if tweet['place']['country_code']=='US':
                                if 'retweeted_status' in tweet and tweet["retweeted_status"]["extended_tweet"]["full_text"] is not None:                                    
                                    if any(s in tweet["retweeted_status"]["extended_tweet"]["full_text"].lower() for s in keyword_activities):
                                        if any(t in tweet["retweeted_status"]["extended_tweet"]["full_text"].lower() for t in keyword_organize):
                                           if any(e in tweet["retweeted_status"]["extended_tweet"]["full_text"].lower() for e in keyword_target):
                                               if not any(d in tweet["retweeted_status"]["extended_tweet"]["full_text"].lower() for d in keyword_negative):
                                                   about_upheavals_US += 1
                                                   bin_value_US = 1
                                                   tweet_type = "retweeted_status_extended"
                                                   rm_score_US = about_upheavals_US/total_tweets_US
                                                   protest_tweets_US.append({'Tweet': tweet["retweeted_status"]["extended_tweet"]["full_text"], "Time": tweet['created_at'], "Tweet_Type": str(tweet_type), "place":tweet['place']['full_name'], "User_ID": tweet['user']['id_str']})                                                  
                                                   print("Of " + str(total_tweets_US) +" tweets," + str(about_upheavals_US) +" tweets were about protests in the United States, and the RM score in that country is " + str(rm_score_US) +" for ", tweet['created_at'] +" and that was an extended retweet")
                                                   df_US.append({'Total Number of Tweets': total_tweets_US, 'Tweets about Protests': about_upheavals_US, 'RM Score':
                                                            rm_score_US, 'Time': tweet['created_at'], 'bin_value': bin_value_US})
                                elif 'retweeted_status' in tweet and tweet['retweeted_status']['text'] is not None:                                    
                                    if any(s in tweet['retweeted_status']['text'].lower() for s in keyword_activities):
                                        if any(t in tweet['retweeted_status']['text'].lower() for t in keyword_organize):
                                           if any(e in tweet['retweeted_status']['text'].lower() for e in keyword_target):
                                               if not any(d in tweet['retweeted_status']['text'].lower() for d in keyword_negative):
                                                   about_upheavals_US += 1
                                                   bin_value_US = 1
                                                   tweet_type = "retweeted_status_regular"
                                                   rm_score_US = about_upheavals_US/total_tweets_US
                                                   protest_tweets_US.append({'Tweet': tweet['retweeted_status']['text'], "Time": tweet['created_at'], "Tweet_Type": str(tweet_type), "place":tweet['place']['full_name'] , "User_ID": tweet['user']['id_str']})                                              
                                                   print("Of " + str(total_tweets_US) +" tweets," + str(about_upheavals_US) +" tweets were about protests in the United States, and the RM score in that country is " + str(rm_score_US) +" for ", tweet['created_at'] +" and that was a regular retweet")
                                                   df_US.append({'Total Number of Tweets': total_tweets_US, 'Tweets about Protests': about_upheavals_US, 'RM Score':
                                                            rm_score_US, 'Time': tweet['created_at'], 'bin_value': bin_value_US})
                                elif 'quoted_status' in tweet and tweet['quoted_status']['text'] is not None:
                                    if any(s in tweet['quoted_status']['text'].lower() for s in keyword_activities):
                                        if any(t in tweet['quoted_status']['text'].lower() for t in keyword_organize):
                                           if any(e in tweet['quoted_status']['text'].lower() for e in keyword_target):
                                               if not any(d in tweet['quoted_status']['text'].lower() for d in keyword_negative):
                                                   about_upheavals_US += 1
                                                   bin_value_US = 1
                                                   tweet_type = "quoted_status"
                                                   rm_score_US = about_upheavals_US/total_tweets_US
                                                   protest_tweets_US.append({'Tweet': tweet['quoted_status']['text'], "Time": tweet['created_at'], "Tweet_Type": str(tweet_type), "place":tweet['place']['full_name'] , "User_ID": tweet['user']['id_str']})
                                                   print("Of " + str(total_tweets_US) +" tweets," + str(about_upheavals_US) +" tweets were about protests in the United States, and the RM score in that country is " + str(rm_score_US) +" for ", tweet['created_at'] +" and that was a regular quoted tweet")
                                                   df_US.append({'Total Number of Tweets': total_tweets_US, 'Tweets about Protests': about_upheavals_US, 'RM Score':
                                                       rm_score_US, 'Time': tweet['created_at'], 'bin_value': bin_value_US})
                                elif 'extended_tweet' in tweet and tweet['extended_tweet']['full_text'] is not None:
                                    if any(s in tweet['extended_tweet']['full_text'].lower() for s in keyword_activities):
                                        if any(t in tweet['extended_tweet']['full_text'].lower() for t in keyword_organize):
                                           if any(e in tweet['extended_tweet']['full_text'].lower() for e in keyword_target):
                                             if not any(d in tweet['extended_tweet']['full_text'].lower() for d in keyword_negative):
                                               about_upheavals_US += 1
                                               bin_value_US = 1
                                               tweet_type = "extended_tweet"
                                               rm_score_US = about_upheavals_US/total_tweets_US
                                               protest_tweets_US.append({'Tweet': tweet['extended_tweet']['full_text'], "Time": tweet['created_at'], "Tweet_Type": str(tweet_type), "place":tweet['place']['full_name'] , "User_ID": tweet['user']['id_str']})
                                               print("Of " + str(total_tweets_US) +" tweets," + str(about_upheavals_US) +" tweets were about protests in the United States, the RM score in that country is " + str(rm_score_US) +" for ", tweet['created_at'] +" and that was an extended tweet")
                                               df_US.append({'Total Number of Tweets': total_tweets_US, 'Tweets about Protests': about_upheavals_US, 'RM Score':
                                                        rm_score_US, 'Time': tweet['created_at'], 'bin_value': bin_value_US})
                                else: 
                                    if 'text' in tweet and tweet['text'] is not None:
                                        if any(s in tweet['text'].lower() for s in keyword_activities):
                                            if any(t in tweet['text'].lower() for t in keyword_organize):
                                               if any(e in tweet['text'].lower() for e in keyword_target):
                                                   if not any(d in tweet['text'].lower() for d in keyword_negative):
                                                       about_upheavals_US += 1
                                                       bin_value_US = 1
                                                       tweet_type = "regular_tweet"
                                                       rm_score_US = about_upheavals_US/total_tweets_US
                                                       protest_tweets_US.append({'Tweet': tweet['text'], "Time": tweet['created_at'], "Tweet_Type": str(tweet_type), "place":tweet['place']['full_name'] , "User_ID": tweet['user']['id_str']})
                                                       print("Of " + str(total_tweets_US) +" tweets," + str(about_upheavals_US) +" tweets were about protests in the United States, and the RM score in that country is " + str(rm_score_US) +" for ", tweet['created_at']+" and that was a regular tweet")
                                                       df_US.append({'Total Number of Tweets': total_tweets_US, 'Tweets about Protests': about_upheavals_US, 'RM Score':
                                                                rm_score_US, 'Time': tweet['created_at'], 'bin_value': bin_value_US})
    

#Today
n = datetime.today().date().strftime("%B %d, %Y")

df_US=pd.DataFrame(df_US)
protest_tweets_US = pd.DataFrame(protest_tweets_US)
protest_tweets_US = protest_tweets_US.drop_duplicates(subset=['Tweet'], keep='first')

remove_ms = lambda x:re.sub("\+\d+\s","",x)  #Add daily dates
mk_dt = lambda x:datetime.strptime(remove_ms(x), "%a %b %d %H:%M:%S %Y")
my_form = lambda x:"{:%Y-%m-%d}".format(mk_dt(x))
df_US['dates'] = df_US.Time.apply(my_form)                             
df_US['dates']

protest_tweets_US['dates'] = protest_tweets_US.Time.apply(my_form)                             
protest_tweets_US['dates']

#Switching working directories because of storage limitation in G: drive

os.chdir("C:\\Users\\AmirAbdulReda\OneDrive - University of Toronto\\PhD\\Papers for publication\\Comparative Politics\\Social Movement Project\\1st Paper\\Draft 3 R&R\\script, data, figures\\With user IDs and sign keyword")


#df_US.to_csv('rm_score_US by ('+n+').csv', index=False, header=False)
#protest_tweets_US.to_csv('protest_tweets_US(rm) by ('+n+').csv', index=False, header=True)

#df_US = pd.read_csv("rm_score_US by (May 16, 2020).csv")

#Plot
#Live evolution of RM score
fig, ax = plt.subplots(figsize=(15,7))
plot_US=df_US.plot(x='Time', y='RM Score', ax=ax)
plot_US
plt.xticks( rotation=80)
plt.ylabel('RM Score')
plt.title('Live Evolution of the Resource Mobilization (RM) Score in the USA',fontdict=None, loc='center')
plt.savefig('Live RM Score in USA by ('+n+').png', dpi=300, bbox_inches = 'tight')
#plt.show()

#Daily number of RM tweets
fig, ax = plt.subplots(figsize=(15,7))
df_US2 = df_US.groupby(['dates']).size()
df_US2.plot(kind='bar', x='Index', y='0', stacked=True, ax=ax)
plot_US
plt.xticks( rotation=80)
plt.ylabel('Number of RM Tweets')
plt.title('Daily Evolution of Resource Mobilization Tweets in the USA',fontdict=None, loc='center')
plt.savefig('Daily # of RM Tweets in USA by ('+n+').png', dpi=300, bbox_inches = 'tight')

#Total Number of Tweets remains constant
fig, ax = plt.subplots(figsize=(15,7))
plot_US=df_US.plot(x='dates', y='Total Number of Tweets', ax=ax)
plot_US
plt.xticks( rotation=80)
plt.ylabel('Total Number of Tweets')
plt.title('Daily Evolution of the Total Number of Tweets in the USA',fontdict=None, loc='center')
plt.savefig('Total Number of Tweets in USA by ('+n+').png', dpi=300, bbox_inches = 'tight')
       

#=========================================================================================================================#

#--------------------------------------#II. Topic Modeling#------------------------------------------------------------------#

#=========================================================================================================================#
#-------------------------------------------------------------------------------------------------------------------------#

#======================================#II.1. Non-Negative Matrix for United States#=================================#

#-------------------------------------------------------------------------------------------------------------------------#

#protest_tweets_US = pd.read_csv("Protest Tweets US.csv", encoding='latin1')

corpus = protest_tweets_US
len(corpus)
corpus[0:2]

corpus.Tweet = corpus.Tweet.apply(lambda x: re.split('https:\/\/.*', str(x))[0]) #removing http stuff/links

#Try without stopwords or with a different one for tweets
stopwords = list(text.ENGLISH_STOP_WORDS)#Removing only regular english stop words

#Note that sign is now added as a stopword too
additional_stopwords = ["sit-in", "protest", "rally",  "picket", "strike", "boycott", "march", "fight"," mobilization", " walkout ", "walk out ",  "petition", "demand", "sign", "we demand",
                        "join", "joined", "stand", "come", "come", "please", "support", "help","going", " be there ", "ll", "don't", "dont", "im", "did", "don", "amp"]

stopwords.extend(additional_stopwords)
stopwords

tfidf = TfidfVectorizer(stop_words=stopwords) #We start again by creating a TF IDF and removing stop words
X = tfidf.fit_transform(corpus.Tweet)#For tweets only (not dates and time)

nmf = NMF(n_components=100) # Non negative matrix with 11 components
nmf.fit(X) # Fitting the TFIDF of the tweets to the matrix 
W = nmf.transform(X) #Transforming to the necessary W matrix 
print(W.shape)

H = nmf.components_ # now the H matrix
print(H.shape)

#Assigning topics to each tweet by looking at the largest column in W
topic_assignment = np.argmax(W, axis=1) 

#If dates is breaking because it doesn't load as strings from csv
n = datetime.today().date().strftime("%B %d, %Y")
remove_ms = lambda x:re.sub("\+\d+\s","",x)  #Add daily dates
mk_dt = lambda x:datetime.strptime(remove_ms(x), "%a %b %d %H:%M:%S %Y")
my_form = lambda x:"{:%Y-%m-%d}".format(mk_dt(x))
protest_tweets_US['dates'] = protest_tweets_US.Time.apply(my_form)                             
protest_tweets_US['dates']

newdf = pd.DataFrame({'Tweet': corpus.Tweet, 'tfidf_topic_id': topic_assignment, 'date': corpus.dates })
newdf.head()

#Retrieving top words for each topic by taking the highest values in the H matrix

def print_top_words(model, feature_names, top_n):
    
    """
    A function that prints the top words for each topic of an NMF model.
    
    model: the name of the object containing the fitted model.
    feature_names: the get_feature_names word list of the vectorizer object.
    top_n: how many top words to print.
    
    """
    
    H = model.components_

    # For each row of H, i.e. for each topic:
    for topic_id, topic in enumerate(H):

        # Print topic number and top words compactly
        message = "Topic #%d: " % topic_id
        message += " ".join([feature_names[i] for i in topic.argsort()[::-1][:top_n]])
        print(message)

print_top_words(nmf, tfidf.get_feature_names(), 10)

newdf = pd.DataFrame({'Tweet': corpus.Tweet, 'nmf_w2v_topic_id': topic_assignment, 'date': corpus.dates, 'Time': corpus.Time, 'Location':corpus.Location, 'User_ID':corpus.User_ID
                      #, 'tfidf_topic_id': newdf.tfidf_topic_id 
                      })
    
#For Turkey pitch newdf['date'] = newdf['date'].astype(str)

newdf.head()


#Saving to CSV
newdf.to_csv('Protest Tweets by Top 100 Topics.csv', index=False, header=True)

#Saving Topwords to Dataframe
topwords_US0 = []
def save_topwords(model, feature_names, top_n):
    B = model.components_
    for topic_id, topic in enumerate(B):
        topwords_US0.append({"nmf_w2v_Topic #": "Topic #%d: " % topic_id, "Topwords": " ".join([feature_names[i] for i in topic.argsort()[::-1][:top_n]])})
vectorizer = CountVectorizer(stop_words=stopwords)
feature_names = vectorizer.get_feature_names()
save_topwords(nmf, tfidf.get_feature_names(), 20)                                                       
topwords_US0

labels = ["nmf_w2v_Topic #", "Topwords"]

topwords_US0 = pd.DataFrame.from_records(topwords_US0, columns=labels)
topwords_US0.to_csv('Top100 Topics_US_Top10 Words.csv', index=False, header=True)


#----------------------------------------------------------------------------------#

#Plotting evolution of topics through time

#----------------------------------------------------------------------------------#

#Today
n = datetime.today().date().strftime("%B %d, %Y")

#Plot US
#Live evolution of topics
#For Turkey pitch I make newdf=newdf2
#newdf2=[]
newdf2=newdf
newdf2.dtypes

#newdf2['lda_w2v_topic_id'] = newdf2.lda_w2v_topic_id.astype(int)
#newdf2['nmf_w2v_topic_id'] = newdf2.nmf_w2v_topic_id.astype(int)

newdf2['cum_nmf_w2v_topic_id'] = newdf2.groupby('nmf_w2v_topic_id').cumcount()+1

#NMF-TCW2V
fig, ax = plt.subplots(figsize=(17,7))
plot_US = newdf2.pivot_table(index='date',columns='nmf_w2v_topic_id',values='cum_nmf_w2v_topic_id',aggfunc='count').plot(ax=ax)
plot_US
plt.xticks( rotation=80)
plt.ylabel('# of Tweets per Topic')
plt.legend(loc='center left',bbox_to_anchor=(1, 0.5), ncol=1, fancybox=True, shadow=True,title = 'Topic #')
plt.xlabel('Date')
plt.title('Live Evolution of Topics in the US (NMF_TC-W2V)',fontdict=None, loc='center')
plt.savefig(' Evolution of Topics in the US (NMF_TC-W2V).png', dpi=300, bbox_inches = 'tight')
#plt.show()


#Daily number of Topics

#NMF-TCW2V
fig, ax = plt.subplots(figsize=(17,7))
plot_US = newdf2.pivot_table(index='date',columns='nmf_w2v_topic_id',values='cum_nmf_w2v_topic_id',aggfunc='count').plot(kind= 'bar', ax=ax)
plot_US
plt.xticks( rotation=80)
plt.ylabel('# of Tweets per Topic')
plt.legend(loc='center left',bbox_to_anchor=(1, 0.5), ncol=1, fancybox=True, shadow=True,title = 'Topic #')
plt.xlabel('Date')
plt.title('Daily Evolution of Topics in the US (NMF_TC-W2V)',fontdict=None, loc='center')
plt.savefig('Daily Evolution of Topics in the US (NMF_TC-W2V).png', dpi=300, bbox_inches = 'tight')

#----------------------------------------------------------------------------------------------#

#Pre-Processing: Topic Number Optimization Tools#

#----------------------------------------------------------------------------------------------#


#Computing the Mimno et al. (2011) coherence score for each topic
#to get the best number of topics

def coherence_score(model, tdm, top_n):

    W = model.transform(tdm)
    H = model.components_
    topic_assignnment = np.argmax(W, axis=1)

    topic_coherence = []

    for topic_id, topic in enumerate(H):

        # Taking the documents assigned to that topic:
        idx = topic_assignnment==topic_id
        temp = tdm[idx,:]
        
        top_words = topic.argsort()[::-1][:top_n]
        coherence = 0.0

        for i in range(2, len(top_words)):
            for j in range(1, i - 1):
                
                word_i = np.array(temp[:,top_words[i]].todense().tolist())
                word_j = np.array(temp[:,top_words[j]].todense().tolist())
                               
                D12 = np.count_nonzero(word_i * word_j) + 1
                D2 = np.count_nonzero(word_j)
                
                coherence += math.log(D12/D2)
                
        topic_coherence.append(coherence)

    return topic_coherence


#We can use the following coherence metrics to find the correct number of topics.
for topic_id, value in enumerate(coherence_score(nmf, X, 10)):
    print("Topic %d's Coherence: %0.3f" %(topic_id, value))
    
    #Average topic coherence:
np.mean(coherence_score(nmf, X, 10))

#Computing the TC-W2V (Topic Coherence Word to Vector(or Word Embedding)) by O'Callahan et al.
#Requires re-computing the non negative matrix factorization only with words vectorized
# rather than transformed into a TF IDF
nlp = spacy.load('en')

vectorizer = CountVectorizer(stop_words=stopwords)
x = vectorizer.fit_transform(corpus.Tweet)
print("Num words:", len(vectorizer.get_feature_names()))

#Defining the function that computes the coherence score
def coherence_score_TCW2V(model, tdm, top_n):

    H = model.components_
    words = tdm.get_feature_names()
    topic_coherence = []

    for topic_id, topic in enumerate(H):
      
        top_words = topic.argsort()[::-1][:top_n]
        
        sim = 0.0
        pairs = 0

        for i in range(2, len(top_words)):
            for j in range(1, i - 1):
                
                word_i = nlp(words[top_words[i]])
                word_j = nlp(words[top_words[j]])
                
                if word_i.has_vector and word_j.has_vector:
                    sim += cosine_similarity(word_i.vector.reshape(1, -1), 
                                             word_j.vector.reshape(1, -1))[0][0]
                else:
                    continue
                
                pairs += 1
           
        topic_coherence.append(sim/pairs)

    return topic_coherence

for topic_id, value in enumerate(coherence_score_TCW2V(nmf, vectorizer, 10)):
    print("Semantic Coherence of Topic %d: %0.3f" %(topic_id, value))
    

# Average topic coherence:
np.mean(coherence_score_TCW2V(nmf, vectorizer, 10))    
print_top_words(nmf, tfidf.get_feature_names(),10)

    
#Optimizing the best number of topics
coh_s=[]    
for k in range(10, 100): # create a loop to try mutliple values.
    nmf = NMF(n_components=k)
    nmf.fit(x)
    w = nmf.transform(x)
    h = nmf.components_
    coh_mean = np.mean(coherence_score_TCW2V(nmf, vectorizer, 10))
    coh_s.append(coh_mean)
    print("The average coherence for k= %d is %0.4f " %(k, coh_mean))
opt_n = (np.argmax(coh_s)  + 10) #Tells us what is the best number of topics according to log likelihood

nmf = NMF(n_components=11) # Non negative matrix with 11 components


nmf.fit(x) # Fitting the TFIDF of the tweets to the matrix 
w = nmf.transform(x) #Transforming to the necessary W matrix 
print(w.shape)

h = nmf.components_ # now the H matrix
print(h.shape)

#Assigning topics to each tweet by looking at the largest column in W
topic_assignment = np.argmax(w, axis=1)