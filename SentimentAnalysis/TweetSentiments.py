#######USe the Python34 Interpreter for this project
import tweepy
import datetime
import csv
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob

print('Twitter Sentiment Analysis')

#Authorization Basics
consumer_Key = 'WiRZ65Ea6JxhO6yo5QdUFxaYL'
consumer_secret = '68UUAs6gfzqLGBJxMHT2YnSuxdCSfzX6MOzTSZXla9z10Ukuls'
access_token = '1043167512903315457-mi1vYIAuntvdbYNLrSs2aZGxqOV0Bl'
access_token_secret = '64BfQQGdtCCmpip3oB8KCAoi8kEEvSCDHFmBf7wKFdmtI'

#Twitter Connection
authT= tweepy.OAuthHandler(consumer_Key,consumer_secret)
authT.set_access_token(access_token,access_token_secret)
tapi=tweepy.API(authT)

#User Inputs
print("Input the topic you want to search tweets for")
topic=input()
formatstr='%Y/%m/%d'
print("Enter the Time frame From and To(YYYY/MM/DD)")
fromDate=input()
datetimeObj1=datetime.datetime.strptime(fromDate,formatstr)
fromDate=datetimeObj1.date()
toDate=input()
datetimeObj2=datetime.datetime.strptime(toDate,formatstr)
toDate=datetimeObj2.date()


public_tweet=tapi.search(q=topic,count=5,since=fromDate,until=toDate)
allPolarity=[]
allSubjectivity=[]

with open('tweetData.csv',mode='w',newline="") as tweetsCSV:
    tweetWriter=csv.writer(tweetsCSV)
    tweetWriter.writerow(['Tweet','UserName'])
    for tweet in public_tweet:
        tweet_text = tweet.text
        print(tweet_text.encode('utf-8'))
        tweet_user = tweet.user.name
        print(tweet_user.encode('utf-8'))
        tweetWriter.writerow([tweet_text.encode('utf-8'), tweet_user.encode('utf-8')])

        sentAnalysis=TextBlob(str(tweet))
        allPolarity.append(sentAnalysis.sentiment[0])
        allSubjectivity.append(sentAnalysis.sentiment[1])
#Polarity=how positive or negative text is
#Subjectivity=how much of opinion it is vs how factual

print(len(allPolarity))
#plt.plot(allPolarity,allSubjectivity)
#plt.show()


def scatterplot(x,y, color="r",
                yscale_log=False):
    _, ax=plt.subplots()
    ax.scatter(x, y, s = 10, color = color, alpha = 0.75)
    if yscale_log == True:
        ax.set_yscale('log')

    ax.set_title("Polarity vs Subjectivity")
    ax.set_xlabel("Polarity")
    ax.set_ylabel("Subjectivity")


scatterplot(allPolarity,allSubjectivity)
plt.show()

