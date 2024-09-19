#Importing the libraries required as we go.

import speech_recognition as sr
import requests as rq
from bs4 import BeautifulSoup as bs
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

#Making a function to Extracting Data from URL using BeautifulSoup4
def extract_text_from_url(url):
    #Using the URL to get the html page content
    page = rq.get(url)
    soup = bs(page.content, "html.parser")
    
    #Find all <p></p> text content and saving them
    paragraphs = soup.find_all("p")
    text = " ".join([p.get_text() for p in paragraphs])
    
    # print(text)
    return text

url = input(str("Input the URL:"))
text = extract_text_from_url(url)

# Tokenizing the text
stopwords = set(stopwords.words("English"))
tokenixed_text = word_tokenize(text)

#Scoring each word in the text by frequency
table = dict()
for words in tokenixed_text:
    #Converting the words to lower
    words = words.lower()
    
    # ignoring stopwords
    if words in stopwords:
        continue
    
    # Adding the unique word in the dict initialised
    if words not in table:
        table[words]=1
        
    # Increasing the score when words are repeated
    if words in table:
        table[words]+=1
        
