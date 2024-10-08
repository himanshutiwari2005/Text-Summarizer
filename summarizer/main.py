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
tokenized_text = word_tokenize(text)

def summarize(text):
    #Scoring each word in the text by frequency
    table = dict()
    for words in tokenized_text:
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
        
    #Scoring each sentence in the text
    sentences_in_text = sent_tokenize(text)

    #Making a dict to store value of sentences
    sentences_in_text_dict = dict()

    #Traversing the tokenized text
    for sentence in sentences_in_text:
        for words, freq in table.items():
            """So here is how we do this, we have found the most repeated words above.
            Now we will find them in the sentences and a sentence's score is increased by the frequency of words in it."""
            if words in sentence.lower():
                """Updating unique sentences in the dict declared above"""
                if sentence in sentences_in_text_dict:
                    sentences_in_text_dict[sentence] += int(freq)
                else:
                    sentences_in_text_dict[sentence] = int(freq)
                    
                
    #Finding the total score of the text
    Score = 0
    for sentences in sentences_in_text_dict:
        Score += sentences_in_text_dict[sentences]
    
    # Finding the average value of text
    average = int(Score / len(sentences_in_text_dict))

    #Summarzing the text by taking sentences that have atleast 20% higher score than an average sentence
    summary = ''
    for sentence in sentences_in_text:
        if (sentence in sentences_in_text_dict) and (sentences_in_text_dict[sentence]>(1.3 * average)):
            summary += " " + sentence
        
    return summary

summary_from_url = summarize(text)
print(summary_from_url)
