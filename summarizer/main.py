#Importing the libraries required as we go.

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import speech_recognition as sr
import requests as rq
from bs4 import BeautifulSoup as bs

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

