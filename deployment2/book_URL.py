#import libraries
import streamlit as st
from transformers import pipeline
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import re
import torch
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
BART_PATH = 'facebook/bart-large-cnn'
from rouge import Rouge
from bs4 import BeautifulSoup
import requests

#url to summarize
url = "https://www.nytimes.com/2021/04/30/technology/robot-surgery-surgeon.html"


#function to get url
def get_url(URL):
    r = requests.get(URL)
    soup = BeautifulSoup(r.text, 'html.parser')
    results = soup.find_all(['body'])
    text = [result.text for result in results]
    article = ' '.join(text)
    return article


#creating variable of text to be summarized with url
my_book = get_url(url)



#function to chunk text
def nest_sentences(document):
    nested = []
    sent = []
    length = 0
    for sentence in nltk.sent_tokenize(document):
        length += len(sentence)
        if length < 500:
            sent.append(sentence)
        else:
            nested.append(sent)
            sent = []
            length = 0

    if sent:
        nested.append(sent)

    return nested

#Creating variable for chunked text
nested = nest_sentences(my_book)


#function to generate summary

def generate_summary(nested_sentences):
    
    #model for summariation & tokenization
    bart_model = BartForConditionalGeneration.from_pretrained(BART_PATH, output_past=True)
    bart_tokenizer = BartTokenizer.from_pretrained(BART_PATH, output_past=True)
     
    # using cuda for summarization
    device = 'cuda'
    summaries = []
    for nested in nested_sentences:
        input_tokenized = bart_tokenizer.encode(' '.join(nested), truncation=True, return_tensors='pt')
        input_tokenized = input_tokenized.to(device)
        summary_ids = bart_model.to('cuda').generate(input_tokenized,
                                        length_penalty=3.0,
                                        min_length=30,
                                        max_length=100)
        output = [bart_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
        summaries.append(output)
    summaries = [sentence for sublist in summaries for sentence in sublist]
    return summaries


# generating two level summary

def finalSummary(chunked_text):
    summary_one = generate_summary(chunked_text)
    nested_summ = nest_sentences(' '.join(summary_one))
    final_summary = generate_summary(nested_summ)
    print(final_summary)


#Calling summary function on chunked text
finalSummary(nested)


