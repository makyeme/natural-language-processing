import streamlit as st
import torch

#nltk packages
import nltk
nltk.download('punkt')
nltk.download('stopwords')

#Sumy packages
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer


#Importing transformers
from transformers import pipeline
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
BART_PATH = 'facebook/bart-large-cnn'


#Web scrapping packages
from bs4 import BeautifulSoup
import requests



#function webscrapping
@st.cache
def get_text(raw_url):

    nested = []
    sent = []
    length = 0
    r = requests.get(raw_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    results = soup.find_all(['h', 'p'])
    text = [result.text for result in results]
    article = ' '.join(text)
    return article


# Function for extractive short/medium text summarization with BART
def short_textBert(document):
    summerizer = pipeline('summarization')
    my_summary = summerizer(document, max_length=500, min_length=200, do_sample=False)
    return my_summary[0]['summary_text']



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


#function to generate BART summary
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


# generating 2 level summary
def finalSummary(chunked_text):
    summary_one = generate_summary(chunked_text)
    nested_summ = nest_sentences(' '.join(summary_one))
    final_summary = generate_summary(nested_summ)
    return final_summary[0]



#sumy summarizer function
def sumy_summarizer(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer('english'))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document, 3)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result


def main():
    '''NLP app with streamlit'''
    st.title('NLP Simplified')
    st.subheader('Natural processing language on the click')

    activities = ['Summarize Via Text', 'Summazrize via URL']
    choice = st.sidebar.selectbox('Text Summarization: Input Mode', activities)

#Text summarization
    if choice == 'Summarize Via Text':
        
        st.subheader('Summarize Text')
        message = st.text_area('', 'Type text here' )
        summarizer_option = st.selectbox('Choose summarizer', ('sumy', 'BERT'))
        if st.button('Summarize via Text'): 

        
           #option sumy summerizer
            if summarizer_option == 'sumy':
                st.text("Using Sumy...")
                summary_result = sumy_summarizer(message)

            #option Bart summarizer
            elif summarizer_option == 'BERT':
                st.text('Using Bert...')
                summary_result = short_textBert(message)
                
            
            #return sumy as default incase of failures
            else: 
                st.warning('Using Default Summarizer')
                st.text('Using Sumy')
                summary_result = sumy_summarizer(message)

            #return summary
            st.success(summary_result)

 


    if choice == 'Summazrize via URL':
        st.subheader("Summarize URL")
        raw_url = st.text_input(' ', 'Enter URL here')
        summarizer_option = st.selectbox('Choose summarizer', ('sumy', 'BART'))

        if st.button("Summarize via URL"):
            if summarizer_option == 'sumy':
                st.text("Using Sumy...")
                result = get_text(raw_url)
                summary_result = sumy_summarizer(result)

            
            elif summarizer_option == 'BART':
                 st.text("Using Bart...")
                 book = get_text(raw_url)
                 result = nest_sentences(book)
                 summary_result = finalSummary(result)   
            
            
            #st.write(result)
            st.subheader("Summarized Text")
            
            
        #return summary
        st.success(summary_result)

            
    


if __name__ == '__main__':
    main()