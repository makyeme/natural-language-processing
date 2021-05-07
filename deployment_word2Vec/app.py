import streamlit as st
from spacy import displacy

#NLP packages
import spacy
from textblob import TextBlob #sentimntal analysis
from gensim.summarization import summarize


#Sumy packages
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

#nltk packages
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize

#Web scrapping packages
from bs4 import BeautifulSoup
from urllib.request import urlopen





#function for tokkenization & lemmatization
def text_analyzer(my_text):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(my_text)
    tokens = [token.text for token in docx]
    my_data = [(f'Token: {token.text} - Lemma: {token.lemma_}') for token in docx]
    return my_data



#function for entity analysis(extraction)
def entity_analyzer(my_text):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(my_text)
    tokens = [token.text for token in docx]
    entities = [(f'Entity: {entity.text} - Label: {entity.label_}') for entity in docx.ents]
    allData = [f'Token: {tokens} - Entities: {entities}']#return both tokens & entities
    return entities


#Function for webscrapping
@st.cache
def get_text(raw_url):
    page = urlopen(raw_url)
    soup = BeautifulSoup(page)
    fetched_text = ' '.join(map(lambda p:p.text,soup.find_all('p')))
    return fetched_text


#Sumamry functions

#sumy summarizer function
def sumy_summarizer(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer('english'))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document, 3)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result

#nltk summarizer function
def nltk_summarizer(docx):
    stopWords = set(stopwords.words("english")) 
    words = word_tokenize(docx) 
    freqTable = dict() 
    for word in words: 
        word = word.lower() 
        if word in stopWords: 
            continue
        if word in freqTable: 
            freqTable[word] += 1
        else: 
            freqTable[word] = 1
       
    sentences = sent_tokenize(docx) 
    sentenceValue = dict() 
   
    for sentence in sentences: 
        for word, freq in freqTable.items(): 
            if word in sentence.lower(): 
                if sentence in sentenceValue: 
                    sentenceValue[sentence] += freq 
                else: 
                    sentenceValue[sentence] = freq 
                    
    sumValues = 0
    for sentence in sentenceValue: 
        sumValues += sentenceValue[sentence] 
        
    average = int(sumValues / len(sentenceValue)) 
    
    summary = '' 
    for sentence in sentences: 
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.5 * average)): 
            summary += " " + sentence 
    return summary 




def main():
    '''NLP app with streamlit'''
    st.title('NLP Simplified')
    st.subheader('Natural processing language on the click')

    #Tokenization
    if st.checkbox('Tokenize | Lemmatize'):
        st.subheader('Tokenize | Lemmatize your Text')
        message = st.text_area(' ', 'Type text here' )
        if st.button('Excute'): 
            nlp_result = text_analyzer(message)
            st.json(nlp_result)

#Named Entity (extract names of: organisations, places, persons) 
    if st.checkbox('Entity Extraction'):
        st.subheader('Extarct Entities from your Text')
        message = st.text_area(' ', 'Type text here' )
        if st.button('Extract'): 
            nlp_result = entity_analyzer(message)
            st.json(nlp_result)

#Sentimental analysis
    if st.checkbox('Sentimental Analysis'):
        st.subheader('Sentiment in your Text')
        message = st.text_area(' ', 'Enter text here' )
        if st.button('Analyze'): 
            blob = TextBlob(message)
            sentiment_result = blob.sentiment
            st.success(sentiment_result)


    activities = ['Summarize Via Text', 'Summazrize via URL']
    choice = st.sidebar.selectbox('Text Summarization: Input Mode', activities)

#Text summarization
    if choice == 'Summarize Via Text':
        
        st.subheader('Summarize Text')
        message = st.text_area('', 'Type text here' )
        summarizer_option = st.selectbox('Choose summarizer', ('gensim', 'sumy', 'NLTK'))
        if st.button('Summarize via Text'): 

            #option gensim summarizer
            if summarizer_option == 'gensim':
                st.text("Using Gensim...")
                summary_result = summarize(message)
            
            #option sumy summerizer
            elif summarizer_option == 'sumy':
                st.text("Using Sumy...")
                summary_result = sumy_summarizer(message)

            #option NLTK summarizer
            elif summarizer_option == 'NLTK':
                st.text('Using NLTK...')
                summary_result = nltk_summarizer(message)
                
            
            #return gensim as default incase of failures
            else: 
                st.warning('Using Default Summarizer')
                st.text('Using Gensim')
                summary_result = summarize(message)

            #return summary
            st.success(summary_result)

    if choice == 'Summazrize via URL':
        st.subheader("Summarize URL")
        raw_url = st.text_input(' ', 'Enter URL here')
        if st.button("Summarize"):
            result = get_text(raw_url)
            
            #st.write(result)
            st.subheader("Summarized Text")
            docx = sumy_summarizer(result)

            html = docx.replace("\n\n" , "\n")
            st.markdown(html,unsafe_allow_html=True)



if __name__ == '__main__':
    main()
