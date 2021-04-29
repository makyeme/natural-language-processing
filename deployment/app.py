import streamlit as st
#NLP packages
import spacy
from textblob import TextBlob #sentimntal analysis

#function for text input
def text_analyzer(my_text):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(my_text)
    tokens = [token.text for token in docx]
    my_data = [(f'Token: {token.text} - Lemma: {token.lemma_}') for token in docx]
    return my_data



def main():
    '''NLP app with streamlit'''
    st.title('NLPiffy with streamlit')
    st.subheader('Natural processing language on the click')

    #Tokenization
    if st.checkbox('Show Tokens and Lemma'):
        st.subheader('Tokenize your Text')
        message = st.text_area('Enter your Text', 'Type text here' )
        if st.button('Analyze'): 
            nlp_result = text_analyzer(message)
            st.json(nlp_result)














if __name__ == '__main__':
    main()
