import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback


import os

load_dotenv('.env')

def split_into_chunks(text):
    splitter = CharacterTextSplitter(separator='\n', chunk_size=600, chunk_overlap=200, length_function=len)
    chunks = splitter.split_text(text)
    return chunks

def extract_text_from_pdf(pdf):
    if pdf is not None:
        loader = PdfReader(pdf)
        extracted_text = ""
        for page in loader.pages:
            extracted_text += page.extract_text()
    return extracted_text

def chat():
    st.set_page_config(page_title="Ask your scientific paper")
    st.header("Ask your scientific paper 👈")

    #Upload file
    pdf = st.file_uploader("Upload your scientific paper", type="pdf")
    

    #Extract file
    if pdf is not None:
        loader = PdfReader(pdf)
        extracted_text = ""
        for page in loader.pages:
            extracted_text += page.extract_text()
        
        

        chunks = split_into_chunks(extracted_text)

    #     #Create embeddings
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_texts(chunks, embeddings)
        
        text_input = st.text_input("Ask you question about the paper", disabled=not pdf,)
        if text_input:
            docs = db.similarity_search(text_input)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=text_input)
                print(cb)
            st.write(response)
        

if __name__ == "__main__":
    chat()
