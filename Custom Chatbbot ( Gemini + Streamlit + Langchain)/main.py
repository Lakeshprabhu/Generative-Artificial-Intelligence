import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
import os

load_dotenv()

key = os.getenv("GEMINI")

embeddings = GoogleGenerativeAIEmbeddings(google_api_key=key  )
st.header("My first Chatbot")

st.subheader("This is the content")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions",type="pdf")


#tHIS IS THE PART OF CODE FOR TEXT EXTRACTION
if file is not None:
    pdf_read = PdfReader(file)
    text = ""
    for page in pdf_read.pages:
        text+=page.extract_text()


    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",chunk_size=1000,chunk_overlap=150,length_function=len
    )

    abc = text_splitter.split_text(text)
    st.write(abc)


vector_store = FAISS.from_texts(abc,embeddings)






#TO BREAK INTO THE CHUNKS