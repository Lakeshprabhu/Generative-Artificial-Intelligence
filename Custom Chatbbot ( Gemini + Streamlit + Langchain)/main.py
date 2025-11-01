import streamlit as st
from PyPDF2 import PdfReader

st.header("My first Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions",type="pdf")



#tHIS IS THE PART OF CODE FOR TEXT EXTRACTION
if file is not None:
    pdf_read = PdfReader(file)
    text = ""
    for page in pdf_read.pages:
        text+=page.extract_text()



#TO BREAK INTO THE CHUNKS