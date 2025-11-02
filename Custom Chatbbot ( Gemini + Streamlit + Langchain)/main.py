import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("CHAT")

key = os.getenv("CHAT")

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
    embeddings = OpenAIEmbeddings(openai_api_key =key ,  model="text-embedding-3-small")



    vector_store = FAISS.from_texts(abc,embeddings)

    user_question = st.text_input("Type your question here")

    if user_question:
        match = vector_store.similarity_search(user_question)


        llm =ChatOpenAI( temperature=0.2)
        prompt = ChatPromptTemplate.from_template(
            "Use the following context to answer the question.\n\n{context}\n\nQuestion: {input}"
        )
        chain = create_stuff_documents_chain(llm,prompt = prompt)
        response = chain.invoke({
            "context": match,  # this must be a list of Documents
            "input": user_question  # this is your actual question text
        })
        st.write(response)





#TO BREAK INTO THE CHUNKS