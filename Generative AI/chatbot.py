import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

# App title
st.title("PDF File Analyzer")
st.subheader("Upload a PDF file to analyze its content")
# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Read the PDF file
    pdf_reader = PdfReader(uploaded_file)
    pdf_content = ""
    for page in pdf_reader.pages:
        pdf_content += page.extract_text() + "\n"

    # Display file content
    # st.write("### File Content:")
    # st.text_area("PDF Content", pdf_content, height=300)

    # Break into Chunks
    text_splitter = RecursiveCharacterTextSplitter(separators='\n', chunk_size=100, chunk_overlap=100,
                                                   length_function=len)
    chunks = text_splitter.split_text(pdf_content)
    # st.write(chunks)
    #Generate Embeddings
    embeddings = OpenAIEmbeddings()
    #Creating Vector Database
    vector_store = FAISS.from_texts(chunks,embeddings)
    user_question = st.text_input('Enter your question here')

    if user_question :
        match = vector_store.similarity_search(user_question)
        #define the LLM
        llm = ChatOpenAI(temperature=0 , max_tokens = 1000, model_name = 'gpt-4o-mini' )

        #output results
        chain = load_qa_chain(llm , chain_type='stuff')
        response = chain.run(input_documents = match , question = user_question)
        st.write(response)





