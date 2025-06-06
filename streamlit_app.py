import os
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Set OpenAI key from Hugging Face secret
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

@st.cache_data
def load_text():
    with open("teaching-in-a-digital-age.txt", "rb") as file:
        raw_bytes = file.read()
        return raw_bytes.decode("utf-8", errors="ignore")  # or "replace"
        
@st.cache_resource
def setup_qa():
    full_text = load_text()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([full_text])
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.3)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

st.title("📘 Teaching in a Digital Age Bot")
st.markdown("Ask a question based on Tony Bates' open textbook about digital pedagogy:")

qa = setup_qa()
query = st.text_input("Your question")

if query:
    st.markdown("### Answer")
    st.write(qa.run(query))
