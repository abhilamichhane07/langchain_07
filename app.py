import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import  HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template,user_template
from langchain.llms import HuggingFaceHub
import streamlit as st
from langchain_community.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
  
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=100,
        chunk_overlap=80,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore
  

def get_conversation_chain(vectorstore):
    
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
    memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever= get_vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain



def main():
  
  load_dotenv()
  st.set_page_config(
    page_title="Chat with Scientists",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
    
)
  from sentence_transformers import SentenceTransformer

  model_name = "bert-base-uncased"  # Example model name
  model = SentenceTransformer(model_name)
  
  st.write(css, unsafe_allow_html = True)
  st.header("Chat with multiple PDFS:books:")
  st.text_input("Asks a question about your documents")
  
  st.write(user_template.replace("{{MSG}}","Hello robot"), unsafe_allow_html = True)
  st.write(bot_template.replace("{{MSG}}","Hello human"),unsafe_allow_html = True)
  
  with st.sidebar:
    st.subheader("Your documents")
    pdf_docs = st.file_uploader("Upload your pdfs here and click on `Process`",accept_multiple_files=True)
    if st.button("Process"):
      with st.spinner("Processing"):
                #  get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)

                # # # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
            


if __name__ == '__main__':
    main()
