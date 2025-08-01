import os
import shutil
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- 1. App Configuration and Setup ---
st.set_page_config(
    page_title="Chat with Your Documents",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Chat with Your Documents")
st.markdown("Ask questions about the content of your PDF files. This app uses a RAG pipeline with Groq, LangChain, and ChromaDB.")

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found. Please add it to your .env file.")
    st.stop()

# --- 2. Caching and Data Loading ---
# Use Streamlit's caching to load the model and data only once.
@st.cache_resource
def load_and_process_data():
    """
    Loads documents from the 'books' directory, splits them into chunks,
    and creates a Chroma vector store. This function is cached.
    """
    with st.spinner("Loading documents and creating vector store... This may take a moment."):
        # Ensure the 'books' directory exists
        if not os.path.exists('books'):
            st.error("The 'books' directory was not found. Please create it and add your PDF files.")
            return None

        # Load documents
        loader = DirectoryLoader(
            path='books',
            glob='*.pdf',
            loader_cls=PyPDFLoader
        )
        docs = loader.load()

        if not docs:
            st.warning("No PDF documents found in the 'books' directory.")
            return None

        # Split documents into smaller chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        # Initialize embedding model
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

        # Create Chroma vector store from chunks
        vector_store = Chroma.from_documents(documents=chunks, embedding=embedding_model)
        
        return vector_store

# Load the vector store
vector_store = load_and_process_data()

if vector_store:
    st.success(f"Vector store created successfully from your documents!")
else:
    st.stop()


# --- 3. LangChain RAG Chain Setup ---
# Initialize the LLM
llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-8b-8192")

# Create the retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Define the prompt template
prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer ONLY from the provided context. Your answer should be concise and to the point.
    If the context is insufficient to answer the question, just say "I don't have enough information to answer that question."

    CONTEXT:
    {context}
    
    QUESTION:
    {question}
    """,
    input_variables=['context', 'question']
)

# Helper function to format the retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build the RAG chain using LCEL
rag_chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 4. Streamlit UI for Interaction ---
st.subheader("Ask a Question")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_question := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_question})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_question)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Get the answer from the RAG chain
            answer = rag_chain.invoke(user_question)
            st.markdown(answer)
            
            # Show the retrieved context in an expander
            with st.expander("Show Retrieved Context"):
                retrieved_docs = retriever.invoke(user_question)
                st.write(retrieved_docs)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})
