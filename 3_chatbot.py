# import basics
import os
from dotenv import load_dotenv

# import streamlit
import streamlit as st

# import langchain
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage

# load environment variables
load_dotenv()  

###############################   INITIALIZE EMBEDDINGS MODEL  #################################################################################################

embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
)

###############################   INITIALIZE CHROMA VECTOR STORE   #############################################################################################

vector_store = Chroma(
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding_function=embeddings,
    persist_directory=os.getenv("DATABASE_LOCATION"), 
)


###############################   INITIALIZE CHAT MODEL   #######################################################################################################

llm = init_chat_model(
    os.getenv("CHAT_MODEL"),
    model_provider=os.getenv("MODEL_PROVIDER"),
    temperature=0
)

       


# Simple RAG function
def get_answer(question):
    # Retrieve relevant documents
    docs = vector_store.similarity_search(question, k=2)
    
    # Build context
    context = ""
    sources = []
    for doc in docs:
        context += f"Content: {doc.page_content}\n\n"
        sources.append(doc.metadata['source'])
    
    # Create prompt
    prompt = f"""Answer the question based on the context provided.

Question: {question}

Context:
{context}

Provide a clear answer. If the context doesn't contain relevant information, say "I don't know"."""
    
    # Get response
    response = llm.invoke(prompt)
    
    # Add sources if we have an answer
    answer = response.content
    if sources and "I don't know" not in answer.lower():
        answer += f"\n\nSources: {', '.join(set(sources))}"
    
    return answer

# initiating streamlit app
st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🦜")
st.title("Chase's Agentic RAG Chatbot")

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)


# create the bar where we can type messages
user_question = st.chat_input("How are you?")


# did the user submit a prompt?
if user_question:

    # add the message from the user (prompt) to the screen with streamlit
    with st.chat_message("user"):
        st.markdown(user_question)

        st.session_state.messages.append(HumanMessage(user_question))


    # Get answer using simple RAG
    ai_message = get_answer(user_question)

    # adding the response from the llm to the screen (and chat)
    with st.chat_message("assistant"):
        st.markdown(ai_message)

        st.session_state.messages.append(AIMessage(ai_message))

