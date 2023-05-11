import tiktoken
import streamlit as st 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from pathlib import Path
from typing import List
from langchain.docstore.document import Document

from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

load_dotenv()

with open('app_explanation.md', 'r') as file:
    app_explanation = file.read()

with open('templates_guidelines.md', 'r') as file:
    templates_guidelines = file.read()

def load_notion_documents(path="Notion_DB/") -> List[Document]:
    """Load documents."""
    ps = list(Path(path).glob("**/*.md"))
    docs = []
    for p in ps:
        with open(p, encoding="utf-8") as f:
            text = f.read()
        metadata = {"source": str(p)}
        docs.append(Document(page_content=text, metadata=metadata))
    return docs

def create_db_from_notion_docs():

    documents = load_notion_documents()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("faiss_index")

    return db
    
def get_vectordb_documents():

    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local("faiss_index", embeddings)

    return db

def num_tokens_used(string: str, model_name: str ="gpt-3.5-turbo") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def answer(db, query, template_type, k=10):
    """
    gpt-3.5-turbo can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = db.similarity_search_with_score(query, k=k)
    docs_data = [doc for doc, _ in docs]
    docs_page_content = " ".join([d.page_content for d in docs_data])
    docs_page_sources = set(d.metadata['source'] for d in docs_data)

    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Template to use for the system message prompt
    template = template_type + """

        {context}

        {chat_history}
        Human: {human_input}
        Chatbot:
        """

    chat_prompt = PromptTemplate(
    input_variables=["chat_history", "human_input", "context"], 
    template=template
    )

    memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")

    chain = load_qa_chain(llm=chat, memory=memory, prompt=chat_prompt)

    response = chain({"input_documents": docs_data, "human_input": query}, return_only_outputs=True)
    response_text = response['output_text']

    query_tokens = num_tokens_used(query)
    docs_tokens = num_tokens_used(docs_page_content)
    response_tokens = num_tokens_used(response_text)
    total_tokens = query_tokens + docs_tokens + response_tokens

    return response_text, chain, docs_page_sources, memory, total_tokens


# Example usage:
# create_db_from_notion_docs()

# Load the persisted database from disk
db = get_vectordb_documents()

st.title('Insights from User Research Sessions')

st.markdown(app_explanation)
st.markdown(templates_guidelines)

template_type = st.radio(
    "What would you like the LLM to do?",
    ('Answer', 'Summarize'))

answer_questions = """
    You are a helpful assistant that that can answer questions from Notion.

    Given the following extracted parts of one or many Notion documents and a question, create a final answer.
    
    Only use the factual information from the documents to answer the question.
    
    If you feel like you don't have enough information to answer the question, say "I don't know".
    
    Your answers should be verbose and detailed.
    """

summarize = """
    You are a helpful assistant that that can summarize content from Notion.

    Given the following extracted parts of one or many Notion documents and a request, create a final answer.
    
    Only use the factual information from the documents to complete the request.
    
    If you feel like you don't have enough information to complete the request, say "I don't know".
    
    Your answers should be verbose and detailed.
        """

if template_type == "Answer":
    template_context = answer_questions
    with st.expander('Template text'):
        st.write(answer_questions)
elif template_type == "Summarize":
    template_context = summarize
    with st.expander('Template text'):
        st.write(summarize)    

col1, col2 = st.columns([9, 1])
text_input = col1.text_input('Enter your question or request here')
col2.write("   ")
col2.write("   ")

# Show stuff to the screen if there's a prompt
if col2.button('Ask'):
        
    with st.spinner('Fetching documents data...'):
        response_text, chain, docs_page_sources, memory, total_tokens = answer(db, text_input, template_type=template_context)
    print(response_text)
    st.write(response_text) 

    with st.expander('Sources'): 
        st.json(docs_page_sources)
