import tiktoken
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from pathlib import Path
from typing import List
from langchain.docstore.document import Document

from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
import textwrap

load_dotenv()

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

def start_convo(db, query, k=10):
    """
    gpt-3.5-turbo can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = db.similarity_search_with_score(query, k=k)
    docs_data = [doc for doc, _ in docs]
    docs_page_content = " ".join([d.page_content for d in docs_data])

    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

    # Template to use for the system message prompt
    template = """
        You are a helpful chatbot having a conversation with a human.

        Given the following extracted parts of a long Notion document and a question, create a final answer.
        
        Only use the factual information from the documents to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.

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

    chain = load_qa_chain(llm=chat, chain_type="stuff", memory=memory, prompt=chat_prompt)

    response = chain({"input_documents": docs_data, "human_input": query}, return_only_outputs=True)
    response_text = response['output_text']

    query_tokens = num_tokens_used(query)
    docs_tokens = num_tokens_used(docs_page_content)
    response_tokens = num_tokens_used(response_text)
    total_tokens = query_tokens + docs_tokens + response_tokens

    return response_text, chain, total_tokens

def print_results(response_text, total_tokens):
    query_cost = round((total_tokens / 1000) * 0.002, 3)

    print(response_text)
    print(f'\nCost of query = ${query_cost}')
    print(f'Total # of tokens used = {total_tokens}\n')


def continue_convo(db, query, chain, k=10):
    """
    gpt-3.5-turbo can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = db.similarity_search_with_score(query, k=k)
    docs_data = [doc for doc, _ in docs]
    docs_page_content = " ".join([d.page_content for d in docs_data])

    response = chain({"input_documents": docs_data, "human_input": query}, return_only_outputs=True)
    response_text = response['output_text']

    query_tokens = num_tokens_used(query)
    docs_tokens = num_tokens_used(docs_page_content)
    response_tokens = num_tokens_used(response_text)
    total_tokens = query_tokens + docs_tokens + response_tokens

    return response_text, chain, total_tokens


# Example usage:
# create_db_from_notion_docs()

# Now we can load the persisted database from disk, and use it as normal.
db = get_vectordb_documents()

query = "What can we learn from the interview to Junot?"
response_text, chain, total_tokens = start_convo(db, query)

print_results(response_text, total_tokens)

query_2 = "Did another client provide similar feedback?"
response_text, chain, total_tokens = continue_convo(db, query_2, chain)

print_results(response_text, total_tokens)