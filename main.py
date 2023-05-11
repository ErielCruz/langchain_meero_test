import tiktoken
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from pathlib import Path
from typing import List
from langchain.docstore.document import Document

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
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

def get_response_from_query(db, query, k=10):
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
        You are a helpful assistant that that can answer questions from Notion 
        based on the Notion documents: {docs}
        
        Only use the factual information from the documents to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=query, docs=docs_page_content)
    # response = response.replace("\n", "")

    query_tokens = num_tokens_used(query)
    docs_tokens = num_tokens_used(docs_page_content)
    response_tokens = num_tokens_used(response)
    total_tokens = query_tokens + docs_tokens + response_tokens

    return response, docs, total_tokens


# Example usage:
create_db_from_notion_docs()

# Now we can load the persisted database from disk, and use it as normal.
# db = get_vectordb_documents()

# query = "What can we learn from the interview to Junot?"
# response, docs, total_tokens = get_response_from_query(db, query)

# query_cost = round((total_tokens / 1000) * 0.002, 3)

# print(textwrap.fill(response, width=120))
# print(f'\nCost of query = ${query_cost}')
# print(f'Total # of tokens used = {total_tokens}\n')