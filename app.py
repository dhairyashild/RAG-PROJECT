from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma


file_path = "./GenAI_Agentic_AI_Notes.pdf"
loader = PyMuPDFLoader(file_path)
docs= loader.load()
print(f"DOCS NUMBER= {len(docs)}")



text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(docs)     # Use split_documents instead of split_text
print(f"TEXTS= {len(texts)}")


# FROM BWLOW ALL CHUNKING CODE NO NEED AS CHROMA AGAIN CONVERT TEXT INTO EMDING NUMBER JUST NEED MODEL CODE =embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", region_name="us-east-1")
chunks = [text.page_content for text in texts]
embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0", region_name="us-east-1"
)
vector = embeddings.embed_documents(
    chunks
)
print(f"EMBEDDINGS GENERATED: {len(vector)} vectors")
print(vector[0][0:10])
#--------------------------------------------------------------------------------------
### VECTOR STORE 
#1 = GO BELOW IN -->Select vector store:AND COPY CHROMA CODE FOR  INTIALIZATION + ADD DOCUMENT CODE BOTH DONE IN THIS CODE BY CHANGING CHROMA CODE LIKE BELOW
# 1. INITIALIZING PERSISTENT STORAGE= InMemoryVectorStore CODE GIVEN AT TOP PAGE WE NOT TOOK AS IT STORE IN RAM

# ✅ BEST PROD STYLE: build Chroma directly from Documents (NO manual `embed_documents`)
vector_store = Chroma.from_documents(              # <--- ADDED: use from_documents
    documents=texts,                               # <--- ADDED FULL LINE NEW: pass split Documents   
    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100) +texts = text_splitter.split_documents(docs) -This already does the chunking, so you don’t need a separate chunks
    embedding=embeddings,                          # <--- CHANGED: param name `embedding`
    collection_name="example_collection",
    persist_directory="./chroma_langchain_db",     # <--- your local persistent folder to save vectors so they aren't lost.
)
#--------------------------------------------------------------------------------------

    # 1. RETRIEVER CONVERSION: Creating the Search Interface
# Found in: Chroma Documentation -> "Query by turning into retriever" section.
# Use: Converts the 'vector_store' database into a 'retriever' object for the RAG chain.
# Logic: 'mmr' ensures diverse results, 'k=3' returns the top 3 most relevant chunks.
retriever = vector_store.as_retriever(
    search_type="mmr",       
    search_kwargs={"k": 3, "fetch_k": 10}
)
#--------------------------------------------------------------------------------------

# 1. RETRIEVER CONVERSION: Creating the Search Interface
# Found in: Chroma Documentation -> "Query by turning into retriever" section.
# Use: Converts the 'vector_store' database into a 'retriever' object for the RAG chain.
# Logic: 'mmr' ensures diverse results, 'k=3' returns the top 3 most relevant chunks.
retriever = vector_store.as_retriever(
    search_type="mmr",       
    search_kwargs={"k": 3, "fetch_k": 10}
)

#--------------------------------------------------------------------------------------

# 1. GENERATION: INITIALIZING THE LLM (Bedrock)
# Use: Sets up the AI "Brain" (Claude 3) to process retrieved context.
# Changes: Uses ChatBedrock to communicate with AWS models.
from langchain_aws import ChatBedrock

llm = ChatBedrock(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    region_name="us-east-1",
    model_kwargs={"temperature": 0.1}
)

# 2. DEFINING THE PROMPT TEMPLATE
# Use: Instructions that force the AI to answer ONLY from your PDF notes.
# Found in: LangChain Docs -> "Prompt Templates" section.
from langchain_core.prompts import ChatPromptTemplate

template = """Answer the question based ONLY on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
#--------------------------------------------------------------------------------------

# 1. CHAIN CONSTRUCTION: The RAG Pipeline
# Use: Connects Retriever -> Prompt -> LLM -> Output Parser into one flow.
# Found in: LangChain Docs -> "Expression Language (LCEL)" section.
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# The '|' (pipe) operator passes data from one step to the next automatically.
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 2. EXECUTION: Asking the Actual Question
# Use: Triggers the entire system to read your PDF and answer.
query = "Explain the key features of Agentic AI from these notes."
response = rag_chain.invoke(query)

print(f"AI RESPONSE:\n{response}")