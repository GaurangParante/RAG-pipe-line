from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()
model = ChatGroq(model="qwen/qwen3-32b",reasoning_format='parsed')


print("Loading PDF Document...")
loader = PyPDFLoader("ai.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Embadding
# Text -> Number -> Vector

print("Building Embaddings...")
embaddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = InMemoryVectorStore.from_documents(all_splits,embaddings)

@tool
def retrive_context(query:str):
    '''Retrive relevent context from the PDF document based on the query.'''
    similar_docs = vector_store.similarity_search(query,k=3)
    data = []
    for doc in similar_docs:
        data.append(f'''
            content:{doc.page_content},
            source:{doc.metadata.get("source","unknown")}
        ''')
    return "\n\n".join(data)


prompt = "You are an agent who retrives context from pdf doc"
agent = create_agent(model,[retrive_context],system_prompt=prompt)

query = "Give me all arithmetic operators and how to use them"

for step in agent.stream({
    "messages":[
        {
            "role":"user",
            "content":query
        }
    ]
},stream_mode="values"):
    step["messages"][-1].pretty_print()
