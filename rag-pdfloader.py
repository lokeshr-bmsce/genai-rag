from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import os
from langchain_community.document_loaders import PyPDFLoader


load_dotenv()

# Constants
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"

llm = None
vector_store = None

# Define headers to mimic a real browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
}

def initialize_components():
    print('in initialize components method.....')
    global llm, vector_store
    if llm is None:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9, max_tokens=500)
    print("groq setup is done.")
    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR)
        )


def process_pdf(file_path: Path):
    """
    This function scraps data from a url and stores it in a vector db
    :param urls: input urls
    :return:
    """
    initialize_components()


    vector_store.reset_collection()

    #loader = UnstructuredURLLoader(urls=urls, headers = headers)
    loader = PyPDFLoader(str(file_path))
    data = loader.load()
    print('pdf data...')
    print(data[0].page_content)
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE
    )
    docs = text_splitter.split_documents(data)

    print(f"Number of documents to add to vector store: {len(docs)}")
    print(f"Number of documents in vector store before adding: {vector_store._collection.count()}")
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)
    print(f"Number of documents in vector store after adding: {vector_store._collection.count()}")

def generate_answer(query):
    if not vector_store:
        raise RuntimeError("Vector database is not initialized ")

    retriever = vector_store.as_retriever()

    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_store.as_retriever())
    result = chain.invoke({"question": query}, return_only_outputs=True)
    sources = result.get("sources", "")

    return result['answer'], sources


if __name__ == "__main__":
    print('in main method.....')
    file_path = Path("zenova_results.pdf")
    process_pdf(file_path)
    answer, sources = generate_answer("Tell me what is the raise in Revenue for Zenova?")
    print(f"Answer: {answer}")
    print(f"Sources: {sources}")