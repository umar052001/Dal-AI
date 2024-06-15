from flask import Flask, request
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from flask_cors import CORS
from langchain_openai import OpenAIEmbeddings
import os
from langchain import hub

app = Flask(__name__)
CORS(app, origins=["*"])

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

# Define constants
FOLDER_PATH = "db"
MODEL_NAME = "text-embedding-ada-002"

# Initialize the LLM with the Qwen2-72B Instruct model
cached_llm = Ollama(model="qwen2:72b-instruct")

# Initialize OpenAI Embeddings
embedding = OpenAIEmbeddings(model=MODEL_NAME)

# Define text splitter for document chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    length_function=len,
    is_separator_regex=False
)

# Define the prompt template
raw_prompt = hub.pull("rlm/rag-prompt")


@app.route("/ai", methods=["POST"])
def ai_post():
    """
    Endpoint to handle simple chat requests.
    """
    print("Post /ai called")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    response = cached_llm.invoke(query)

    print(response)

    response_answer = {"answer": response}
    return response_answer

@app.route("/ask_pdf", methods=["POST"])
def ask_pdf_post():
    """
    Endpoint to handle questions related to PDF documents using RAG.
    """
    print("Post /ask_pdf called")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    print("Loading vector store")
    vector_store = Chroma(persist_directory=FOLDER_PATH, embedding_function=embedding)

    print("Creating chain")
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 20}
    )

    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)

    result = chain.invoke({"input": query})

    print(result)

    sources = [{"source": doc.metadata["source"], "page_content": doc.page_content} for doc in result["context"]]

    response_answer = {"answer": result["answer"], "sources": sources}
    return response_answer

@app.route("/pdf", methods=["POST"])
def pdf_post():
    """
    Endpoint to handle PDF uploads, process and store embeddings.
    """
    file = request.files["file"]
    file_name = file.filename
    save_file = f"pdf/{file_name}"
    file.save(save_file)
    print(f"filename: {file_name}")

    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    print(f"docs len={len(docs)}")

    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=FOLDER_PATH
    )

    vector_store.persist()

    response = {
        "status": "Successfully Uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks),
    }
    return response

def start_app():
    """
    Start the Flask application.
    """
    app.run(host="0.0.0.0", port=8080, debug=True)

if __name__ == "__main__":
    start_app()
