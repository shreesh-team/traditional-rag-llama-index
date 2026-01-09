from fastapi import FastAPI
from llama_index.core import Settings, Document, VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter


app = FastAPI()


# Configure llm
Settings.llm = Ollama(model="gemma3:12b", temperature=0.1, request_timeout=300)

# Configure Embedding
Settings.embed_model = OllamaEmbedding(model_name="embeddinggemma", dimensions=512)

# Configure Text Chunking
Settings.chunk_size = 1024
Settings.chunk_overlap = 200

# Configure Node Parser
Settings.node_parser = SentenceSplitter(
    chunk_size=Settings.chunk_size, chunk_overlap=Settings.chunk_overlap
)

# Create sample documents (in practice, load from files)
documents = [
    Document(
        text="""
        LlamaIndex is a data framework for large language models (LLMs). 
        It provides tools to ingest, structure, and access private or domain-specific data.
        LlamaIndex was created to solve the problem of connecting LLMs to external data sources.
        The framework supports various data sources including PDFs, databases, APIs, and web pages.
        """,
        metadata={"source": "intro", "category": "overview"},
    ),
    Document(
        text="""
        Vector embeddings are numerical representations of text that capture semantic meaning.
        In LlamaIndex, embeddings enable semantic search - finding relevant content based on meaning,
        not just keyword matching. The default embedding model is OpenAI's text-embedding-3-small,
        which produces 1536-dimensional vectors. Other models like all-MiniLM-L6-v2 produce 384 dimensions.
        """,
        metadata={"source": "embeddings", "category": "technical"},
    ),
    Document(
        text="""
        The VectorStoreIndex is the most common index type in LlamaIndex. It stores document embeddings
        in a vector database and performs similarity search during queries. When you query the index,
        it retrieves the most semantically similar chunks and passes them to the LLM as context.
        This is the foundation of Retrieval-Augmented Generation (RAG).
        """,
        metadata={"source": "vector_index", "category": "technical"},
    ),
]


class BasicVectorStore:
    def __init__(self, documents):
        self.documents = []
        self.index = None
        if len(documents) > 0:
            self.index = VectorStoreIndex.from_documents(
                documents=documents,
                show_progress=True,
            )

        self.query_engine = None

    def get_vector_store_index(self, document):
        return self.index

    def check_doc_count_in_index(self):
        return len(self.index.docstore.docs)

    def set_query_engine(self):
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=2, response_mode="compact"
        )


vector_store = BasicVectorStore(documents)
vector_store.set_query_engine()


@app.get("/")
async def root():
    return {
        "Configuration": {
            "llm": Settings.llm,
            "embed_model": Settings.embed_model,
            "chunk_size": Settings.chunk_size,
            "chunk_overlap": Settings.chunk_overlap,
            "node_parser": Settings.node_parser,
        }
    }


@app.get("/llm-health")
async def llm_health():
    response = Settings.llm.complete("Hello, are you healthy?")
    print(f"LLM Check: {response}")
    return response.text


@app.get("/rag")
async def rag(query: str):
    response = vector_store.query_engine.query(query)
    return response.response
