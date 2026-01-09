from fastapi import FastAPI
from llama_index.core import Settings
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
