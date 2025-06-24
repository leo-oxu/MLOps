import json
from pathlib import Path
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

def load_cleaned_data(path: Path):
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)

# Create a unified textual representation for embedding
def to_text(entry):
  return (
    f"{entry['name']}, {entry['country']}. "
    f"Tags: {', '.join(entry['tags'])}. "
    f"Best seasons: {', '.join(entry['season'])}. "
    f"Climate: {entry['climate']}, Budget: {entry['budget_level']}. "
    f"Highlights: {', '.join(entry['highlights'])}."
  )

def convert_to_documents(data):
  return [Document(page_content=to_text(entry), metadata=entry) for entry in data]

# Use sentence-transformers model (CPU-friendly)
def generate_save_embedding(docs, output_path=Path("data/vectorstore"), model_name="all-MiniLM-L6-v2"):
  embedding_model = HuggingFaceEmbeddings(model_name=model_name)
  db = FAISS.from_documents(docs, embedding_model) # Embed and store in FAISS
  output_path.mkdir(parents=True, exist_ok=True)
  db.save_local(str(output_path))
  return str(output_path)

if __name__ == "__main__":
  INPUT = Path("data/processed/cleaned_data.json")
  OUTPUT = Path("data/vectorstore")
  data = load_cleaned_data(INPUT)
  docs = convert_to_documents(data)
  outpath = generate_save_embedding(docs, OUTPUT)
  print(outpath)