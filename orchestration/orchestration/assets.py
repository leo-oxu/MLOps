from dagster import asset
from pathlib import Path
from scripts import embed_data

@asset
def cleaned_data_path() -> Path:
  return Path("data/processed/cleaned_data.json")

@asset
def cleaned_data(cleaned_data_path: Path):
  return embed_data.load_cleaned_data(cleaned_data_path)

@asset
def faiss_output_path(cleaned_data):
  docs = embed_data.convert_to_documents(cleaned_data)
  outpath = embed_data.generate_save_embedding(docs)
  print(f"Saving FAISS DB to: {outpath}")
  return outpath




