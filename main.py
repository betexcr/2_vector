import sqlite3
import json
import importlib.resources
import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(base_url=os.getenv("OLLAMA_URL"), api_key="ollama")
# create in-memory DB and allow it to load extensions
db = sqlite3.connect(":memory:")
db.enable_load_extension(True)

extension = importlib.resources.files("sqlite_vector.binaries") / "vector"

db.load_extension(str(extension))

db.execute("CREATE TABLE data (text TEXT, embedding BLOB)")


def embed_text(text):
    response = client.embeddings.create(
        model=os.getenv(
            "TEXT_EMBED_MODEL", ""
        ),  # or "text-embedding-3-small" if using OpenAI
        input=text,
    )
    # Extract the actual float array from the OpenAI/Ollama response object
    return response.data[0].embedding


texts = [
    "Python is awesome!",
    "SQLite is awesome!",
    "Oranges are awesome!",
    "Apples are great!",
    "I like Windows"
]

for text in texts:
    db.execute(
        "INSERT INTO data VALUES (?, vector_as_f32(?))",
        (text, json.dumps(embed_text(text))),
    )

# Initialize the embeddings as embedding vectors
db.execute(
    'SELECT vector_init("data", "embedding", "type=FLOAT32,dimension=768,distance=COSINE")'
)

query = "I like bananas!"

for row in db.execute(
    "SELECT data.text,v.distance FROM vector_full_scan('data', 'embedding', ?, 2) AS v JOIN data on data.rowid = v.rowid",
    (json.dumps(embed_text(query)),),
):
    print(row)
