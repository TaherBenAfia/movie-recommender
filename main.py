import pymongo
from fastapi import Request, FastAPI
import requests
from sentence_transformers import SentenceTransformer   
app = FastAPI() 
MONGO_URI = "mongodb+srv://taherbenafia:taherbenafia123@cluster0.uoj3jnq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = pymongo.MongoClient(MONGO_URI)
db = client.sample_mflix
collection = db.movies



model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def generate_embedding(text: str) -> list[float]:
    embedding = model.encode(text)
    return embedding.tolist()

# for doc in collection.find({'plot':{"$exists": True}}):
#    doc['plot_embedding_hf'] = generate_embedding(doc['plot'])
#    collection.replace_one({'_id': doc['_id']}, doc)

query = "humans with superpowers" 
@app.post("/recommend")
async def recommend(req: Request):
    data = await req.json()
    query = data["query"]
    query_vector = model.encode(query).tolist()
    results = collection.aggregate([
        {
        "$vectorSearch": {
            "queryVector": generate_embedding(query),
            "path": "plot_embedding_hf",
            "numCandidates": 100,
            "limit": 4,
            "index": "PlotSemanticSearch"
        }
        }
        ])
    result = next(results, None)
    return {"result": result}
