from dotenv import load_dotenv
load_dotenv()
import pinecone
from pinecone import Pinecone, ServerlessSpec
import os

pc= Pinecone(api_key = os.environ.get("PINECONE_API_KEY"), environment=os.environ.get("PINECONE_ENV"))
print(pc.list_indexes())
index_name= "my-index"
dimension = 3
metric = "cosine"

if index_name in [index.name for index in pc.list_indexes()]:
    pc.delete_index(index_name)
    print(f"{index_name} Successfully deleted")
else:
    print(f"{index_name} not in index list")

print(pc.list_indexes())
pc.create_index(
    name = index_name,
    dimension= dimension,
    metric=metric,
    spec= ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)
print(pc.list_indexes())