# CONVERT - WORDS TO VECTORS
# NLP (Bag of words(freq based), TF-IDF(overall freq in all docs))
# NERUAL NETWORK - Word to vector (google) - predict a word from given context and vice versa 
# TRANSFORMER & LLM- BERT(embedding change on context), ELMO(Embedding from Language Model) - LSTM(Long Short term memory)
import pandas as pd
from dotenv import load_dotenv, find_dotenv
load_dotenv()
from pinecone import Pinecone, ServerlessSpec
import os
from sentence_transformers import SentenceTransformer
load_dotenv(find_dotenv(), override=True)
pc= Pinecone(api_key = os.environ.get("PINECONE_API_KEY"), environment = os.environ.get('PINECONE_ENV'))

files = pd.read_csv(r"D:\PythonFolder\nlp_foundation\course_descriptions.csv", encoding="ANSI")

def create_course_description(row):
    return f'''The course name is the {row["course_name"]}, the slug is  {row["course_slug"]}, the technology is  {row["course_technology"]} and the course is  {row["course_topic"]}'''
files['course_description_new'] = files.apply(create_course_description, axis=1)

index_name = "my-index"
dimension=384
metric="cosine"

if index_name in [index.name for index in pc.list_indexes()]:
    pc.delete_index(index_name)
    print(f"{index_name} successfully deleted")
else:
    print(f"{index_name} not in index list")

pc.create_index(
    name=index_name,
    dimension=dimension,
    metric="cosine",
    spec=ServerlessSpec(
        cloud='aws',
        region='us-east-1'
    )
)
index= pc.Index(index_name)

model = SentenceTransformer("all-MiniLM-L6-v2")
def create_embeddings(row):
    combined_text = ' '.join([str(row[field]) for field in ["course_description","course_description_new","course_description_short"]])
    embedding = model.encode(combined_text, show_progress_bar=False)
    return embedding
files["embedding"] = files.apply(create_embeddings, axis=1)
vectors_to_upsert  = [(str(row["course_name"]),row["embedding"].tolist()) for _, row in files.iterrows()]
index.upsert(vectors =vectors_to_upsert)
print("Data upserted to pinecone index")

query = "clustering" 
query_embedding = model.encode(query, show_progress_bar=False).tolist()
query_results = index.query(
    vector = [query_embedding],
    top_k=12,
    include_values = True
)

print(query_results)

for match in query_results["matches"]:
    print(f"Matched item ID: {match['id']}, score: {match['score']}")

# from dotenv import load_dotenv, find_dotenv
# load_dotenv()
# from datasets import load_dataset
# import pinecone
# from pinecone import Pinecone, ServerlessSpec
# import os
# from sentence_transformers import SentenceTransformer

# fw= load_dataset("HuggingFaceFW/fineweb", name = "sample-10BT", split="train", streaming=True)
# print(fw)
# print(fw.features)
# model = SentenceTransformer("all=MiniLM-L6-v2") #general embedding algo
# load_dotenv(find_dotenv(), override=True)
# pc= Pinecone(api_key = os.environ.get("PINECONE_API_KEY"), environment = os.environ.get('PINECONE_ENV'))
# pc.create_index(
#     name="text",
#     dimension=model.get_sentence_embedding_diemnsion(),
#     metric="cosine",
#     spec=ServerlessSpec(
#         cloud='aws',
#         region='us-east-1'
#     )
# )

# index = pc.Index(name="text")
# #define number of items to process
# subset_size = 10000
# #iterate over dataset and prepare data for upserting 
# vectors_to_upsert = []
# for i, item in enumerate(fw):
#     if i >= subset_size:
#         break
#     text = item['text']
#     unique_id = str(item['id'])
#     language = item['language']

#     #create an embedding for the text
#     embedding = model.encode(text, show_progress_bar=False).tolist()
#     #prepare metadaa
#     metadata={'language':language}
#     #append the tuple (id, embedding , metadata) to the list
#     vectors_to_upsert.append((unique_id, embedding, metadata))

# #Upsert data to pinecone in batches
# batch_size = 1000
# for i in range(0, len(vectors_to_upsert), batch_size):
#     batch = vectors_to_upsert[i:i+batch_size]
#     index.upsert(vectors=batch)
# print("Subset of data upserted pinecone index")



# from dotenv import load_dotenv
# load_dotenv()
# import pinecone
# from pinecone import Pinecone, ServerlessSpec
# import os

# pc= Pinecone(api_key = os.environ.get("PINECONE_API_KEY"), environment=os.environ.get("PINECONE_ENV"))
# print(pc.list_indexes())
# index_name= "my-index"
# dimension = 3
# metric = "cosine"

# if index_name in [index.name for index in pc.list_indexes()]:
#     pc.delete_index(index_name)
#     print(f"{index_name} Successfully deleted")
# else:
#     print(f"{index_name} not in index list")

# print(pc.list_indexes())
# pc.create_index(
#     name = index_name,
#     dimension= dimension,
#     metric=metric,
#     spec= ServerlessSpec(
#         cloud="aws",
#         region="us-east-1"
#     )
# )
# print(pc.list_indexes())

# #UPSERTING- Updating and inserting data (more dimensions-refined description)
# index = pc.Index(name=index_name)
# index.upsert([
#     ("Dog",[4.,0.,1.]),
#     ("Cat",[4.,0.,1.]),
#     ("Chicken",[2.,2.,1.]),
#     ("Mantis",[6.,2.,3.]),
#     ("Elephant",[4.,0.,1.])
# ])