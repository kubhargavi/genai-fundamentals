# Vector Retriever
import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever

# Connect to Neo4j database
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(
        os.getenv("NEO4J_USERNAME"), 
        os.getenv("NEO4J_PASSWORD")
    )
)

# Create embedder
# embedder to convert users queries into vectors
# must use the same embedding model as the one used to create the movie plots embeddings, text-embedding-ada-002, to ensure the vectors are compatible.
embedder = OpenAIEmbeddings(model="text-embedding-ada-002")

# Create retriever
# retriever that uses the moviePlots vector index
# The retriever allows you to specify what properties to return from the nodes that match the query.
retriever = VectorRetriever(
    driver,
    index_name="moviePlots",
    embedder=embedder,
    return_properties=["title", "plot"],
)

# Search for similar items
# You can use the retriever to search the vector index by passing a query and the number of results to return.
# The retriever will use the embedder to convert the query into a vector to use in the search.
result = retriever.search(query_text="Toys coming alive", top_k=5)

# Parse results
# The search method returns a list of items that match the query.
# Iterate over the items and print the results
for item in result.items:
    print(item.content, item.metadata["score"])

# Close the database connection
driver.close()