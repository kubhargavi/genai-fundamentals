# RAG Pipeline

import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG

# Connect to Neo4j database
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(
        os.getenv("NEO4J_USERNAME"), 
        os.getenv("NEO4J_PASSWORD")
    )
)

# Create embedder
embedder = OpenAIEmbeddings(model="text-embedding-ada-002")

# Create retriever
retriever = VectorRetriever(
    driver,
    index_name="moviePlots",
    embedder=embedder,
    return_properties=["title", "plot"],
)

# Create the LLM
# You will need an LLM to generate the response based on the users query and the context provided by the vector retriever.
llm = OpenAILLM(model_name="gpt-4o")


# Create GraphRAG pipeline
# The GraphRAG class allows you to create a RAG pipeline including a retriever and an LLM.
# The graphrag pipline will use the retriever to find relevant context based on the user’s query.
# Pass the user’s query and the retrieve context to the LLM.
rag = GraphRAG(retriever=retriever, llm=llm)

# Search
# You can use the search method to submit a query.
# The search method takes the user’s query and returns the generated response from the LLM. You can also specify addition retriever_config, such as the number of results to return.
query_text = "Find me movies about toys coming alive"

response = rag.search(
    query_text=query_text, 
    retriever_config={"top_k": 5}
)

print(response.answer)
print("CONTEXT:", response.retriever_result.items)

# Close the database connection
driver.close()