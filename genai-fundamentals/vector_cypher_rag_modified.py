# Graph-Enhanced Vector Retriever

import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorCypherRetriever
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

# Define retrieval query
# The retrieval query is a Cypher query that will be used to get data from the graph after the nodes are returned by the vector search.
# The query receives the node and score variables yielded by the vector search.
# The query traverses the graph to find related nodes for genres and actors, as well as sorting the results by the user rating.
retrieval_query = """
MATCH (node)<-[r:RATED]-()
RETURN 
  node.title AS title, node.plot AS plot, score AS similarityScore, 
  collect { MATCH (node)-[:IN_GENRE]->(g) RETURN g.name } as genres, 
  collect { MATCH (node)<-[:ACTED_IN]->(a) RETURN a.name } as actors, 
  collect { MATCH (node)<-[:DIRECTED]-(d) RETURN DISTINCT d.name } AS directors, 
  avg(r.rating) as userRating
ORDER BY userRating DESC
"""

# Create retriever
# The VectorCypherRetriever allows you to perform vector searches and then traverse the graph to find related nodes or entities.
# The retriever requires the vector index name (moviePlots), the retrieval query, and the embedder to encode the query.
retriever = VectorCypherRetriever(
    driver,
    index_name="moviePlots",
    embedder=embedder,
    retrieval_query=retrieval_query,
)

#  Create the LLM
llm = OpenAILLM(model_name="gpt-4o")

# Create GraphRAG pipeline
rag = GraphRAG(retriever=retriever, llm=llm)

# Search
query_text = "Who has directed movies about weddings?"

response = rag.search(
    query_text=query_text, 
    retriever_config={"top_k": 5},
    return_context=True
)

print(response.answer)
print("CONTEXT:", response.retriever_result.items)
# The context is returned after the response, allowing you to see what data was used to generate the response.
# This transparency is important for understanding how the LLM arrived at its response and for debugging purposes

# Close the database connection
driver.close()