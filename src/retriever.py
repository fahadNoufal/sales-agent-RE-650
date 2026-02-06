from src.custom_logger import logging

class RAGRetriever:
    def __init__(self, vector_store, embedding_manager):
        # Initializing the retriever
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5) -> str:
        
        logging.info(f"Received retrieval query: '{query}' with top_k={top_k}.")
        # Retrieves documents only if they meet a minimum similarity score.
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        
        results = self.vector_store.query(
            query_embeddings=[query_embedding.tolist()], 
            n_results=top_k
        )
        
        if not results['ids'] or not results['documents'][0]:
            return "No relevant documents found."

        relevant_chunks = []
        
        # similarity = 1 / (1 + distance)
        for i in range(len(results['documents'][0])):
            distance = results['distances'][0][i]
            similarity = 1 / (1 + distance) 
            
            content = results['documents'][0][i]
            source = results['metadatas'][0][i].get('policy_type', 'Unknown')
            relevant_chunks.append(f"[Source: {source} | Confidence: {similarity:.2f}]\n{content}")

        if not relevant_chunks:
            logging.info("No documents met the similarity threshold.")
            return "I'm sorry, I couldn't find any specific policy information related to your request."

        return "\n\n---\n\n".join(relevant_chunks)