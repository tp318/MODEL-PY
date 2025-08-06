from typing import Dict

def print_search_results(results):
    """
    Print formatted semantic search results.

    Args:
        results (Dict): The result dictionary returned from semantic_search().
    """

    print("\nSearch Results:\n" + "-" * 50)

    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        meta = results['metadatas'][0][i]
        distance = results['distances'][0][i]

        print(f"\nResult {i + 1}")
        print(f"Source: {meta['source']}, Chunk {meta['chunk']}")
        print(f"Distance: {distance}")
        print(f"Content: {doc}\n")
        
print_search_results(results)