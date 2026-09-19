import chromadb

# Path alla cartella locale del DB vettoriale
CHROMA_PATH = "./chroma_db_iva"

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name="iva_documents")

print(f"Totale chunk indicizzati: {collection.count()}")

# Query di prova
test_query = "Qual è l'aliquota IVA per le prestazioni alberghiere?"
# test_query ="Which is the VAT rate for hotel services?"

results = collection.query(
    query_texts=[test_query],
    n_results=3
)

print("\n--- RISULTATI RETRIEVAL ---")
for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
    print(f"\n[Chunk {i+1}] Fonte: {meta.get('source', 'N/A')} - Sezione: {meta.get('section', 'N/A')}")
    print(f"Testo: {doc[:300]}...")