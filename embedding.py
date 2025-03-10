import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

class RAGSystem:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        self.vector_dim = 384  
        self.embeddings = []  # Menyimpan embedding dokumen
        self.texts = []  # Menyimpan teks asli

    def embed_text(self, text):
        # Fungsi untuk mengubah teks menjadi embedding
        tokens = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = self.model(**tokens)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.numpy()[0]

    def add_to_database(self, text):
        # Menambahkan teks ke database
        embedding = self.embed_text(text)
        self.embeddings.append(embedding)
        self.texts.append(text)

    def semantic_search(self, query, top_k=None):
        # Melakukan pencarian berdasarkan cosine similarity
        query_embedding = self.embed_text(query)
        
        if not self.embeddings:
            return [], []
        
        # Menghitung cosine similarity
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Jika top_k tidak ditentukan, kembalikan semua hasil
        if top_k is None:
            top_k = len(self.embeddings)
            
        # Urutkan berdasarkan skor tertinggi
        sorted_indices = np.argsort(similarities)[::-1][:top_k]
        retrieved_texts = [self.texts[i] for i in sorted_indices]
        retrieved_scores = [similarities[i] for i in sorted_indices]
        
        return retrieved_texts, retrieved_scores

    def visualize_vector(self, text, filename='vector_visualization.png'):
        # Fungsi untuk visualisasi vektor
        embedding = self.embed_text(text)
        
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        sns.heatmap(embedding.reshape(1, -1), cmap='viridis', 
                    annot=False, cbar_kws={'label': 'Embedding Value'})
        plt.title(f'Vector Heatmap untuk "{text}"')
        plt.xlabel('Vector Dimensions')
        plt.ylabel('Vector')
        
        plt.subplot(1, 2, 2)
        plt.plot(embedding, marker='o')
        plt.title(f'Vector Line Plot untuk "{text}"')
        plt.xlabel('Dimension')
        plt.ylabel('Value')
        plt.tight_layout()
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualisasi vektor untuk '{text}'")
        print(f"Rentang Nilai: [{embedding.min()}, ... , {embedding.max()}]")

def main():
    rag_system = RAGSystem()

    # Menambahkan contoh dokumen ke database
    dokumen_database = [
        "anime adalah film animasi dari jepang",
        "aku suka anime",
        "penggemar anime sering disebut wibu"
    ]

    for dok in dokumen_database:
        rag_system.add_to_database(dok)

    query = "siapa suka anime"

    print("\n--- Visualisasi Vektor Dokumen Database ---")
    for dok in dokumen_database:
        rag_system.visualize_vector(dok)
        print("\n")

    print("--- Visualisasi Vektor Query ---")
    rag_system.visualize_vector(query)

    print("\n--- Semantic Search (Semua Dokumen) ---")
    hasil_search, skor = rag_system.semantic_search(query, top_k=None)
    for i in range(len(hasil_search)):
        print(f"Dokumen {i+1}: {hasil_search[i]} (cosine similarity: {skor[i]:.4f})")

if __name__ == "__main__":
    main() 
