import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import os
from datetime import datetime

class RAGSystem:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", output_dir="visualizations"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        self.vector_dim = 384  
        self.embeddings = []  # Menyimpan embedding dokumen
        self.texts = []  # Menyimpan teks asli
        
        # Membuat direktori output jika belum ada
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

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
        
        return retrieved_texts, retrieved_scores, query_embedding

    def visualize_vector(self, text, filename=None):
        # Fungsi untuk visualisasi vektor
        embedding = self.embed_text(text)
        
        # Buat nama file otomatis jika tidak disediakan
        if filename is None:
            # Gunakan timestamp untuk mencegah tumpang tindih file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Buat nama file yang aman (tanpa karakter khusus)
            safe_text = ''.join(c if c.isalnum() else '_' for c in text[:20])
            filename = os.path.join(self.output_dir, f"vector_{safe_text}_{timestamp}.png")
        else:
            filename = os.path.join(self.output_dir, filename)
        
        plt.figure(figsize=(15, 6))
        
        # Heatmap visualization
        plt.subplot(1, 2, 1)
        sns.heatmap(embedding.reshape(1, -1), cmap='viridis', 
                    annot=False, cbar_kws={'label': 'Embedding Value'})
        plt.title(f'Vector Heatmap untuk "{text[:30]}..."' if len(text) > 30 else f'Vector Heatmap untuk "{text}"')
        plt.xlabel('Vector Dimensions')
        plt.ylabel('Vector')
        
        # Line plot visualization
        plt.subplot(1, 2, 2)
        plt.plot(embedding, marker='.', linestyle='-', alpha=0.7)
        plt.title(f'Vector Line Plot untuk "{text[:30]}..."' if len(text) > 30 else f'Vector Line Plot untuk "{text}"')
        plt.xlabel('Dimension')
        plt.ylabel('Value')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Simpan gambar dengan kualitas tinggi
        plt.savefig(filename, dpi=300, bbox_inches='tight', format='png')
        plt.close()
        
        print(f"Visualisasi vektor untuk '{text[:50]}...' disimpan di '{filename}'")
        print(f"Rentang Nilai: [{embedding.min():.4f}, ... , {embedding.max():.4f}]")
        
        return filename, embedding

    def visualize_comparison(self, query, results, scores, query_embedding, filename=None):
        """Visualisasi perbandingan query dengan hasil pencarian"""
        if not results:
            print("Tidak ada hasil untuk divisualisasikan.")
            return None
            
        # Buat nama file otomatis jika tidak disediakan
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"comparison_{timestamp}.png")
        else:
            filename = os.path.join(self.output_dir, filename)
            
        # Jumlah hasil yang akan ditampilkan
        n_results = min(3, len(results))
        
        plt.figure(figsize=(15, 4 * (n_results + 1)))
        
        # Plot query vector
        plt.subplot(n_results + 1, 1, 1)
        plt.plot(query_embedding, color='blue', marker='.', label='Query')
        plt.title(f'Query: "{query[:50]}..."' if len(query) > 50 else f'Query: "{query}"')
        plt.xlabel('Dimension')
        plt.ylabel('Value')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Plot top results
        for i in range(n_results):
            result_text = results[i]
            result_score = scores[i]
            result_embedding = self.embed_text(result_text)
            
            plt.subplot(n_results + 1, 1, i + 2)
            plt.plot(result_embedding, color='green', marker='.', label=f'Result {i+1}')
            # Overlay query vector dengan transparansi untuk perbandingan
            plt.plot(query_embedding, color='blue', alpha=0.3, linestyle='--')
            plt.title(f'Result {i+1}: "{result_text[:50]}..." (Similarity: {result_score:.4f})')
            plt.xlabel('Dimension')
            plt.ylabel('Value')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight', format='png')
        plt.close()
        
        print(f"Visualisasi perbandingan disimpan di '{filename}'")
        return filename

def main():
    rag_system = RAGSystem()

    # Menambahkan contoh dokumen ke database
    dokumen_database = [
        "anime adalah film animasi dari jepang",
        "aku suka anime",
        "penggemar anime sering disebut wibu",
        "manga adalah komik dari jepang",
        "cosplay adalah kegiatan mengenakan kostum karakter"
    ]

    for dok in dokumen_database:
        rag_system.add_to_database(dok)

    query = "siapa suka anime"

    print("\n--- Visualisasi Vektor Dokumen Database ---")
    for i, dok in enumerate(dokumen_database):
        filename = f"dokumen_{i+1}.png"
        rag_system.visualize_vector(dok, filename)
        print("\n")

    print("--- Visualisasi Vektor Query ---")
    query_file, _ = rag_system.visualize_vector(query, "query.png")

    print("\n--- Semantic Search (Semua Dokumen) ---")
    hasil_search, skor, query_embedding = rag_system.semantic_search(query, top_k=None)
    for i in range(len(hasil_search)):
        print(f"Dokumen {i+1}: {hasil_search[i]} (cosine similarity: {skor[i]:.4f})")
    
    # Buat visualisasi perbandingan
    print("\n--- Membuat Visualisasi Perbandingan ---")
    rag_system.visualize_comparison(query, hasil_search, skor, query_embedding, "perbandingan_query_hasil.png")

if __name__ == "__main__":
    main()