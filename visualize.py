import numpy as np
import matplotlib.pyplot as plt

def visualize_distance():
    # Contoh vektor 2D untuk demonstrasi
    vec1 = np.array([1, 2])
    vec2 = np.array([4, 5])

    plt.figure(figsize=(8, 6))
    plt.scatter(vec1[0], vec1[1], color='red', label='Vektor 1')
    plt.scatter(vec2[0], vec2[1], color='blue', label='Vektor 2')
    
    # Gambar garis antara vektor
    plt.plot([vec1[0], vec2[0]], [vec1[1], vec2[1]], 'g--', label='Jarak Euclidean')
    
    plt.title('Visualisasi Jarak Euclidean')
    plt.xlabel('Dimensi 1')
    plt.ylabel('Dimensi 2')
    plt.legend()
    plt.grid(True)
    plt.savefig('euclidean_distance.png')
    plt.close()

visualize_distance()
