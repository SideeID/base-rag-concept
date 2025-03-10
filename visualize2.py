import numpy as np
import matplotlib.pyplot as plt

def visualize_vector_difference(v1, v2):
    diff = v1 - v2
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title("Vektor 1")
    plt.plot(v1)
    
    plt.subplot(1, 3, 2)
    plt.title("Vektor 2")
    plt.plot(v2)
    
    plt.subplot(1, 3, 3)
    plt.title("Perbedaan Vektor")
    plt.plot(diff)
    
    plt.tight_layout()
    plt.show()
