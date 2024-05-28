import numpy as np

# Función para aplicar el modelo de Hopfield
def hopfield_model(image):
    # Aplanar la imagen en una matriz unidimensional
    flattened_image = image.flatten()
    
    # Crear la matriz de pesos para el modelo de Hopfield
    weights = np.outer(flattened_image, flattened_image)
    
    # Iterar hasta que se alcance un estado estable
    while True:
        # Actualizar los valores de los nodos
        new_image = np.sign(np.dot(weights, flattened_image))
        
        # Comprobar si se ha alcanzado un estado estable
        if np.array_equal(flattened_image, new_image):
            break
        
        flattened_image = new_image
    
    # Volver a darle forma a la imagen original
    cleaned_image = np.reshape(flattened_image, image.shape)
    
    return cleaned_image

# Ejemplo de imagen con ruido
image_with_noise = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

# Aplicar el modelo de Hopfield para limpiar la imagen
cleaned_image = hopfield_model(image_with_noise)

# Mostrar la imagen original y la imagen limpia
print("Imagen original:")
print(image_with_noise)
print("\nImagen limpia:")
print(cleaned_image)
