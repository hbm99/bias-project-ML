import os
import cv2
import imgaug.augmenters as iaa

# Función para aumentar una imagen
def augment_image(image_name, input_path, output_path, count_img):
    # Crear el objeto de aumento de datos
    augmenter = iaa.Sequential([
        iaa.Fliplr(0.5),  # Reflejo horizontal
        iaa.Affine(rotate=(-10, 10)),  # Rotación aleatoria
        iaa.GaussianBlur(sigma=(0, 1.0)),  # Desenfoque gaussiano
        iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255))])  # Ruido gaussiano
    
    input_path = os.path.join(input_path, image_name)
    image = cv2.imread(input_path)
    augmented_image = augmenter.augment_image(image)
    name = f"{image_name}{count_img}.jpg"
    # Guardar la imagen aumentadas
    output_path = os.path.join(output_path, name)
    cv2.imwrite(output_path, augmented_image)
    return name
