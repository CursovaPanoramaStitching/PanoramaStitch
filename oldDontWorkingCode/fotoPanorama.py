import cv2
import numpy as np
import os
import threading

# Function to load images from a directory
def load_images(directory):
  images = []
  for filename in os.listdir(directory):
    if filename.endswith(('.JPG', '.png')):  # Filter image formats
      image_path = os.path.join(directory, filename)
      image = cv2.imread(image_path)
      images.append(image)
  return images

# Функція для пошуку збігів між парою зображень
def find_matches_for_pair(images, i, j):
    """
    Знаходить збіги між парою зображень.

    Аргументи:
        images: Список зображень.
        i: Індекс першого зображення.
        j: Індекс другого зображення.

    Повертає:
        Пара (збіг, індекс першого зображення, індекс другого зображення).
    """

    match = cv2.matchTemplate(images[i], images[j], cv2.TM_SQDIFF_NORMED)
    return match

# Функція для паралельного пошуку збігів
def find_matches_parallel(images):
    """
    Знаходить збіги між всіма парами зображень паралельно.

    Аргументи:
        images: Список зображень.

    Повертає:
        Список пар (збіг, індекс першого зображення, індекс другого зображення).
    """

    threads = []
    matches = []

    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            thread = threading.Thread(target=find_matches_for_pair, args=(images, i, j))
            threads.append(thread)
            thread.start()

    for thread in threads:
        while thread.is_alive():
            pass

    return matches

# Функція для зшивання зображень
def stitch_images(matches, images):
    """
    Зшиває зображення, використовуючи результати пошуку збігів.

    Аргументи:
        matches: Список пар (збіг, індекс першого зображення, індекс другого зображення).
        images: Список зображень.

    Повертає:
        Склейоване зображення.
    """

    panorama = np.zeros_like(images[0])
    for match, i, j in matches:
        x0, y0, x1, y1 = match.shape
        panorama[y0:y1, x0:x1] = images[j]
    return panorama

# Головний код
images_directory = "D:\Programing\Cursova\project1"  # Specify the directory containing images
images = load_images(images_directory)

if images:
    matches = find_matches_parallel(images)
    panorama = stitch_images(matches, images)
    cv2.imwrite("panorama.jpg", panorama)
    print("Panorama created successfully!")
else:
    print("Error: No images found in the specified directory.")