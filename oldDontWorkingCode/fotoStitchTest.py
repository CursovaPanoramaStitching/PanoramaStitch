import cv2
import numpy as np

def stitch_images(images):
  """Склеює кілька зображень в панораму.

  Args:
    images: Список зображень, які потрібно склеїти.

  Returns:
    Панорамне зображення.
  """

  # Розрахуйте базову матрицю гомографії.
  H = cv2.estimateAffine2D(images[0], images[1], confidence=0.99, ransacReprojThreshold=10)

  # Склейте зображення за допомогою базової матриці гомографії.
  panorama = cv2.warpAffine(images[0], H, (images[0].shape[1] * len(images), images[0].shape[0]))

  # Додайте інші зображення до панорами.
  for image in images[1:]:
    panorama = cv2.addWeighted(panorama, 0.5, cv2.warpAffine(image, H, (panorama.shape[1], panorama.shape[0])), 0.5, 0)

  return panorama


images = [cv2.imread(f) for f in ["project1\DJI_0714.JPG", "project1\DJI_0717.JPG"]]

# Додайте центральну точку до набору точок.
center = (images[0].shape[1] // 2, images[0].shape[0] // 2)
images[0] = cv2.circle(images[0], center, 10, (0, 0, 255), -1)

panorama = stitch_images(images)

cv2.imshow("Panorama", panorama)
cv2.waitKey(0)