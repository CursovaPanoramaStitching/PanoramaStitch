import os
import cv2
import numpy as np
import os
import shutil
from datetime import datetime
from PIL import Image

def process_images(source_dir, destination_dir):
    # Перевіряємо, чи існує директорія призначення, якщо ні - створюємо
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Копіюємо всі фото з вказаної директорії в photo/temp
    for filename in os.listdir(source_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            shutil.copy(os.path.join(source_dir, filename), os.path.join(destination_dir, filename))

    # Видалення фото висота зйомки менше 150 метрів
    for filename in os.listdir(destination_dir):
        file_path = os.path.join(destination_dir, filename)
        with Image.open(file_path) as img:
            exif_data = img._getexif()
            if 40963 in exif_data:  # EXIF tag for height above sea level
                height = exif_data[40963]
                if height is not None and height < 150:
                    os.remove(file_path)

    # Створення масиву файлів відсортованих за часом зйомки
    sorted_files = sorted(os.listdir(destination_dir), key=lambda x: os.path.getmtime(os.path.join(destination_dir, x)))

    # Переіменування файлів
    for i, filename in enumerate(sorted_files):
        timestamp = os.path.getmtime(os.path.join(destination_dir, filename))
        time_str = datetime.fromtimestamp(timestamp).strftime('%Y%m%d_%H%M%S')
        new_filename = f"{i}_{time_str}.jpg"
        os.rename(os.path.join(destination_dir, filename), os.path.join(destination_dir, new_filename))

    # Видалення метаданих з фото
    for filename in os.listdir(destination_dir):
        file_path = os.path.join(destination_dir, filename)
        with Image.open(file_path) as img:
            img.save(file_path)


# Параметри
source_directory = 'project\\source'
destination_directory = 'project\\temp'

# Виклик функції для обробки зображень
process_images(source_directory, destination_directory)

# Отримуємо список усіх зображень у директорії
image_dir = 'project\\temp'
image_files = os.listdir(image_dir)
image_files = [os.path.join(image_dir, f) for f in image_files if f.endswith(".jpg") or f.endswith(".JPG") or f.endswith(".JPEG") or f.endswith(".jpeg") or f.endswith(".png") or f.endswith(".PNG")]

# Зчитуємо перше зображення для використання як основи панорами
base_img = cv2.imread(image_files[0])
base_img_gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)

# Перший елемент масиву результатів - базове зображення
result = base_img

def warpImages(img1, img2, H):

  rows1, cols1 = img1.shape[:2]
  rows2, cols2 = img2.shape[:2]

  list_of_points_1 = np.float32([[0,0], [0, rows1],[cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
  temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)

  # When we have established a homography we need to warp perspective
  # Change field of view
  list_of_points_2 = cv2.perspectiveTransform(temp_points, H)

  list_of_points = np.concatenate((list_of_points_1,list_of_points_2), axis=0)

  [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
  [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
  
  translation_dist = [-x_min,-y_min]
  
  H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

  output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
  output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img1

  return output_img

# Проходимо по усіх наступних зображеннях та додаємо їх до панорами
for img_file in image_files[1:]:
    img = cv2.imread(img_file)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create our SIFT detector and detect keypoints and descriptors
    sift = cv2.SIFT_create(nfeatures=8000)
    
    # Find the key points and descriptors with SIFT
    keypoints1, descriptors1 = sift.detectAndCompute(result, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img_gray, None)
    
    # Create a BFMatcher object.
    # It will find all of the matching keypoints on two images
    bf = cv2.BFMatcher()
    
    # Find matching points
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    # Finding the best matches
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    
    # Set minimum match condition
    MIN_MATCH_COUNT = 100
    
    if len(good) > MIN_MATCH_COUNT:
        # Конвертуємо ключові точки в аргумент для findHomography
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    
        # Встановлюємо гомографію
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
        # Об'єднуємо зображення
        result = warpImages(result, img, M)

# Відображаємо та зберігаємо результат
cv2.imshow('result', result)
cv2.imwrite('panorama_result_SIFT_array.jpg', result)  # Зберігаємо результат
cv2.waitKey(0)