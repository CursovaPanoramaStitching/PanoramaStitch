import psutil
import cv2
import numpy as np
import os
import shutil
import exifread
import re

from datetime import datetime
from PIL import Image

import exifread
from fractions import Fraction

import logging
import threading

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
        with open(file_path, 'rb') as f:
            tags = exifread.process_file(f, details=False, stop_tag='GPS GPSAltitude')
            if 'GPS GPSAltitude' in tags:
                altitude_fraction = tags['GPS GPSAltitude'].values[0]
                altitude = float(Fraction(altitude_fraction))  # Перетворюємо дріб у число
                if altitude is not None and altitude < 150:
                    os.remove(file_path)

    # Створення масиву файлів відсортованих за часом створення
    sorted_files = sorted(os.listdir(destination_dir), key=lambda x: datetime.strptime(Image.open(os.path.join(destination_dir, x))._getexif()[306], '%Y:%m:%d %H:%M:%S').timestamp())

    # Переіменування файлів
    for i, filename in enumerate(sorted_files):
        timestamp = Image.open(os.path.join(destination_dir, filename))._getexif()[306]
        time_str = datetime.strptime(timestamp, '%Y:%m:%d %H:%M:%S').strftime('%Y%m%d_%H%M%S')
        new_filename = f"{i}_{time_str}.jpg"
        os.rename(os.path.join(destination_dir, filename), os.path.join(destination_dir, new_filename))

    # Видалення метаданих з фото
    for filename in os.listdir(destination_dir):
        file_path = os.path.join(destination_dir, filename)
        with Image.open(file_path) as img:
            img.save(file_path)

def stitch_image_in_sub_directory(input_directory, output_directory):

    # Add logging configuration
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    try:
        if os.path.isdir(input_directory):
            # Виклик функції для обробки зображень
            process_images(input_directory, os.path.join(output_directory, 'temp'))

            # Тепер проходимо по обробленим зображенням у temp директорії
            for dirpath, _, filenames in os.walk(os.path.join(output_directory, 'temp')):
                filenames = sorted(filenames, key=lambda x: int(x.split('_')[0]))
                print('Stitching is starting...')
                print('---------------------------------')

                # Розбиття списку filenames на 10 частин
                num_parts = 15
                filenames_parts = np.array_split(filenames, num_parts)

                # Створення ThreadPool з 10 потоками
                from concurrent.futures import ThreadPoolExecutor
                pool = ThreadPoolExecutor(max_workers=4)

                def stitch_image_part(filenames, output_directory, group_number):
                    try:
                        logger.info('Поток %s: Початок склеювання %s', threading.get_ident(), filenames)
                        # Створіть зображення-колаж з першого зображення
                        image_collage = cv2.imread(os.path.join(output_directory, 'temp', filenames[0]))

                        # Пройдіть по решті зображень у частині
                        for filename in filenames[1:]:
                            # Завантажте наступне зображення
                            main_image = cv2.imread(os.path.join(output_directory, 'temp', filename))

                            # Склейте два зображення
                            image_collage = first_step(image_collage, main_image)

                        # Збережіть зображення-колаж
                        save_image(output_directory, f'temp\\temp_image_stitched_{group_number:04d}.jpg', image_collage)
                        
                        logger.info('Поток %s: Завершено склеювання %s', threading.get_ident(), filenames)

                    except Exception as e:
                        print(f"An error occurred: {e}")

                def stitch_image_pair(filenames, output_directory):
                    try:
                        logger.info('Поток %s: Початок склеювання пари', threading.get_ident())

                        filename1, filename2 = filenames[:2]

                        # Отримання чисел з імен файлів
                        id1 = re.findall(r'\d{4}', filename1)[0]
                        id2 = re.findall(r'\d{4}', filename2)[0]

                        # Склейте два зображення
                        image_collage = first_step(cv2.imread(os.path.join(output_directory, 'temp', filename1)),
                                                cv2.imread(os.path.join(output_directory, 'temp', filename2)))

                        # Збережіть зображення-колаж
                        save_image(output_directory, f'temp\\temp_image_stitched_{id1}_{id2}.jpg', image_collage)

                        new_filenames.append(f'temp_image_stitched_{id1}_{id2}.jpg')

                        logger.info('Поток %s: Завершено склеювання пари', threading.get_ident())

                    except Exception as e:
                        print(f"An error occurred: {e}")


                # Додавання завдань до пулу для кожної частини filenames_parts
                for i, filenames_part in enumerate(filenames_parts):
                    future = pool.submit(stitch_image_part, filenames_part, output_directory, i)

                # Зачекайте на завершення всіх завдань
                pool.shutdown(wait=True)

                # Отримайте список файлів
                filenames = os.listdir(os.path.join(output_directory, 'temp'))

                # Відфільтруйте файли, які відповідають шаблону
                new_filenames = [filename for filename in filenames if filename.startswith('temp_image_stitched_')]

                # Відсортуйте filenames за XXXX
                new_filenames = sorted(new_filenames, key=lambda x: int(x.split('.')[0][-4:]))
                print('---------------------------------')
                print('First sort')

                while len(new_filenames) > 1:

                    # Створіть ThreadPoolExecutor з 3 потоками
                    pool2 = ThreadPoolExecutor(max_workers=3)

                    # Сортування перед склеюванням наступного ряду
                    new_filenames = sorted(new_filenames, key=lambda x: int(x.split('.')[0][-4:]))
                    print('Sort happen')
                    print('Sorted array')
                    print(new_filenames)

                    while len(new_filenames) >= 2:
                        # Візьміть перші два імені файлів з черги
                        filename1, filename2 = new_filenames[:2]

                        print('---------------------------------')
                        print(f'Stitch thread create {filename1}, {filename2} go stitch')
                        # Додайте завдання до списку
                        pool2.submit(stitch_image_pair, [filename1, filename2], output_directory)

                        # Видаліть оброблені імена з черги
                        new_filenames = new_filenames[2:]
                        print(f'Delete {filename1}, {filename2} from array')
                        print()
                        print('---------------------------------')

                    # Зачекайте на завершення всіх завдань
                    pool2.shutdown(wait=True)
                    print('---------------------------------')

                # Збережіть остаточне зображення
                if len(new_filenames) == 1:
                    print('---------------------------------')
                    print('Finish')
                    final_image = cv2.imread(os.path.join(output_directory, 'temp', filenames[0]))
                    save_image(output_directory, 'final_image_stitched.jpg', final_image)

    except Exception as e:
        print(f"An error occurred: {e}")



def save_image(directory, file_name, image):
    #     check directory is exist and create if not exit
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(directory+'\\'+file_name, image)

def warp_images(img1, img2, h):
    # Отримання розмірів зображень
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    # Визначення точок для перспективного перетворення
    list_of_points_1 = np.array([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]], dtype=np.float32)
    temp_points = np.array([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]], dtype=np.float32)

    # Перетворення точок перспективного перетворення для другого зображення
    list_of_points_2 = cv2.perspectiveTransform(temp_points.reshape(-1, 1, 2), h).reshape(-1, 2)

    # Об'єднання точок зображень
    list_of_points = np.vstack((list_of_points_1, list_of_points_2))

    # Визначення меж об'єднаної області
    min_x, min_y = np.int32(list_of_points.min(axis=0))
    max_x, max_y = np.int32(list_of_points.max(axis=0))

    # Визначення зсуву для першого зображення
    translation_dist = [-min_x, -min_y]

    # Розрахунок розміру вихідного зображення
    output_width = max(max_x - min_x, cols1)
    output_height = max(max_y - min_y, rows1)

    # Матриця трансляції
    h_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    # Виконання перспективного перетворення для другого зображення
    output_img = cv2.warpPerspective(img2, np.dot(h_translation, h), (output_width, output_height))

    # Копіювання першого зображення на вихідне зображення з урахуванням зсуву
    output_img[translation_dist[1]:rows1 + translation_dist[1], translation_dist[0]:cols1 + translation_dist[0]] = img1

    return output_img

def print_memory_info():
    memory = psutil.virtual_memory()
    print("Memory available is {:.2f}GB ({:.2f}%)".format(memory.available / (1024.0 ** 3), memory.available * 100 / memory.total))

def preprocess_image(image):
  # Перетворення на чорно-білий
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  gray_image = gray_image.astype("uint8")

  # Фільтр Канні
  # edges_image = cv2.Canny(gray_image, 100, 200)

  # Зменшення розміру
  # resized_image = cv2.resize(edges_image, (0, 0), fx=1, fy=1)

  # Видалення чорного кольору
  thresh = 254
  black_mask = cv2.threshold(gray_image, thresh, 255, cv2.THRESH_BINARY_INV)[1]
  gray_image = cv2.bitwise_and(gray_image, black_mask)

  # Гістограма вирівнювання: Цей метод покращує контрастність зображення, що може допомогти знайти більше деталей.
  gray_image = cv2.equalizeHist(gray_image)
 
  kernel_size = (5, 5)
  sigma = 1.0
  gray_image = cv2.GaussianBlur(gray_image, kernel_size, sigma)

  # cv2.imshow('оброблене фото', gray_image)
  # cv2.waitKey(0)

  return gray_image

def first_step(img1, img2):
    if img1 is None or img2 is None:
        print("One or both images are not loaded correctly.")
        return None
    # Create our ORB detector and detect keypoints and descriptors
    sift = cv2.SIFT_create(nfeatures=4000)

    # img2 = match_contrast(img1, img2)

    # TODO пофіксити цей момент, покищо так далі будем бачити
    # Видалення чорного кольору
    thresh = 254
    black_mask = cv2.threshold(img1, thresh, 255, cv2.THRESH_BINARY_INV)[1]
    img1 = cv2.bitwise_and(img1, black_mask)

    black_mask = cv2.threshold(img2, thresh, 255, cv2.THRESH_BINARY_INV)[1]
    img2 = cv2.bitwise_and(img2, black_mask)

    # Попередня обробка
    processed_img1 = preprocess_image(img1)
    processed_img2 = preprocess_image(img2)

    # Find the key points and descriptors with ORB
    keypoints1, descriptors1 = sift.detectAndCompute(processed_img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(processed_img2, None)

    # Create a BFMatcher object.
    # It will find all of the matching keypoints on two images
    bf = cv2.BFMatcher()

    # Find matching points
    matches = bf.knnMatch(descriptors1, descriptors2,k=2)

    # Finding the best matches
    good = []
    for m, n in matches:
        if m.distance < 0.78 * n.distance:
            good.append(m)

    # Set minimum match condition
    MIN_MATCH_COUNT = 50

    if len(good) >= MIN_MATCH_COUNT:
        # Convert keypoints to an argument for findHomography
        src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        # Establish a homography
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        result = warp_images(img2, img1, M)
        #cv2.imshow('result', result)
        #cv2.waitKey(0)
        return result

    else: return None


input_directory = 'project\\source'
output_directory = 'output'
stitch_image_in_sub_directory(input_directory, output_directory)
