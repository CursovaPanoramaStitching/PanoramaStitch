import cv2
import numpy as np

# Зчитуємо дві фотографії
image1 = cv2.imread('project1\DJI_0714.JPG')
image2 = cv2.imread('project1\DJI_0717.JPG')

# Ініціалізуємо дескриптор SIFT
sift = cv2.SIFT_create()

# Знаходимо ключові точки та дескриптори для кожного зображення
kp1, des1 = sift.detectAndCompute(image1, None)
kp2, des2 = sift.detectAndCompute(image2, None)

# Знаходимо відповідності між дескрипторами
bf = cv2.BFMatcher()
matches = bf.knnMatch(des2, des1, k=1)

# Фільтруємо відповідності за допомогою методу RANSAC
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Отримуємо координати точок відповідностей
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Знаходимо матрицю перетворення (перспективне перетворення)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Зводимо зображення до однієї перспективи
result1 = cv2.warpPerspective(image1, M, (image1.shape[1] + image2.shape[1], image1.shape[0]))
result1[0:image2.shape[0], 0:image2.shape[1]] = image2

# Показуємо результат
cv2.imshow('Combined Image', result1)
cv2.waitKey(0)
cv2.destroyAllWindows()



