import cv2
import numpy as np
from scipy.interpolate import CubicSpline, interp1d

# Вхідна панорама
panorama = cv2.imread('result\\final_image_stitched.jpg')

# Контрольні точки (гео + пікселі)
points = np.array([[48.54334883179478, 35.0948654426536, 6641, 9270],
                   [48.542041777503826, 35.09759793192003, 8518, 7017],
                   [48.54071025778341, 35.09941698099394, 10415, 5526],
                   [48.53931743764255, 35.10125316177766, 12402, 4040],
                   [48.54097494166265, 35.10354552977222, 10335, 1848]])

# Інтерполяція
num_interpolated_points = 20  # Кількість інтерпольованих точок між контрольними точками
interpolated_points = []
# перший варіант інтерполяції точок
####################################################################################################
# for i in range(len(points) - 1):
#     start_geo = points[i][:2]
#     end_geo = points[i + 1][:2]
#     start_pixel = points[i][2:]
#     end_pixel = points[i + 1][2:]
#
#     # Лінійна інтерполяція між контрольними точками
#     for j in range(num_interpolated_points):
#         ratio = j / num_interpolated_points
#         interpolated_geo = start_geo + (end_geo - start_geo) * ratio
#         interpolated_pixel = start_pixel + (end_pixel - start_pixel) * ratio
#         interpolated_points.append(np.concatenate((interpolated_geo, interpolated_pixel)))
# print(interpolated_points)
#
# # Проходимо по інтерпольованим точкам
# for point in interpolated_points:
#     x, y = point[2:]
#     pt = (int(x), int(y))
#     cv2.drawMarker(panorama, pt, (255, 0, 0), cv2.MARKER_CROSS, 50, thickness=5)
#
# # Збереження результату
# cv2.imwrite('result\\result_panorama.jpg', panorama)

########################################################################################################
    # другий варіант інтерполяції точок
interpolated_geo_points = []
for i in range(len(points) - 1):
    # Відповідність гео- та піксельних координат
    geo_coords = points[i:i+2, :2]
    pixel_coords = points[i:i+2, 2:]

    # Інтерполяція
    x_of_pixel, y_of_pixel = pixel_coords[:, 0], pixel_coords[:, 1]
    x_of_geo, y_of_geo = geo_coords[:, 0], geo_coords[:, 1]

    y_linear = interp1d(x_of_pixel, y_of_pixel)
    y_linear_geo = interp1d(x_of_geo, y_of_geo)

    # Отримуємо інтерпольовані значення
    x_interp = np.linspace(np.min(x_of_pixel), np.max(x_of_pixel), num_interpolated_points)
    x_interp_geo = np.linspace(np.min(x_of_geo), np.max(x_of_geo), num_interpolated_points)

    y_interp = y_linear(x_interp)
    y_interp_geo = y_linear_geo(x_interp_geo)

    # Збираємо координати точок
    interpolated_points.extend(np.column_stack((x_interp, y_interp)))
    interpolated_geo_points.extend(np.column_stack((x_interp_geo, y_interp_geo)))

# Проходимо по інтерпольованим точкам
for point in interpolated_points:
    x, y = point
    pt = (int(x), int(y))
    cv2.drawMarker(panorama, pt, (255, 0, 0), cv2.MARKER_CROSS, 50, thickness=5)

# Збереження результату
cv2.imwrite('result\\result_panorama.jpg', panorama)
#####################################################################################################################
# example of defining geo data of new points

# # Convert lists to NumPy arrays
# interpolated_points = np.array(interpolated_points)
# interpolated_geo_points = np.array(interpolated_geo_points)
# # Function to determine geographical coordinates based on pixel coordinates
# def get_geo_coordinates(pixel_coordinates):
#     # Split known points into separate coordinate arrays
#     x_known = interpolated_points[:, 0]
#     y_known = interpolated_points[:, 1]
#     lat_known = interpolated_geo_points[:, 0]
#     lon_known = interpolated_geo_points[:, 1]
#
#     # Perform linear interpolation to determine geographical coordinates
#     lat_interpolated = interp1d(x_known, lat_known, kind='linear', fill_value='extrapolate')(pixel_coordinates[0])
#     lon_interpolated = interp1d(y_known, lon_known, kind='linear', fill_value='extrapolate')(pixel_coordinates[1])
#
#     return lat_interpolated, lon_interpolated
#
# # Example usage:
# pixel_coords = (11009, 8772)  # Pixel coordinates of the point to be determined
# lat, lon = get_geo_coordinates(pixel_coords)
# print("Geographical coordinates of the point:", lat, lon)



