import cv2
import numpy as np
import rasterio
from rasterio.plot import show
####################################################################################################
#Створення tiff файлу на основі 5 точок
# Контрольні точки (гео + пікселі)
img_path = 'result\\final_image_stitched.jpg'
img = cv2.imread(img_path)
outputPath = 'result\\georefRaster.tif'
new_outputPath = 'result\\new_georefRaster.tif'

points = np.array([[48.54334883179478, 35.0948654426536, 6641, 9270],
                   [48.542041777503826, 35.09759793192003, 8518, 7017],
                   [48.54071025778341, 35.09941698099394, 10415, 5526],
                   [48.53931743764255, 35.10125316177766, 12402, 4040],
                   [48.54097494166265, 35.10354552977222, 10335, 1848]])

#open un_geo_referenced raster
unRefRaster = rasterio.open(img_path)

# Створення списку контрольних точок
gcps = []
for point in points:
    x_of_geo, y_of_geo, x_of_pixel, y_of_pixel  = point
    gcps.append(rasterio.control.GroundControlPoint(row=y_of_pixel, col=x_of_pixel, x=y_of_geo, y=x_of_geo))

transformation = rasterio.transform.from_gcps(gcps)

#create raster and write bands
with rasterio.open(
        outputPath,
        'w',
        driver='GTiff',
        height=unRefRaster.read(1).shape[0],
        width=unRefRaster.read(1).shape[1],
        count=3,
        dtype=unRefRaster.read(1).dtype,
        crs=rasterio.crs.CRS.from_epsg(4326),
        transform=transformation,
) as dst:
    dst.write(unRefRaster.read(1), 1)
    dst.write(unRefRaster.read(2), 2)
    dst.write(unRefRaster.read(3), 3)

###################################################################################################################
# створення додаткових контрольних точок та визначення їх гео координат
# Create our ORB detector and detect keypoints and descriptors
orb = cv2.ORB_create(nfeatures=30)
keypoints = orb.detect(img, None)

# Отримання піксельних координат для кожної ключової точки
geo_coordinates = []
pixel_coordinates = [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]

# Відкриття TIFF файлу
with rasterio.open(outputPath) as src:
    # Доступ до даних та метаданих
    data = src.read()
    profile = src.profile

    # Приклад використання
    for pixel in pixel_coordinates:
        geo_lon, geo_lat = src.xy(pixel[1], pixel[0])
        geo_coordinates.append((geo_lat, geo_lon))

for i in range(len(pixel_coordinates)):
    print(f'pixel {pixel_coordinates[i][0]} {pixel_coordinates[i][1]} --> geo {geo_coordinates[i][0]} {geo_coordinates[i][1]}')

for point in pixel_coordinates:
    x, y = point
    pt = (int(x), int(y))
    cv2.drawMarker(img, pt, (255, 0, 0), cv2.MARKER_CROSS, 50, thickness=5)

# Збереження результату
cv2.imwrite('result\\result_panorama.jpg', img)



# Додавання географічних та піксельних координат до масиву points
for i in range(len(geo_coordinates)):
    geo = geo_coordinates[i]
    pixel = pixel_coordinates[i]
    new_row = np.array([geo[0], geo[1], pixel[0], pixel[1]])
    points = np.vstack([points, new_row])

print(f'points : {points}')
#open un_geo_referenced raster
unRefRaster = rasterio.open(img_path)

# Створення списку контрольних точок
gcps = []
for point in points:
    x_of_geo, y_of_geo, x_of_pixel, y_of_pixel  = point
    gcps.append(rasterio.control.GroundControlPoint(row=y_of_pixel, col=x_of_pixel, x=y_of_geo, y=x_of_geo))

transformation = rasterio.transform.from_gcps(gcps)

#create raster and write bands
with rasterio.open(
        new_outputPath,
        'w',
        driver='GTiff',
        height=unRefRaster.read(1).shape[0],
        width=unRefRaster.read(1).shape[1],
        count=3,
        dtype=unRefRaster.read(1).dtype,
        crs=rasterio.crs.CRS.from_epsg(4326),
        transform=transformation,
) as dst:
    dst.write(unRefRaster.read(1), 1)
    dst.write(unRefRaster.read(2), 2)
    dst.write(unRefRaster.read(3), 3)

with rasterio.open(new_outputPath) as src:
    geo_lon, geo_lat = src.xy(8619, 9484)
    print("test")
    print(f'pixel 9484, 8619 --> geo {geo_lat} {geo_lon}')