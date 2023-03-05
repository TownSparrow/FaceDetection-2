import cv2
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN

def img_drawing_with_boxes(img, results_list):
    data = pyplot.imread(img)
    pyplot.imshow(data)
    ax = pyplot.gca()
    for result in results_list:
        x, y, width, height = result['box']
        for key, value in result['keypoints'].items():
            print(key, value)
            if key in ['left_eye', 'right_eye', 'nose']:
               line = Rectangle((value[0], y), 0, height, fill=False, color='red')
               ax.add_patch(line)
    pyplot.show()

img = 'Assets/localtest.jpg'
pixels = pyplot.imread(img)
detector = MTCNN()
faces_db = detector.detect_faces(pixels)
img_drawing_with_boxes(img, faces_db)