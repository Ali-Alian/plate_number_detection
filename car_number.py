import cv2

import matplotlib.pyplot as plt

img = cv2.imread('pic/audi.jpg')

def Display(img):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ax.imshow(new_img)
    plt.show()

# Display(img)

plate_casecade = cv2.CascadeClassifier('XML_file/haarcascade_russian_plate_number.xml')

def detect_plate(img):

    plate_img = img.copy()

    plate_detect = plate_casecade.detectMultiScale(plate_img)

    for (x,y,w,h) in plate_detect:
        cv2.rectangle(plate_img, (x,y), (x+w,y+h),(250,0,0), 5)

    return plate_img

result = detect_plate(img)

Display(result)