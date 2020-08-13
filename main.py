import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from keras.datasets import mnist
from sklearn.metrics import accuracy_score

#load_data
(x_train, y_train),(x_test, y_test) = mnist.load_data()

#cho x_train
x_train_feature = []
for i in range(len(x_train)):
    feature = hog(x_train[i], orientations=9, pixels_per_cell=(14,14), cells_per_block=(1,1), block_norm="L2")
    x_train_feature.append(feature)
x_train_feature = np.array(x_train_feature, dtype=np.float32)

#cho x_test
x_test_feature = []
for i in range(len(x_test)):
    feature = hog(x_test[i], orientations=9, pixels_per_cell=(14,14), cells_per_block=(1,1), block_norm="L2")
    x_test_feature.append(feature)
x_test_feature = np.array(x_test_feature, dtype=np.float32)

model = LinearSVC(C=10)
model.fit(x_train_feature, y_train)
y_pre = model.predict(x_test_feature)
print(accuracy_score(y_test, y_pre))

image = cv2.imread("digit.jpg")
im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
im_blur = cv2.GaussianBlur(im_gray, (5,5), 0)
im, thre = cv2.threshold(im_blur, 90, 255, cv2.THRESH_BINARY_INV)
_,contours, hierachy = cv2.findContours(thre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(cnt) for cnt in contours]

for i in contours:
    (x, y, w, h) = cv2.boundingRect(i)
    cv2.rectangle(image, (x,y), (x+w,y+h), (255, 255, 0), 3)
    roi = thre[y:y+h, x:x+w]
    roi = np.pad(roi, (20, 20), 'constant', constant_values=(0, 0))
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3,3))

    #calculate the HOG features
    roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1,1), block_norm="L2")
    nbr  = model.predict(np.array([roi_hog_fd], np.float32))
    cv2.putText(image, str(int(nbr[0])), (x,y), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 3)
    cv2.imshow("Result", image)
cv2.imwrite("image_pand.jpg", image)
cv2.waitKey()
cv2.destroyAllWindows()