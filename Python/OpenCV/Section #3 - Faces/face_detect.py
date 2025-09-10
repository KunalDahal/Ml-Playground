import cv2 as cv

# Load image
img = cv.imread('Y:/CODIII/Journey/OpenCV/Resources/Photos/group 2.jpg')

# Load Haar Cascade
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# Detect faces
faces_rect = haar_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=6)

# Draw rectangles around faces
for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

# Put text once at top-left corner
text = f'Number of faces found = {len(faces_rect)}'
cv.putText(img, text, (10, 30), cv.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)

# Show result
cv.imshow('Detected Faces', img)
cv.waitKey(0)
cv.destroyAllWindows()
