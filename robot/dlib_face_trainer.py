import dlib
import cv2
import numpy as np

# Initialize dlib's face detector and face recognition model
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Download from dlib's website
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')  # Download from dlib's website

# Load the image
image = cv2.imread('junghyun.jpg')

# Resize the image to 160x160 (optional step, if required for your task)
#image_resized = cv2.resize(image, (160, 160))

# Convert image to grayscale for detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = detector(gray)
print(faces)


# rectangles[[(233, 161) (448, 376)]]





# Initialize a list to hold the face embeddings
face_embeddings = []

# For each face, we extract the face features (embedding) using dlib's face recognition model
for face in faces:
    # Get the landmarks (facial key points)
    shape = sp(gray, face)
    print(shape)
    
    # Get the face descriptor (embedding)
    face_descriptor = facerec.compute_face_descriptor(image, shape)
    
    # Convert the face descriptor to a numpy array
    face_descriptor = np.array(face_descriptor)
    
    # Append the descriptor to the list of embeddings
    face_embeddings.append(face_descriptor)

    # Draw the rectangle around the face
    x1, y1, x2, y2 = (face.left(), face.top(), face.right(), face.bottom())
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Save the face embeddings to a file (e.g., numpy .npy format)
np.save('faqwerqewre_embedasdfdings.npy', face_embeddings)

# Show the image with detected faces
cv2.imshow("Detected Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
