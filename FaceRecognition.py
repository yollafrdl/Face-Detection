from matplotlib import pyplot
from matplotlib.patches import Rectangle
from PIL import Image
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
import numpy as np
import cv2

def face_classification(face):
    pixels = face.astype('float32')
    samples = np.expand_dims(pixels, axis=0)
    samples = preprocess_input(samples, version=2)
    model = VGGFace(model='resnet50')
    yhat = model.predict(samples)
    results = decode_predictions(yhat)
    label = results[0][0][0][3:-2]

    return label
    

def extract_face(img, required_size=(224, 224)):
    detector = MTCNN()
    results = detector.detect_faces(img)
    detection = []
    for result in results:
        x1, y1, width, height = result['box']
        x2, y2 = x1 + width, y1 + height
        location = x1, y1, x2, y2
        face = img[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = np.asarray(image)
        name = face_classification(face_array)
        detection.append((location, name))
    return detection

def detect_image(url):
    while True:
        # url = 'channing jessie.jpg'
        image = pyplot.imread(url)
        face = extract_face(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for loc, name in face:
            left, top, right, bottom = loc
            cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 1)
            cv2.rectangle(image, (left, top - 35), (right, top), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image, name, (left + 3, top - 3), font, 0.5, (255, 255, 255), 1)

        cv2.imshow('output', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def detect_video(url):
    vidcap = cv2.VideoCapture(url)
    while True:
        ret, frame = vidcap.read()
        face = extract_face(frame)
        for loc, name in face:
            left, top, right, bottom = loc
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)
            cv2.rectangle(frame, (left, top - 35), (right, top), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 3, top - 3), font, 0.5, (255, 255, 255), 1)

        cv2.imshow('output', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    vidcap.release()
    cv2.destroyAllWindows()

# url = 'Data/coba3.mp4'
# detect_video(url)

url = '1D.jpg'
detect_image(url)

       




