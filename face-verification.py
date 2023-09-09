from keras_facenet import FaceNet
from scipy import spatial
import cv2

embedder = FaceNet()

input = 'data/brad.jpeg'
inp_detections = embedder.extract(input, threshold=0.95)
inp_embedding = inp_detections[0].get('embedding')
print(inp_embedding)


def draw_result(image, face, result, similarity_score):
    x, y, w, h = face
    color = (0, 255, 0) if result == "Verified" else (0, 0, 255)
    result = result + " " + str(round(similarity_score*100,2)) + "%"
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    text = f"Face: {result}"
    cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def generate_embeddings(face):
    return embedder.extract(face, threshold=0.95)


# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 represents the default webcam


while True:
    ret, frame = cap.read()  # Read a frame from the webcam

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    detection = generate_embeddings(image_rgb)

    for itr in range(len(detection)):
        cur = detection[itr]
        embedding = cur['embedding']
        
        similarity_score = 1 - spatial.distance.cosine(embedding, inp_embedding)

        verification_result = "checking"

        if similarity_score > 0.65:
            verification_result = "Verified"
        else:
            verification_result = "Not Verified"

        draw_result(frame, cur['box'], verification_result, similarity_score)

    cv2.imshow('Face Verification', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
