import cv2
import numpy as np
import datetime
import sys
from pymongo import MongoClient
from gridfs import GridFS
from sklearn.metrics.pairwise import cosine_similarity
import insightface
from insightface.app import FaceAnalysis

# MongoDB Setup
client = MongoClient('mongodb://localhost:27017/')
db = client['face_recognition']
fs = GridFS(db)
users_collection = db['users']
reports_collection = db['reports']
categories_collection = db['categories']

# Report Generator
def generate_blog_report(celebrity_matches):
    if not celebrity_matches:
        print("No matches with high confidence.")
        return

    for match in celebrity_matches:
        try:
            category, filename = match['Celebrity'].split('/')
        except ValueError:
            print(f"Invalid Celebrity format: {match['Celebrity']}")
            continue

        existing_report = reports_collection.find_one({"matches.Celebrity": filename})
        if existing_report:
            print(f"Report for {filename} already exists.")
            continue

        report_data = {
            "category": category,
            "generated_at": datetime.datetime.now(),
            "matches": [match],
            "report_name": filename
        }
        result = reports_collection.insert_one(report_data)
        print(f"Report stored with ID: {result.inserted_id}")

# Face Recognition Class
class FaceRecognition:
    def __init__(self):
        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0)
        self.known_face_embeddings = []
        self.known_face_names = []
        self.matched_celebrities = {}

    def encode_faces(self):
        category_docs = list(categories_collection.find())
        if not category_docs:
            print("No categories found in the database.")
            return

        for cat in category_docs:
            cat_name = cat.get('name', 'unknown')
            persons = cat.get('persons', [])
            for person in persons:
                person_name = person.get('name')
                for img in person.get('images', []):
                    file_id = img.get('file_id')
                    if not file_id:
                        continue
                    try:
                        image_file = fs.get(file_id).read()
                    except Exception as e:
                        print(f"Error retrieving file: {e}")
                        continue

                    np_image = np.frombuffer(image_file, dtype=np.uint8)
                    bgr_image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
                    faces = self.app.get(bgr_image)
                    if faces:
                        emb = faces[0].embedding
                        self.known_face_embeddings.append(emb)
                        self.known_face_names.append(f"{cat_name}/{img.get('filename', file_id)}")
                    else:
                        print(f"No face found in: {img.get('filename')}")

        print(f"Encoded {len(self.known_face_names)} known faces.")

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            sys.exit("Video stream not accessible.")

        celebrity_matches = []

        print("Starting face recognition. Press Ctrl+C to stop.")
        try:
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    print("Frame not received.")
                    break

                faces = self.app.get(frame)
                for face in faces:
                    emb = face.embedding.reshape(1, -1)
                    if not self.known_face_embeddings:
                        continue

                    similarity_scores = cosine_similarity(emb, np.array(self.known_face_embeddings))
                    best_idx = np.argmax(similarity_scores)
                    best_score = similarity_scores[0][best_idx]

                    if best_score > 0.6:
                        name = self.known_face_names[best_idx]
                        category, filename = name.split('/')

                        last_seen = self.matched_celebrities.get(name)
                        now = datetime.datetime.now()
                        if last_seen and (now - last_seen).total_seconds() < 60:
                            continue
                        self.matched_celebrities[name] = now

                        _, snapshot = cv2.imencode('.jpg', frame)
                        timestamp = now.strftime('%Y%m%d%H%M%S')
                        snapshot_filename = f"snapshot_{filename}_{timestamp}.jpg"
                        snapshot_id = fs.put(snapshot.tobytes(), filename=snapshot_filename)

                        match_data = {
                            'Celebrity': name,
                            'Confidence': f"{best_score * 100:.2f}%",
                            'Timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
                            'Image': snapshot_id,
                            'Location': {
                                'x': int(face.bbox[0]),
                                'y': int(face.bbox[1]),
                                'w': int(face.bbox[2] - face.bbox[0]),
                                'h': int(face.bbox[3] - face.bbox[1])
                            }
                        }

                        celebrity_matches.append(match_data)
                        print(f"[MATCH] {filename} with {best_score * 100:.2f}% confidence.")

                # Optional display (won't work in some environments)
                try:
                    cv2.imshow("Real-time Face Recognition", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except:
                    pass

        except KeyboardInterrupt:
            print("Recognition stopped by user.")

        generate_blog_report(celebrity_matches)
        video_capture.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass

# Run System
if __name__ == '__main__':
    fr = FaceRecognition()
    fr.encode_faces()
    fr.run_recognition()
