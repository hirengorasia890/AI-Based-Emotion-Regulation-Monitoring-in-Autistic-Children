import os
os.environ["GEMINI_API_KEY"] = "AIzaSyD4twwZeMEaSCJHEweHIqO-NwF-QrrDKqU"

# =======================
# === 1. IMPORTS =========
# =======================
import cv2
import os
import glob
import numpy as np
import joblib
from collections import Counter
from retinaface import RetinaFace
from tensorflow import keras
from keras.models import load_model, Model
from keras.applications.resnet import preprocess_input as resnet_preprocess
from keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from deepface import DeepFace
from google.colab.patches import cv2_imshow
import google.generativeai as genai

# =======================
# === 2. API SETUP =======
# =======================
API_KEY = os.getenv("GEMINI_API_KEY")  # set in Colab: os.environ["GEMINI_API_KEY"] = "your-key"

if not API_KEY:
    raise ValueError("Gemini API key not found. Run: os.environ['GEMINI_API_KEY'] = 'YOUR_KEY'")

genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")
print("✅ Gemini API configured successfully.")

# =======================
# === 3. MODEL LOADING ===
# =======================
print("⏳ Loading models...")

# Child / Non-child (ResNet152 + Ensembles)
resnet_model = load_model('/content/model_save/model_save/resnet152_model_save/finetuned_resnet152_model01.keras')
feature_extractor = Model(inputs=resnet_model.input, outputs=resnet_model.layers[-20].output)
voting_ensemble = joblib.load('/content/model_save/model_save/ensemble_model_save/ensemble_voting_classifier.pkl')
stacking_ensemble = joblib.load('/content/model_save/model_save/ensemble_model_save/ensemble_stacking_classifier.pkl')
xgb_clf = joblib.load('/content/model_save/model_save/machine_learning_model_save/xgb_classifier_finetuned.pkl')
lgbm_clf = joblib.load('/content/model_save/model_save/machine_learning_model_save/lightgbm_classifier.pkl')
rf_clf = joblib.load('/content/model_save/model_save/machine_learning_model_save/random_forest_classifier.pkl')
lr_clf = joblib.load('/content/model_save/model_save/machine_learning_model_save/logistic_regression_classifier.pkl')
catboost_clf = joblib.load('/content/model_save/model_save/machine_learning_model_save/catboost_classifier.pkl')
autism_model = load_model('/content/model_save/model_save/autism_classifier(1).keras')

# =======================
# === 4. FUNCTIONS =======
# =======================
img_size = 224
class_names = ['child', 'non-child']

def preprocess_for_resnet(img):
    img = cv2.resize(img, (img_size, img_size))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    return resnet_preprocess(np.expand_dims(img_rgb, axis=0))

def preprocess_for_mobilenet(img):
    img = cv2.resize(img, (img_size, img_size))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    return mobilenet_preprocess(np.expand_dims(img_rgb, axis=0))

# Face detection (RetinaFace)
def detect_faces(img):
    detections = RetinaFace.detect_faces(img)
    faces = []
    if isinstance(detections, dict):
        for key, face in detections.items():
            x1, y1, x2, y2 = face["facial_area"]
            faces.append((x1, y1, x2 - x1, y2 - y1))
    return faces

# Child prediction
def predict_child_status(img):
    x = preprocess_for_resnet(img)
    resnet_prob = resnet_model.predict(x, verbose=0)[0][0]
    features = feature_extractor.predict(x, verbose=0).flatten().reshape(1, -1)

    probs = {
        "resnet": resnet_prob,
        "voting": voting_ensemble.predict_proba(features)[0][1],
        "stacking": stacking_ensemble.predict_proba(features)[0][1],
        "xgb": xgb_clf.predict_proba(features)[0][1],
        "lgbm": lgbm_clf.predict_proba(features)[0][1],
        "rf": rf_clf.predict_proba(features)[0][1],
        "lr": lr_clf.predict_proba(features)[0][1],
        "catboost": catboost_clf.predict_proba(features)[0][1]
    }

    preds = [class_names[int(p > 0.5)] for p in probs.values()]
    vote_count = Counter(preds)
    final_pred = vote_count.most_common(1)[0][0]
    return final_pred, vote_count

# Autism prediction
def predict_autism_status(img):
    x = preprocess_for_mobilenet(img)
    prob = autism_model.predict(x, verbose=0)[0][0]
    return "autistic" if prob > 0.5 else "not autistic"

# Emotion prediction
def predict_emotion(img):
    try:
        analysis = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        return analysis[0]['dominant_emotion']
    except:
        return "neutral"

# Gemini suggestion
def get_suggestion(child_status, autism_status, emotion):
    prompt = f"""
    This {child_status} is {autism_status} and currently appears {emotion}.
    Suggest a simple, kind, caregiver-friendly activity or support idea.
    Keep it short and practical.
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini error: {e}"

# Main analysis
def analyze_image(img_path):
    img = cv2.imread(img_path)
    faces = detect_faces(img)

    results = []
    if not faces:
        print("⚠️ No face found, using full image")
        faces = [(0, 0, img.shape[1], img.shape[0])]

    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]

        child_status, votes = predict_child_status(face_img)
        autism_status = "N/A"
        if child_status == "child":
            autism_status = predict_autism_status(face_img)
        emotion = predict_emotion(face_img)

        suggestion = get_suggestion(child_status, autism_status, emotion)

        results.append({
            "child_status": child_status,
            "autism_status": autism_status,
            "emotion": emotion,
            "suggestion": suggestion
        })

        # Draw on image
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(img, f"{child_status}, {autism_status}, {emotion}",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,255,0), 2)

    return img, results

# =======================
# === 7. TEST ===========
# =======================
if __name__ == "__main__":
    image_folder = "/content/autistic-children-facial-data-set/test/autistic"
    image_paths = glob.glob(os.path.join(image_folder, "*.jpg")) + \
                  glob.glob(os.path.join(image_folder, "*.png")) + \
                  glob.glob(os.path.join(image_folder, "*.jpeg"))

    print(f"Found {len(image_paths)} images in {image_folder}")

    for img_path in image_paths:
      print(f"\n🔍 Analyzing {img_path}...")
      output_img, predictions = analyze_image(img_path)
      print(predictions)

      # Show in Colab
      from google.colab.patches import cv2_imshow
      cv2_imshow(output_img)

    cv2.destroyAllWindows()
# =======================