import cv2
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model


def extract_frames(video_path, num_frames=5):
    cap = cv2.VideoCapture(video_path)
    frames=[]
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, length// num_frames)
    
    
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i* frame_interval)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.resize(frame, (224, 224)))
    cap.release()
    return frames

def extract_features(frames, model):
    features=[]
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.expand_dims(frame, axis=0)
        feature = model.predict(frame)
        features.append(feature.flatten())
    return np.mean(features, axis=0)

base_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

def get_video_features(video_path):
    frames = extract_frames(video_path)
    features = extract_features(frames, model)
    return features

input_video_features = get_video_features('/home/the_priest/Desktop/ML/PROJECT_CODE/chryswen_project_code/random_snippet.mp4')

def get_similarity(features1, features2):
    return cosine_similarity([features1], [features2])[0][0]

video_dir = '/home/the_priest/Desktop/ML/PROJECT_CODE/chryswen_project_code/video_tests/'
similarities = []

for video_file in os.listdir(video_dir):
    video_path = os.path.join(video_dir, video_file)
    video_features = get_video_features(video_path)
    similarity = get_similarity(input_video_features, video_features)
    similarities.append((video_file, similarity))

# Normalize to percentage
similarities = [(video, sim * 100) for video, sim in similarities]

results = dict()
# Print results
for video, similarity in similarities:
#     print(f"Similarity with {video}: {similarity:.2f}%")
    results[video] = similarity

    
def similarity_percentage(result_dictionary:dict = results):
    max_key = max(result_dictionary, key=result_dictionary.get)
    print (f"{max_key[:-4]} {result_dictionary[max_key]:.1f}")
    
similarity_percentage()

# end_time = time.time()

# end_time - start_time
