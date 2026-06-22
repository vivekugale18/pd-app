import os
import librosa
import numpy as np

dataset_path = "D:\Finall\datasett"

durations = []

for label in ["PD", "HC"]:
    class_path = os.path.join(dataset_path, label)
    
    for person in os.listdir(class_path):
        person_path = os.path.join(class_path, person)
        
        for file in os.listdir(person_path):
            if file.endswith(".wav"):
                file_path = os.path.join(person_path, file)
                y, sr = librosa.load(file_path, sr=None)
                duration = librosa.get_duration(y=y, sr=sr)
                durations.append(duration)

print("Min duration:", np.min(durations))
print("Max duration:", np.max(durations))
print("Mean duration:", np.mean(durations))