import os
import numpy as np

from preprocessing import segment_audio, extract_features


def build_dataset(dataset_path):

    X = []
    y = []

    for label_name in ["PD", "HC"]:

        class_path = os.path.join(dataset_path, label_name)

        label = 1 if label_name == "PD" else 0

        for item in os.listdir(class_path):

            item_path = os.path.join(class_path, item)

            # ---- CASE 1: Direct .wav file ----
            if item.endswith(".wav"):

                if "dataset2" in dataset_path:
                    segments = segment_audio(item_path, segment_duration=1)
                else:
                    segments = segment_audio(item_path, segment_duration=3)    
                    
                    
    

    
                for segment in segments:
                    features = extract_features(segment)
                    X.append(features)
                    y.append(label)

            # ---- CASE 2: Person folder ----
            elif os.path.isdir(item_path):

                for file in os.listdir(item_path):

                    if file.endswith(".wav"):

                        file_path = os.path.join(item_path, file)

                        segments = segment_audio(file_path)

                        for segment in segments:
                            features = extract_features(segment)
                            X.append(features)
                            y.append(label)

    return np.array(X), np.array(y)


# ---- Build Clients ----

print("Building Client 1 (Italian Dataset)...")
X1, y1 = build_dataset("D:/Finall/datasett")

print("Building Client 2 (41+41 Dataset)...")
X2, y2 = build_dataset("D:/Finall/dataset2")

print("Building Client 3 (Large Dataset)...")
X3, y3 = build_dataset("D:/Finall/dataset3")


print("\nClient 1 shape:", X1.shape)
print("Client 2 shape:", X2.shape)
print("Client 3 shape:", X3.shape)