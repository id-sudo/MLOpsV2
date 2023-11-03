import os
from PIL import Image
import numpy as np

def load_xray_dataset(dataset_dir):
   
    images = []
    labels = []

    # Define label mapping: 0 for normal, 1 for ill
    label_mapping = {"normal": 0, "pneumonia": 1}

    for label in label_mapping.keys():
        label_dir = os.path.join(dataset_dir, label)
        if not os.path.exists(label_dir):
            continue

        for filename in os.listdir(label_dir):
            img_path = os.path.join(label_dir, filename)
            try:
                # Attempt to open the image file with PIL
                img = Image.open(img_path)
                #img.verify()  # Verify the image file's integrity
               # img = np.array(img)  # Convert PIL image to NumPy array
                img.resize((128,128))
                img.convert('L')
                images.append(img)
                labels.append(label_mapping[label])
            except (IOError, SyntaxError) as e:
                # If the image is corrupted, print an error message and continue
                print(f"Removing {img_path} due to error: {e}")
                os.remove(img_path)

    return images,labels