import numpy as np
def process(images,labels):
    data = np.array(images)
    labels = np.array(labels)


# Normalize the pixel values (assuming your data is grayscale)
    data = data / 255.0  
    data = data.reshape(-1, 128, 128, 1)
    return data,labels