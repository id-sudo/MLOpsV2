# Your code here for Model Deployment (Bonus)
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import requests
from PIL import Image
import os
print ("hello")
app = Flask(__name__)

# Load your pre-trained model here
model = tf.keras.models.load_model('C://Users/ZBook/Downloads/Tp1/MLOps/pneumonia_under_sampling.h5')

# Preprocessing function
def preprocess_image(image_file):
    img = Image.open(image_file)
    img = img.resize((128, 128))
    #img = img.convert('L')  # Convert to grayscale
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = img_array.reshape(1, 128, 128, 1)  # Reshape for model input
    return img_array


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        print("avant de lire image")
        
        image_file = request.files['image']
        print("apres lecture")
        file_path = "tempo_img.jpg"
        image_file.save(file_path)
        print("saving")
        print(image_file)
        
        
        # Preprocess the data using your preprocessing function
        preprocessed_data = preprocess_image(image_file)
        print("processing")
        print(preprocessed_data)
        # Make predictions using your pre-trained model
        predictions = model.predict(preprocessed_data)
        print(predictions)
        if predictions > 0.5:
            response = "Pneumonia"
        else:
            response = "Normal"
        
        os.remove(file_path)
    

        # Convert predictions to a response format (e.g., JSON)
        response = {'prediction':response }
        
        return jsonify(response), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
    






if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
