from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Function to predict a single image
def predict_image(img_path, model, labels):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = labels[predicted_class_idx]
    return predicted_class

# Load the saved model
model = load_model('skin_lesion_classifier.h5')

# List of labels based on your training data
labels = ['Acne', 'Eczema', 'Psoriasis', 'Benign Lesion', 'Malignant Lesion', 'Allergic Reaction', 'Infection']

# Example usage: Change 'path/to/your/image.jpg' to the path of your image
image_path = 'ec.jpg'
predicted_class = predict_image(image_path, model, labels)
print(f"Predicted class: {predicted_class}")
