import requests
from PIL import Image
import base64
import io

# Set your Hugging Face model API URL and API token
API_URL = "https://api-inference.huggingface.co/models/b0c-beep/ft-plant-identifier"  # Replace with your model path
API_TOKEN = "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"  # Replace with your Hugging Face API token

LABEL_MAP = {
    0: "Aloe vera",
    1: "Banana",
    2: "Bilimbi",
    3: "Cantaloupe",
    4: "Cassava",
    5: "Coconut",
    6: "Corn",
    7: "Cucumber",
    8: "Curcuma",
    9: "Eggplant",
    10: "Galangal",
    11: "Ginger",
    12: "Guava",
    13: "Kale",
    14: "Longbeans",
    15: "Mango",
    16: "Melon",
    17: "Orange",
    18: "Paddy",
    19: "Papaya",
    20: "Peperchili",  
    21: "Pineapple",
    22: "Pomelo",
    23: "Shallot",
    24: "Soybeans",
    25: "Spinach",
    26: "Sweetpotatoes",
    27: "Tobacco",
    28: "Waterapple",
    29: "Watermelon",
}

# Function to read and encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to classify an image using the Hugging Face API
def classify_image(image_path):
    encoded_image = encode_image(image_path)
    payload = {"inputs": encoded_image}
    headers = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}
    
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        predictions = response.json()
        
        # Extract the label from the first prediction (highest score)
        predicted_label = predictions[0]['label']  # e.g., 'LABEL_17'
        
        # Extract the numeric part from the label
        predicted_class = int(predicted_label.split('_')[1])  # Get the numeric part
        
        # Get the corresponding plant name
        plant_name = LABEL_MAP[predicted_class]
        
        print(f"Predicted Plant: {plant_name} (Confidence: {predictions[0]['score']:.2f})")
    else:
        print(f"Error: {response.status_code}, {response.text}")

# Path to the image you want to classify
image_path = "./melon891.jpg"  # Replace with your image file name

# Call the function to classify the image
classify_image(image_path)
