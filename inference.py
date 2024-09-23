import os
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

# Correct the path to the folder where the fine-tuned model was saved
model_path = "fine_tuned_model"  # Make sure this is the directory where you saved your model

# Load the processor and model from the saved directory
processor = AutoImageProcessor.from_pretrained(model_path)
model = AutoModelForImageClassification.from_pretrained(model_path)

# Define the path to your test folder containing plant images
test_folder = "test"

# Updated mapping of label IDs to plant names, including all plants from your test folder
id2label = {
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

# Iterate through each folder and image in the test directory
for plant_folder in os.listdir(test_folder):
    folder_path = os.path.join(test_folder, plant_folder)
    if os.path.isdir(folder_path):
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            if image_path.endswith(('.png', '.jpg', '.jpeg')):
                # Load the image
                image = Image.open(image_path)

                # Process the image
                inputs = processor(images=image, return_tensors="pt")

                # Make prediction
                with torch.no_grad():
                    logits = model(**inputs).logits

                # Get the predicted class
                predicted_class = logits.argmax(-1).item()
                plant_name = id2label[predicted_class]

                print(f"Image: {image_name} | Predicted Plant: {plant_name}")
  