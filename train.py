from transformers import AutoImageProcessor, AutoModelForImageClassification, Trainer, TrainingArguments
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch
import os

# Define the plant names
plant_names = [
    "aloevera",
    "banana",
    "bilimbi",
    "cantaloupe",
    "cassava",
    "coconut",
    "corn",
    "cucumber",
    "curcuma",
    "eggplant",
    "galangal",
    "ginger",
    "guava",
    "kale",
    "longbeans",
    "mango",
    "melon",
    "orange",
    "paddy",
    "papaya",
    "peperchili",
    "pineapple",
    "pomelo",
    "shallot",
    "soybeans",
    "spinach",
    "sweetpotatoes",
    "tobacco",
    "waterapple",
    "watermelon"
]


# Create a mapping from plant names to indices
label2id = {name: index for index, name in enumerate(plant_names)}
num_labels = len(label2id)

# Load the model and processor
processor = AutoImageProcessor.from_pretrained("umutbozdag/plant-identity")
model = AutoModelForImageClassification.from_pretrained(
    "umutbozdag/plant-identity",
    num_labels=num_labels,
    ignore_mismatched_sizes=True
)


# Custom Dataset
class CustomDataset(ImageFolder):
    def __getitem__(self, index):
        path, target = super().__getitem__(index)
        # Get the corresponding plant name
        plant_name = self.classes[target]
        label = label2id[plant_name]  # Map plant name to index
        # Process the image
        image = processor(images=path, return_tensors="pt", do_rescale=False)["pixel_values"].squeeze(0)
        return {"pixel_values": image, "label": label}

# Define the path to your test folder
test_folder = "train"  # Change this to your actual path

dataset = CustomDataset(root=test_folder, transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
]))

# Split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    logging_dir="./logs",
    eval_strategy="epoch",  # Update this to `eval_strategy`
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Save the model and processor
model.save_pretrained("fine_tuned_model")
processor.save_pretrained("fine_tuned_model")
