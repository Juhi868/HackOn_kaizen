import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification

# Load dataset
dataset_path = "Disease_symptom_and_patient_profile_dataset.csv"
df = pd.read_csv(dataset_path)

# Define valid symptoms (case-insensitive, spaces handled)
valid_symptoms = ["Fever", "Cough", "Fatigue", "Difficulty Breathing"]

# Extract unique disease labels
disease_labels = df['Disease'].unique()
num_labels = len(disease_labels)
label_to_id = {label: idx for idx, label in enumerate(disease_labels)}
id_to_label = {idx: label for label, idx in label_to_id.items()}

# Load BioBERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
model = BertForSequenceClassification.from_pretrained("dmis-lab/biobert-v1.1", num_labels=num_labels)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Display available symptom options
print("\nAvailable Symptoms:")
print(", ".join(valid_symptoms), "\n")

# Get user input and validate symptoms
while True:
    symptom_input = input("Enter symptoms (separated by commas): ").strip().lower()

    if not symptom_input:
        print("❌ No input provided. Please enter at least one symptom.")
        continue

    symptom_list = [symptom.strip().title() for symptom in symptom_input.split(",")]
    
    # Check for invalid symptoms
    invalid_inputs = [symptom for symptom in symptom_list if symptom not in valid_symptoms]

    if invalid_inputs:
        print(f"\n❌ Invalid symptoms detected: {', '.join(invalid_inputs)}")
        print(f"✅ Please enter valid symptoms from this list: {', '.join(valid_symptoms)}\n")
    else:
        break  # Valid input received

# Tokenize the input text
symptom_text = " ".join(symptom_list)
tokens = tokenizer(symptom_text, padding=True, truncation=True, return_tensors="pt")
tokens = {key: value.to(device) for key, value in tokens.items()}

# Get model predictions
with torch.no_grad():
    outputs = model(**tokens)

# Convert logits to predicted class
predicted_class = torch.argmax(outputs.logits, dim=1).item()
predicted_disease = id_to_label.get(predicted_class, "Unknown Disease")

# Display results
print(f"\n🩺 Predicted Disease: **{predicted_disease}**")
