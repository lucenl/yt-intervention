from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer
from datasets import load_dataset, DatasetDict
import numpy as np
import pandas as pd
import wandb
from scipy.special import softmax

# 0) W&B login
wandb.login()

# 1) Load Data
test_csv_path = "results_details_all_puppets.csv"  # Adjust path as needed
test_dataset = load_dataset("csv", data_files={"test": test_csv_path}, split="test")

# Save original data for output
original_data = {
    "intended_harmful_percentage": test_dataset["intended_harmful_percentage"],
    "puppet_id": test_dataset["puppet_id"],
    "section": test_dataset["section"],
    "video_id": test_dataset["video_id"],
    "title": test_dataset["title"],
    "description": test_dataset["description"],
    "transcript": test_dataset["transcript"]
}

# 2) Binary Classification
# Load binary model and tokenizer
BINARY_MODEL_DIR = 'binary'
binary_model = RobertaForSequenceClassification.from_pretrained(BINARY_MODEL_DIR)
binary_tokenizer = RobertaTokenizer.from_pretrained(BINARY_MODEL_DIR)

MAX_LEN = binary_tokenizer.model_max_length
THRESHOLD = 0.8

def tokenize_fn(examples):
    ds = examples.get("description", [""] * len(examples["title"]))
    ts = examples.get("transcript", [""] * len(examples["title"]))
    texts = [f"{t} {d} {tr}" for t, d, tr in zip(examples["title"], ds, ts)]
    return binary_tokenizer(texts, padding="max_length", truncation=True, max_length=MAX_LEN)

# Tokenize for binary classification
binary_tokenized = test_dataset.map(tokenize_fn, batched=True)
binary_tokenized.set_format("torch", columns=["input_ids", "attention_mask"])

# Predict with binary model
binary_trainer = Trainer(model=binary_model)
binary_pred_out = binary_trainer.predict(binary_tokenized)
binary_logits = binary_pred_out.predictions

# Compute binary predictions
binary_probs = softmax(binary_logits, axis=1)
binary_preds = (binary_probs[:, 1] > THRESHOLD).astype(int)  # 1 for harmful, 0 for harmless

# 3) Multiclass Classification for Harmful Videos
# Filter harmful videos
harmful_indices = np.where(binary_preds == 1)[0]
harmful_dataset = test_dataset.select(harmful_indices)

# Load multiclass model and tokenizer
MULTICLASS_MODEL_DIR = 'multiclass'
multiclass_model = RobertaForSequenceClassification.from_pretrained(MULTICLASS_MODEL_DIR)
multiclass_tokenizer = RobertaTokenizer.from_pretrained(MULTICLASS_MODEL_DIR)

# Tokenize for multiclass classification
def multiclass_tokenize_fn(examples):
    ds = examples.get("description", [""] * len(examples["title"]))
    ts = examples.get("transcript", [""] * len(examples["title"]))
    texts = [f"{t} {d} {tr}" for t, d, tr in zip(examples["title"], ds, ts)]
    return multiclass_tokenizer(texts, padding="max_length", truncation=True, max_length=multiclass_tokenizer.model_max_length)

multiclass_tokenized = harmful_dataset.map(multiclass_tokenize_fn, batched=True)
multiclass_tokenized.set_format("torch", columns=["input_ids", "attention_mask"])

# Predict with multiclass model
multiclass_trainer = Trainer(model=multiclass_model)
multiclass_pred_out = multiclass_trainer.predict(multiclass_tokenized)
multiclass_logits = multiclass_pred_out.predictions

# Compute multiclass predictions
multiclass_preds = np.argmax(multiclass_logits, axis=1)  # e.g., 0: Violence, 1: Hate Speech, 2: Misinformation

# Map multiclass predictions to category names
# category_mapping = {0: "Violence", 1: "Hate Speech", 2: "Misinformation"}  # Adjust based on your model's labels
multiclass_categories = [pred for pred in multiclass_preds]

# 4) Combine Results
# Initialize harmful_category column with None
harmful_categories = [None] * len(test_dataset)
# Assign categories to harmful videos
for idx, category in zip(harmful_indices, multiclass_categories):
    harmful_categories[idx] = category

# Create results DataFrame
results_df = pd.DataFrame({
    "intended_harmful_percentage": original_data["intended_harmful_percentage"],
    "puppet_id": original_data["puppet_id"],
    "section": original_data["section"],
    "video_id": original_data["video_id"],
    "title": original_data["title"],
    "description": original_data["description"],
    "transcript": original_data["transcript"],
    "binary_prediction": binary_preds,  # 0: harmless, 1: harmful
    "harmful_category": harmful_categories  # e.g., "Violence", "Hate Speech", None for harmless
})

# 5) Save to CSV
output_csv = "results_details_all_puppets_with_predictions.csv"
results_df.to_csv(output_csv, index=False)
print(f"Saved predictions to {output_csv}")

# 6) Log to W&B
wandb.init(project="roberta_inference", name="binary_and_multiclass_predictions", reinit=True)
wandb.log({"predictions": wandb.Table(dataframe=results_df)})
wandb.finish()