"""
RoBERTa-Based Video Content Classification Module

This module provides classification of YouTube video content using RoBERTa models,
optimized for speed and compatibility with JSON-based experiment logging.
It includes a binary classifier for harm detection and a multiclass classifier
for categorizing harmful content.
"""

from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch
from scipy.special import softmax
from concurrent.futures import ThreadPoolExecutor
import logging
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
MODELS_DIR = REPO_ROOT / "models"
BINARY_MODEL_DIR = MODELS_DIR / "binary"
MULTICLASS_MODEL_DIR = MODELS_DIR / "multiclass"

class RoBERTaTextDataset(Dataset):
    """
    Custom Dataset for RoBERTa classification.
    """
    def __init__(self, texts, tokenizer, max_length):
        """
        Initialize the dataset with texts for classification.

        Args:
            texts: List of text strings to classify
            tokenizer: RoBERTa tokenizer
            max_length: Maximum sequence length
        """
        self.encodings = tokenizer(texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    
    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.encodings["input_ids"])
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary of encoded inputs
        """
        return {key: val[idx] for key, val in self.encodings.items()}

class RoBERTaClassifier:
    """
    Binary classification using RoBERTa model with optimized inference.
    """

    def __init__(self, model_path, threshold=0.8, logger=None, batch_size=32):
        """
        Initialize classifier.

        Args:
            model_path: Path to RoBERTa model checkpoint
            threshold: Classification threshold for harmful content
            logger: Optional logger
            batch_size: Batch size for inference
        """
        self.model_path = model_path
        self.threshold = threshold
        self.logger = logger or logging.getLogger(__name__)
        self.batch_size = batch_size
        
        # Load model and tokenizer
        self.logger.info(f"Loading RoBERTa model from {model_path}")
        self.model = RobertaForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.max_len = self.tokenizer.model_max_length
        
        # Set up device and model for inference
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        self.logger.info(f"Using device: {self.device} with batch size: {self.batch_size}")

    def _classify_batch(self, texts: List[str]) -> np.ndarray:
        """
        Classify a batch of texts efficiently.

        Args:
            texts: List of text strings to classify

        Returns:
            Array of harm scores (probabilities)
        """
        dataset = RoBERTaTextDataset(texts, self.tokenizer, self.max_len)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_probs = []
        with torch.no_grad():  # Disable gradient computation for inference
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask)
                probs = softmax(outputs.logits.cpu().numpy(), axis=1)
                all_probs.append(probs[:, 1])  # Harm score is the positive class probability
        
        return np.concatenate(all_probs)

    def classify_batch(self, metadata_df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify a batch of metadata entries and add harm scores.

        Args:
            metadata_df: DataFrame containing metadata with 'video_id', 'title', 'description', 'transcript'

        Returns:
            DataFrame with an additional 'harm_score' column
        """
        # Prepare texts for classification
        texts = [f"{row['title']} {row['description']}".strip() 
                 for _, row in metadata_df.iterrows()]
        
        # Classify texts
        harm_scores = self._classify_batch(texts)
        
        # Add harm scores to DataFrame
        metadata_df['harm_score'] = harm_scores
        return metadata_df

class MulticlassClassifier:
    """
    Multiclass classification using RoBERTa model to categorize harmful videos.
    """
    def __init__(self, model_path, batch_size=32, logger=None):
        """
        Initialize multiclass classifier.

        Args:
            model_path: Path to RoBERTa multiclass model checkpoint
            category_mapping: Dict mapping class indices to category names (e.g., {0: "Violence", 1: "Hate Speech"})
            logger: Optional logger
        """
        self.model_path = model_path
        self.logger = logger or logging.getLogger(__name__)
        self.category_mapping = {0: "HH", 1: "SXL", 2: "PH"}
        self.batch_size = batch_size
        
        # Load model and tokenizer
        self.logger.info(f"Loading multiclass RoBERTa model from {model_path}")
        self.model = RobertaForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path, local_files_only=True)
        self.max_len = self.tokenizer.model_max_length
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        self.logger.info(f"Using device: {self.device} for multiclass classification")

    def _classify_batch(self, texts: List[str]) -> np.ndarray:
        dataset = RoBERTaTextDataset(texts, self.tokenizer, self.max_len)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        all_preds = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                preds = np.argmax(outputs.logits.cpu().numpy(), axis=1)
                all_preds.extend(preds)
        return all_preds
        
    def classify_batch(self, metadata_df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify a batch of metadata entries and add category labels.

        Args:
            metadata_df: DataFrame containing metadata with 'video_id', 'title', 'description', 'transcript'

        Returns:
            DataFrame with an additional 'category' column
        """
        # Convert DataFrame to a dictionary format expected by tokenize_fn
        texts = [f"{row['title']} {row['description']}".strip() 
                 for _, row in metadata_df.iterrows()]
        preds = self._classify_batch(texts)
        categories = [self.category_mapping[pred] for pred in preds]
        return pd.DataFrame({
            "video_id": metadata_df["video_id"],
            "category": categories
        })

if __name__ == '__main__':

    roberta_classifier = RoBERTaClassifier(model_path=str(BINARY_MODEL_DIR))
    multiclass_classifier = MulticlassClassifier(model_path=str(MULTICLASS_MODEL_DIR))

    from flask import Flask, request
    app = Flask(__name__)

    @app.route('/classify_roberta', methods=['POST'])
    def classify_roberta():
        """
        Endpoint to classify a batch of video metadata.
        Expects JSON input with 'metadata' key containing a list of metadata dictionaries.
        """
        data = request.json
        metadata_df = pd.DataFrame(data['metadata'])
        
        # Classify and return results
        result_df = roberta_classifier.classify_batch(metadata_df)
        return result_df.to_json(orient='records')

    @app.route('/classify_multiclass', methods=['POST'])
    def classify_multiclass():
        """
        Endpoint to classify a batch of video metadata into categories.
        Expects JSON input with 'metadata' key containing a list of metadata dictionaries.
        """
        data = request.json
        metadata_df = pd.DataFrame(data['metadata'])
        
        # Classify and return results
        result_df = multiclass_classifier.classify_batch(metadata_df)
        return result_df.to_json(orient='records')

    app.run(host='localhost', port=9000)
