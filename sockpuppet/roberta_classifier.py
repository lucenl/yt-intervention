from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer
from datasets import load_dataset
import torch
from scipy.special import softmax
from concurrent.futures import ThreadPoolExecutor
from metadata_extractor import MetadataExtractor
import logging
import os
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
Classification module using RoBERTa model
"""
class RoBERTaClassifier:
    """Classification using RoBERTa model"""
    
    def __init__(self, model_path, threshold=0.5, logger=None):
        """
        Initialize classifier
        
        Args:
            model_path: Path to RoBERTa model checkpoint
            threshold: Classification threshold
            logger: Optional logger
        """
        self.model_path = model_path
        self.threshold = threshold
        self.logger = logger or logging.getLogger(__name__)
        
        # Load model and tokenizer
        self.logger.info(f"Loading RoBERTa model from {model_path}")
        self.model = RobertaForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        self.max_len = self.tokenizer.model_max_length
        
        # Set up device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.logger.info(f"Using device: {self.device}")
    
    def classify_from_csv(self, csv_path):
        """
        Classify videos from CSV file
        
        Args:
            csv_path: Path to CSV with video metadata
            
        Returns:
            DataFrame with predictions added
        """
        self.logger.info(f"Classifying videos from {csv_path}")
        
        # Load dataset
        dataset = load_dataset("csv", data_files={"test": csv_path}, split="test")
        
        # Tokenize
        def tokenize_fn(examples):
            ds = examples.get("description", [""] * len(examples["title"]))
            ts = examples.get("transcript", [""] * len(examples["title"]))
            texts = []
            for t, d, tr in zip(examples["title"], ds, ts):
                t = t or ""
                d = d or ""
                tr = tr or ""
                texts.append(f"{t} {d} {tr}".strip())
            return self.tokenizer(texts, padding="max_length", truncation=True, max_length=self.max_len)
        
        tokenized_dataset = dataset.map(tokenize_fn, batched=True)
        tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask"])
        
        # Run prediction
        trainer = Trainer(model=self.model)
        pred_output = trainer.predict(tokenized_dataset)
        
        # Process predictions
        logits = pred_output.predictions
        probs = softmax(logits, axis=1)
        preds = (probs[:, 1] > self.threshold).astype(int)
        
        # Add predictions to original data
        df = pd.read_csv(csv_path)
        df['harm_score'] = probs[:, 1]
        df['prediction'] = preds
        
        # Save results
        results_path = os.path.splitext(csv_path)[0] + "_classified.csv"
        df.to_csv(results_path, index=False)
        
        self.logger.info(f"Classification complete. Found {df['prediction'].sum()} harmful videos out of {len(df)}")
        
        return df
    
    def classify_video_list(self, videos, video_metadata=None):
        """
        Classify a list of videos
        
        Args:
            videos: List of Video objects
            video_metadata: Optional pre-loaded metadata DataFrame
            
        Returns:
            List of harm scores
        """
        # Get video IDs
        video_ids = [v.videoId for v in videos]
        
        if video_metadata is not None:
            # Use pre-loaded metadata
            df = video_metadata
            
            # Filter to just the videos we need
            df = df[df['video_id'].isin(video_ids)]
            
            # Keep original order
            df = pd.DataFrame([df[df['video_id'] == vid].iloc[0] if vid in df['video_id'].values else None 
                             for vid in video_ids])
            
        else:
            # Create temporary metadata file
            temp_df = pd.DataFrame({
                'links': [f"https://youtube.com/watch?v={vid}" for vid in video_ids],
                'video_id': video_ids,
                'title': [getattr(v, 'title', '') for v in videos],
                'description': [''] * len(videos),
                'transcript': [''] * len(videos)
            })
            
            temp_path = "temp_classification.csv"
            temp_df.to_csv(temp_path, index=False)
            
            # Use our classification function
            df = self.classify_from_csv(temp_path)
            
            # Clean up
            os.remove(temp_path)
        
        # Get harm scores in the same order as input videos
        harm_scores = []
        for video_id in video_ids:
            matching_rows = df[df['video_id'] == video_id]
            if len(matching_rows) > 0:
                harm_scores.append(float(matching_rows.iloc[0]['harm_score']))
            else:
                # Default to threshold if video not found
                harm_scores.append(self.threshold)
        
        return harm_scores


