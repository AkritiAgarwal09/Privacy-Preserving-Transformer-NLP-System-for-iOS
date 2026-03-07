"""
Privacy-Preserving Multi-Task NLP System
Training Pipeline — Multi-Task Transformer Fine-Tuning

Tasks supported:
  - Sentiment Classification (positive / negative / neutral)
  - Intent Detection (question / command / statement / greeting / complaint)
  - Emotion Classification (joy / sadness / anger / fear / surprise / neutral)
  - Toxicity Detection (safe / toxic / severe_toxic)

Model: DistilBERT (distilbert-base-uncased)
Framework: PyTorch + HuggingFace Transformers
"""

import os
import json
import time
import torch
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertModel,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import f1_score, accuracy_score
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Task Definitions
# ─────────────────────────────────────────────

TASKS = {
    "sentiment": {
        "labels": ["negative", "neutral", "positive"],
        "num_labels": 3,
    },
    "intent": {
        "labels": ["question", "command", "statement", "greeting", "complaint"],
        "num_labels": 5,
    },
    "emotion": {
        "labels": ["joy", "sadness", "anger", "fear", "surprise", "neutral"],
        "num_labels": 6,
    },
    "toxicity": {
        "labels": ["safe", "toxic", "severe_toxic"],
        "num_labels": 3,
    },
}


# ─────────────────────────────────────────────
# Synthetic Dataset (replace with real data)
# ─────────────────────────────────────────────

SYNTHETIC_DATA = {
    "sentiment": [
        ("I love this product, it works great!", 2),
        ("Terrible experience, never buying again.", 0),
        ("It's okay, nothing special.", 1),
        ("Absolutely amazing quality!", 2),
        ("Waste of money, very disappointed.", 0),
        ("Decent enough for the price.", 1),
        ("Could not be happier with my purchase!", 2),
        ("Broke after two days. Very poor quality.", 0),
        ("Average product, does what it says.", 1),
        ("This is the best thing I have ever bought!", 2),
    ],
    "intent": [
        ("What is the weather like today?", 0),
        ("Turn off the lights.", 1),
        ("I went to the store yesterday.", 2),
        ("Hello, how are you?", 3),
        ("This service is unacceptable!", 4),
        ("Can you help me with this?", 0),
        ("Please send me the report.", 1),
        ("The meeting went well.", 2),
        ("Good morning everyone!", 3),
        ("I demand a refund immediately.", 4),
    ],
    "emotion": [
        ("I am so happy today!", 0),
        ("I feel so alone and miserable.", 1),
        ("This makes me furious!", 2),
        ("I am terrified of what might happen.", 3),
        ("Wow, I did not expect that at all!", 4),
        ("Just another ordinary day.", 5),
        ("This is the best news ever!", 0),
        ("I miss my family so much.", 1),
        ("Stop doing that, it drives me crazy!", 2),
        ("What if something goes wrong?", 3),
    ],
    "toxicity": [
        ("Have a great day everyone!", 0),
        ("You are an idiot.", 1),
        ("I will destroy you, you worthless piece of trash.", 2),
        ("Thanks for your help today.", 0),
        ("Shut up, nobody cares what you think.", 1),
        ("That was absolutely disgusting behavior.", 1),
        ("Let us work together on this.", 0),
        ("Go kill yourself.", 2),
        ("I respectfully disagree with your point.", 0),
        ("You are worthless and should disappear.", 2),
    ],
}


class MultiTaskDataset(Dataset):
    def __init__(self, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        for task, examples in SYNTHETIC_DATA.items():
            for text, label in examples:
                self.samples.append({
                    "text": text,
                    "task": task,
                    "label": label,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        encoding = self.tokenizer(
            sample["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "task": sample["task"],
            "label": torch.tensor(sample["label"], dtype=torch.long),
        }


# ─────────────────────────────────────────────
# Multi-Task Model Architecture
# ─────────────────────────────────────────────

class MultiTaskNLPModel(nn.Module):
    """
    Shared DistilBERT encoder with per-task classification heads.
    Architecture:
        DistilBERT (shared backbone)
            ├── Task Router (lightweight MLP)
            ├── Sentiment Head  (3 classes)
            ├── Intent Head     (5 classes)
            ├── Emotion Head    (6 classes)
            └── Toxicity Head   (3 classes)
    """

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size  # 768

        # Shared projection layer
        self.shared_projection = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Per-task classification heads
        self.task_heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, config["num_labels"]),
            )
            for task, config in TASKS.items()
        })

    def forward(self, input_ids, attention_mask, task: str):
        # Shared encoding
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        # Shared projection
        projected = self.shared_projection(cls_output)

        # Task-specific head
        logits = self.task_heads[task](projected)
        return logits

    def get_embeddings(self, input_ids, attention_mask):
        """Return CLS embeddings for semantic similarity."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────

@dataclass
class TrainingConfig:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 128
    batch_size: int = 8
    num_epochs: int = 5
    learning_rate: float = 2e-5
    warmup_steps: int = 50
    output_dir: str = "./checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train(config: TrainingConfig):
    logger.info(f"Training on device: {config.device}")
    os.makedirs(config.output_dir, exist_ok=True)

    tokenizer = DistilBertTokenizerFast.from_pretrained(config.model_name)
    model = MultiTaskNLPModel(config.model_name).to(config.device)

    dataset = MultiTaskDataset(tokenizer, config.max_length)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    total_steps = len(dataloader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, config.warmup_steps, total_steps
    )

    loss_fn = nn.CrossEntropyLoss()
    history = []

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0
        all_preds, all_labels = [], []

        for batch in dataloader:
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            labels = batch["label"].to(config.device)

            # Group by task for correct head routing
            tasks_in_batch = batch["task"]
            unique_tasks = list(set(tasks_in_batch))

            total_loss = torch.tensor(0.0, device=config.device)

            for task in unique_tasks:
                mask = [t == task for t in tasks_in_batch]
                mask_tensor = torch.tensor(mask, dtype=torch.bool)

                task_input_ids = input_ids[mask_tensor]
                task_attention_mask = attention_mask[mask_tensor]
                task_labels = labels[mask_tensor]

                logits = model(task_input_ids, task_attention_mask, task)
                loss = loss_fn(logits, task_labels)
                total_loss += loss

                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(task_labels.cpu().numpy())

            total_loss /= len(unique_tasks)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / len(dataloader)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")

        metrics = {"epoch": epoch + 1, "loss": avg_loss, "accuracy": acc, "f1": f1}
        history.append(metrics)
        logger.info(f"Epoch {epoch+1}/{config.num_epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

    # Save model
    model_path = os.path.join(config.output_dir, "multitask_nlp_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "task_configs": TASKS,
        "history": history,
    }, model_path)
    tokenizer.save_pretrained(config.output_dir)
    logger.info(f"Model saved to {model_path}")

    # Save training history
    with open(os.path.join(config.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return model, tokenizer, history


if __name__ == "__main__":
    config = TrainingConfig()
    model, tokenizer, history = train(config)
    print("\nTraining complete.")
    print(f"Final metrics: {history[-1]}")
