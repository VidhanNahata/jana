import os
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score, classification_report

def compute_metrics(eval_pred):
    """
    Hook for the HuggingFace Trainer returning active statistical evaluation variables.
    Calculates Accuracy alongside weighted and granular Per-Class F1-scores dynamically.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # 1. Base Accuracy metric calculation
    acc = accuracy_score(labels, predictions)
    
    # 2. General overall F1 Score bounds for reporting summary maps
    f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
    
    # 3. Direct Class-Level Exact F1 Score evaluations metrics 
    f1_per_class = f1_score(labels, predictions, average=None, zero_division=0)
    
    metrics = {
        "accuracy": acc,
        "f1_weighted": f1_weighted
    }
    
    # Embed multi-class bounds cleanly into the logging dictionaries
    for i, score in enumerate(f1_per_class):
        metrics[f"f1_class_{i}"] = score
        
    return metrics

def run_indicbert_classification_pipeline():
    """
    Constructs and backpropogates a 4-class multi-lingual NLP evaluation model mapped to civic domains.
    Handles all 6 target Indian languages intrinsically due to standard IndicBERT ALBERT embedding overlap.
    """
    csv_file = "complaints_text.csv"
    output_dir = "indicbert_classifier/"
    model_name = "ai4bharat/indic-bert"
    
    # 0. Pipeline Mocking/Validation fallback limits
    if not os.path.exists(csv_file):
        print(f"Warning: Missing required dataset '{csv_file}'. Dynamically booting a synthetic validation payload...")
        
        # Build syntheic mock dataset containing Indic and Code-Mixed string sequences to bypass IO lockouts
        mock_data = pd.DataFrame({
            "text": [
                "The pothole on MG Road is massive and very dangerous for cars.", 
                "Garbage hasn't been collected from sector 4 in three days.",
                "Water supply is contaminated and smelling horrible locally.",
                "Streetlights are completely dead in the alleyway.",
                "सड़क पर बहुत बड़ा गड्ढा है जो दुर्घटनाओं का कारण बन रहा है।", # Hindi representation
                "కుళాయి పగిలి నీరు వృధా అవుతోంది.", # Telugu representation
                "There is a huge heap of dirty garbage here! \u0915\u0943\u092A\u092F\u093E \u0938\u093E\u092B \u0915\u0930\u0947\u0902", # Code-Mixed
                "बिजली गुल है।"
            ] * 20, # Expand to populate viable evaluation metrics length 
            "label_name": ["Roads", "Sanitation", "Water", "Electricity"] * 40
        })
        mock_data.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"Synthetic Mock IO Pipeline Generated: '{csv_file}'")

    # Defined Structural Mappings
    label_list = ["Roads", "Sanitation", "Water", "Electricity"]
    label2id = {lbl: i for i, lbl in enumerate(label_list)}
    id2label = {i: lbl for i, lbl in enumerate(label_list)}

    # 1. Dataset Extraction Bounds
    print(f"\n[1] Importing HuggingFace Datasets architecture from '{csv_file}'...")
    raw_dataset = load_dataset('csv', data_files=csv_file)
    
    def apply_categorical_mapping(example):
        """Re-map String textual categories strictly into Machine Integer IDs (0-3)."""
        lbl = example.get("label_name", "Roads")
        if lbl not in label2id:
            lbl = "Roads" 
        return {"labels": label2id[lbl]}
    
    # Process string maps dynamically across the target CSV columns
    if "labels" not in raw_dataset["train"].column_names:
        raw_dataset = raw_dataset.map(apply_categorical_mapping)
        
    # Split native extraction securely maintaining random structural sampling (80/20)
    dataset = raw_dataset["train"].train_test_split(test_size=0.2, seed=42)

    # 2. Transformer Loading Directives
    print(f"\n[2] Loading pre-trained ALBERT parameters directly via '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Force HuggingFace parameters bounds into exact 4 target neuron classifications overriding default config
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=4, 
        id2label=id2label, 
        label2id=label2id
    )

    # 3. Text Tokenization Mapping (SentencePiece ALBERT Wrapper logic)
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=128)

    print("\n[3] Formatting entire dataset schema via token limits...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Dynamically pad multi-block sequence lengths into equal matrix computations lazily
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 4. Define Configs (5 Epochs + AdamW Wrapper + Native Warmup)
    print("\n[4] Initializing Multi-GPU scalable HuggingFace Training Engine (5 Epochs)...")
    training_args = TrainingArguments(
        output_dir="./indicbert_temp_checkpoints",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        evaluation_strategy="epoch",  
        save_strategy="epoch",
        load_best_model_at_end=True,
        warmup_steps=100,            # 100 Step Learning Rate Warmup implemented natively via API
        optim="adamw_torch",         # Force standard AdamW execution loop algorithms exclusively
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    # 5. Core Finetuning Operations
    print("\n[5] Launching Trainer Loop Backend limits...")
    trainer.train()

    # 6. Comprehensive Evaluative Scoring outputs
    print("\n--- FINAL EVALUATION LIMITS ---")
    predictions = trainer.predict(tokenized_datasets["test"])
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    print("\nClassification Report (Per-Class Execution F1/Acc Scores):")
    report = classification_report(
        true_labels, pred_labels, 
        target_names=label_list, 
        labels=range(4), 
        zero_division=0
    )
    print(report)

    # 7. Saving and Export 
    print(f"\n[6] Writing optimized NLP architecture into '{output_dir}'")
    trainer.save_model(output_dir)
    print("Optimization Tasks Complete.")

if __name__ == "__main__":
    run_indicbert_classification_pipeline()
