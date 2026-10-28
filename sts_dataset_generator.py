import pandas as pd
import random
import torch
import warnings
from datasets import load_dataset
from transformers import MarianMTModel, MarianTokenizer
from sentence_transformers import SentenceTransformer, util

# Suppress HuggingFace warnings for cleaner output
warnings.filterwarnings("ignore")

class STSDatasetGenerator:
    """
    Pipeline to generate an English-Hindi Semantic Textual Similarity (STS) dataset.
    Mixes translated pairs from GLUE's STSb benchmark with civic-domain augmented pairs.
    """
    def __init__(self, translation_model_name="Helsinki-NLP/opus-mt-en-hi", sbert_model_name="sentence-transformers/LaBSE"):
        # 1. Load Helsinki-NLP Translation Model (English to Hindi)
        print(f"Loading translation model: {translation_model_name}...")
        self.tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
        self.translator = MarianMTModel.from_pretrained(translation_model_name)
        
        # 2. Load Cross-Lingual Semantic Model (LaBSE) for accurate 0.0-1.0 scoring
        print(f"Loading similarity model for labeling: {sbert_model_name}...")
        self.similarity_model = SentenceTransformer(sbert_model_name)
        
        # Optimizations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using compute device: {self.device}")
        self.translator.to(self.device)
        self.similarity_model.to(self.device)

    def translate_en_to_hi(self, texts):
        """Batch translates a list of English texts to Hindi."""
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            translated_tokens = self.translator.generate(**inputs)
        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]
        
    def generate_augmented_civic_pairs(self, target_pairs=200):
        """Generates domain-specific civic complaint pairs (both exact and conflicting)."""
        base_complaints = [
            "The streetlights are not working in our locality.",
            "There is a huge garbage dump near the school.",
            "Water supply is contaminated and smells foul.",
            "Potholes on the main road are causing daily accidents.",
            "Open manhole on the generic pavement is a severe hazard.",
            "Public park is poorly maintained and overgrown.",
            "Illegal parking is blocking the pedestrian crossing.",
            "Traffic signals are malfunctioning at the intersection.",
            "Stray dogs are attacking pedestrians in sector 4.",
            "Sewage pipes are leaking onto the main streets."
        ]
        
        civic_pairs = []
        
        # Exact Translation Pairs (Score ~ 1.0)
        # E.g., EN: "Pothole is dangerous" -> HI: "Pothole is dangerous"
        for text in base_complaints:
            hi_text = self.translate_en_to_hi([text])[0]
            civic_pairs.append({
                "sentence_1": text,
                "sentence_2": hi_text,
                "type": "civic_exact_duplicate"
            })
            
        # Hard negative pairs (Different complaints, Score ~ 0.0 to 0.4)
        # E.g., EN: "Garbage dump" -> HI: "Broken streetlight"
        for _ in range(target_pairs - len(base_complaints)):
            en_text = random.choice(base_complaints)
            diff_text = random.choice([c for c in base_complaints if c != en_text])
            hi_text = self.translate_en_to_hi([diff_text])[0]
            civic_pairs.append({
                "sentence_1": en_text,
                "sentence_2": hi_text,
                "type": "civic_different"
            })
            
        return civic_pairs

    def process_stsb(self, num_samples=800):
        """Loads the GLUE STSb dataset, extracts pairs, and translates one side to Hindi."""
        print("Loading STSb dataset from HuggingFace...")
        dataset = load_dataset("glue", "stsb", split="train")
        
        # Shuffle and take a stable subset
        dataset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))
        
        stsb_pairs = []
        batch_size = 16
        
        print(f"Translating {num_samples} STSb pairs to Hindi using {self.device}...")
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            en_sentences1 = batch['sentence1']
            
            # STSb comes in (English, English) pairs. We translate sentence 2 to Hindi.
            # Thus forming (English, Hindi) parallel STS pairs.
            en_sentences2 = batch['sentence2'] 
            hi_sentences2 = self.translate_en_to_hi(en_sentences2)
            
            for j in range(len(en_sentences1)):
                stsb_pairs.append({
                    "sentence_1": en_sentences1[j],
                    "sentence_2": hi_sentences2[j],
                    "type": "stsb_crosslingual"
                })
            
            if (i + batch_size) % 100 == 0:
                print(f"  Translated {min(i + batch_size, len(dataset))}/{len(dataset)} pairs...")
                
        return stsb_pairs

    def label_and_save(self, all_pairs, output_csv="indic_sts_training.csv"):
        """Computes true Cosine Similarities for the mixed dataset and exports to CSV."""
        print(f"Labeling similarities for {len(all_pairs)} pairs (0.0 to 1.0)...")
        results = []
        
        batch_size = 32
        for i in range(0, len(all_pairs), batch_size):
            batch = all_pairs[i:i+batch_size]
            s1_list = [p["sentence_1"] for p in batch]
            s2_list = [p["sentence_2"] for p in batch]
            
            # Encode EN and HI texts
            emb1 = self.similarity_model.encode(s1_list, convert_to_tensor=True)
            emb2 = self.similarity_model.encode(s2_list, convert_to_tensor=True)
            
            # Compute parallel batched cosine similarities
            cosine_scores = util.pytorch_cos_sim(emb1, emb2)
            
            # Extract the diagonal (the matching pairs)
            for j, score in enumerate(torch.diagonal(cosine_scores)):
                val = round(max(0.0, min(1.0, score.item())), 4)
                results.append({
                    "sentence_1": s1_list[j],
                    "sentence_2": s2_list[j],
                    "similarity_score": val,
                    "source": batch[j]["type"]
                })
                
        # Export
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"Pipeline Complete: Labeled {len(df)} pairs and saved to {output_csv}.")


if __name__ == "__main__":
    print("--- STS Dataset Generation Pipeline for IndicSBERT ---")
    
    generator = STSDatasetGenerator(
        translation_model_name="Helsinki-NLP/opus-mt-en-hi",
        sbert_model_name="sentence-transformers/LaBSE"
    )
    
    # 1. Translate exactly 800 general semantic pairs from STSb benchmark
    stsb_data = generator.process_stsb(num_samples=800)
    
    # 2. Augment exactly 200 civic-domain pairs manually shifted for edge cases
    civic_data = generator.generate_augmented_civic_pairs(target_pairs=200)
    
    # Combine to equal ~1000 items
    all_data = stsb_data + civic_data
    
    # Label natively via LaBSE and save
    generator.label_and_save(all_data, output_csv="indic_sts_training.csv")
