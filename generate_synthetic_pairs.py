import pandas as pd
import random
from sentence_transformers import SentenceTransformer, util
# Note: For production, use IndicTrans2. We use googletrans here as an accessible proof-of-concept.
from googletrans import Translator
import time

# Target languages and their Google Translate codes
LANGUAGES = {
    'hi': 'Hindi',
    'ta': 'Tamil',
    'te': 'Telugu',
    'mr': 'Marathi',
    'bn': 'Bengali',
    'ml': 'Malayalam'
}

def generate_synthetic_dataset(seed_complaints, pairs_per_lang=500, output_csv="multilingual_complaints.csv"):
    translator = Translator()
    # LaBSE is exceptional for cross-lingual similarity (0 to 1 scoring)
    print("Loading LaBSE model for similarity scoring...")
    model = SentenceTransformer('sentence-transformers/LaBSE')
    
    dataset_rows = []
    
    # Process one base language against translations
    for lang_code, lang_name in LANGUAGES.items():
        print(f"\nProcessing language: {lang_name}")
        translated_complaints = []
        
        # 1. Translate seed complaints to create ground-truth pool
        for i, text in enumerate(seed_complaints):
            try:
                # Add delay to avoid API rate limits if using free APIs
                time.sleep(0.5)
                translated = translator.translate(text, dest=lang_code).text
                translated_complaints.append((text, translated))
                if i % 10 == 0:
                    print(f"  Translated {i}/{len(seed_complaints)}...")
            except Exception as e:
                print(f"  Translation error: {e}")
                
        # We need pairs_per_lang total pairs (mix of duplicates and non-duplicates)
        target_duplicates = pairs_per_lang // 2
        target_non_duplicates = pairs_per_lang - target_duplicates
        
        added_dups = 0
        added_non_dups = 0
        
        print(f"Scoring pairs for {lang_name}...")
        
        # 2. Build Duplicate Pairs
        for en_text, tl_text in translated_complaints:
            if added_dups >= target_duplicates: break
            
            # Compute similarity (should be close to 1.0)
            emb1 = model.encode(en_text, convert_to_tensor=True)
            emb2 = model.encode(tl_text, convert_to_tensor=True)
            sim_score = util.pytorch_cos_sim(emb1, emb2).item()
            
            dataset_rows.append({
                "complaint_1": en_text,
                "complaint_2": tl_text,
                "language_1": "English",
                "language_2": lang_name,
                "is_duplicate": 1,
                "similarity_score": round(max(0.0, sim_score), 4) # bound it cleanly
            })
            added_dups += 1
            
        # 3. Build Non-Duplicate Pairs
        while added_non_dups < target_non_duplicates:
            # Pick two random DIFFERENT complaints
            idx1, idx2 = random.sample(range(len(translated_complaints)), 2)
            en_text = translated_complaints[idx1][0]
            tl_text = translated_complaints[idx2][1] # A different complaint's translation
            
            emb1 = model.encode(en_text, convert_to_tensor=True)
            emb2 = model.encode(tl_text, convert_to_tensor=True)
            sim_score = util.pytorch_cos_sim(emb1, emb2).item()
            
            dataset_rows.append({
                "complaint_1": en_text,
                "complaint_2": tl_text,
                "language_1": "English",
                "language_2": lang_name,
                "is_duplicate": 0,
                "similarity_score": round(max(0.0, min(1.0, sim_score)), 4)
            })
            added_non_dups += 1
            
    # Export securely to CSV
    df = pd.DataFrame(dataset_rows)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\nPipeline complete! Saved {len(df)} pairs to {output_csv}.")

if __name__ == "__main__":
    # Example Seed Complaints Toolkit (Extract these from the Kaggle BMC dataset)
    sample_seeds = [
        "There is a huge pothole on MG Road that is causing traffic jams.",
        "Garbage has not been collected from sector 4 for the past three days.",
        "The streetlight near the main intersection is broken and flickering.",
        "A water pipe burst near the residential area, flooding the entire street.",
        "Open manhole on the generic pavement is a severe hazard for pedestrians."
    ]
    
    # Warning: For production with 500+ pairs per language, ensure you are using 
    # bulk translation APIs (like IndicTrans2) instead of the rate-limited googletrans.
    print("Initiating Civic Dataset Synthetic Generation Pipeline...")
    generate_synthetic_dataset(seed_complaints=sample_seeds, pairs_per_lang=10, output_csv="multilingual_complaints.csv")
