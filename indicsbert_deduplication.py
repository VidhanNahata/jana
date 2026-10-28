import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from geopy.distance import geodesic
import json

class SemanticDeduplicator:
    """
    NLP module that seamlessly evaluates complaints across Indic languages 
    and deduplicates them based on semantic meaning and geographical constraints.
    """
    def __init__(self, model_name='l3cube-pune/indic-sentence-bert-nli'):
        print(f"Loading IndicSBERT architecture: '{model_name}'...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the HuggingFace L3Cube natively scaled parameter blocks
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"Model initialized safely on bounds: {self.device.type.upper()}")

    def find_duplicates(self, new_complaint_text: str, new_lat: float, new_lng: float, existing_complaints_df: pd.DataFrame):
        """
        Cross-validates an incoming complaint string against an entire DataFrame.
        Criteria for Duplication Rule:
            1. IndicSBERT Cosine Similarity > 0.85
            2. Geodesic distance <= 500 meters
        Returns the top-3 closest existing matches logically mapped.
        """
        if existing_complaints_df.empty:
            return {"is_duplicate_overall": False, "top_matches": []}
            
        # 1. Encode Target Text (Generate exactly 1 vector [1, 768])
        new_emb = self.model.encode(new_complaint_text, convert_to_tensor=True)
        
        # 2. Encode Entire Existing Database (Yields [N, 768] matrix)
        # Note: In a true massive-scale production environment, you would cache `existing_embs` into 
        # a FAISS or ChromaDB vector store, rather than regenerating embeddings dynamically per payload!
        existing_texts = existing_complaints_df['text'].tolist()
        existing_embs = self.model.encode(existing_texts, convert_to_tensor=True)
        
        # 3. Compute massive Parallel Cosine Similarities instantaneously via PyTorch dot products
        cosine_scores = util.pytorch_cos_sim(new_emb, existing_embs)[0]
        
        # 4. Extract Top-3 Matches bounded explicitly
        top_k = min(3, len(existing_texts))
        top_results = torch.topk(cosine_scores, k=top_k)
        
        is_duplicate_flag = False
        top_matches = []
        new_coords = (new_lat, new_lng)
        
        for score_tensor, idx_tensor in zip(top_results[0], top_results[1]):
            score = score_tensor.item()
            idx = idx_tensor.item()
            
            # Map index dynamically back to exact DataFrame row
            match_row = existing_complaints_df.iloc[idx]
            match_text = match_row['text']
            match_lat = float(match_row['latitude'])
            match_lng = float(match_row['longitude'])
            
            match_coords = (match_lat, match_lng)
            
            # Execute Geopy Vincenty mapping converting WGS84 Coords natively into strict metric meters
            dist_meters = geodesic(new_coords, match_coords).meters
            
            # Check Compound Deduplication Constraints (>0.85 exact sim AND <=500m logic limit)
            is_dup = bool(score > 0.85 and dist_meters <= 500.0)
            if is_dup:
                is_duplicate_flag = True
                
            top_matches.append({
                "complaint_id": match_row.get('id', idx),
                "text": match_text,
                "similarity_score": round(score, 4),
                "distance_meters": round(dist_meters, 2),
                "flagged_duplicate": is_dup
            })
            
        return {
            "is_duplicate_overall": is_duplicate_flag,
            "top_matches": top_matches
        }

if __name__ == "__main__":
    print("--- IndicSBERT Civic Deduplication System ---")
    
    # Generate Synthetic SQL-like Database for mapping validation
    # (Notice how it contains cross-lingual equivalents of the identical problem)
    df = pd.DataFrame({
        'id': [101, 102, 103, 104],
        'text': [
            "There is a huge pothole causing immense traffic jams on MG Road.",
            "सड़क पर बहुत बड़ा गड्ढा है, जिससे ट्रैफिक रुक रहा है।",    # Direct translation/duplicate pattern in Hindi
            "Overflowing garbage on the sidewalk near the park.",
            "Pothole issue on airport road far away."             # Same issue type, but physically distant
        ],
        'latitude': [18.5204, 18.5205, 18.5210, 19.0896],
        'longitude': [73.8567, 73.8568, 73.8570, 72.8656]
    })
    
    deduplicator = SemanticDeduplicator()
    
    # New hypothetical user complaint (Roman code-mixed + English context mapping directly to ID 101/102)
    new_complaint = "MG Road par ek massive pothole hai please fix it!"
    new_lat = 18.52045
    new_lng = 73.85675
    
    print(f"\nEvaluating Input: '{new_complaint}' at Coords ({new_lat}, {new_lng})")
    
    # Infer Pipeline Execution
    res = deduplicator.find_duplicates(
        new_complaint_text=new_complaint, 
        new_lat=new_lat, 
        new_lng=new_lng, 
        existing_complaints_df=df
    )
    
    print("\n--- Pipeline Deduplication Result ---")
    print(json.dumps(res, indent=4, ensure_ascii=False))
