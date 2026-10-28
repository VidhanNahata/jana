import re
import torch
from transformers import AutoTokenizer
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

class IndicBERTPreprocessor:
    def __init__(self, model_name='ai4bharat/indic-bert'):
        """
        Initializes the preprocessor with the pretrained IndicBERT ALBERT/SentencePiece tokenizer.
        """
        print(f"Loading '{model_name}' tokenizer (SentencePiece-based)...")
        # IndicBERT uses an AlbertTokenizer which behaves as a wrapper over SentencePiece
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.normalizer_factory = IndicNormalizerFactory()

    def normalize_unicode(self, text, lang_code='hi'):
        """
        Normalizes Unicode for Indic scripts to ensure consistent byte representations 
        (e.g., handling Nuktas, splitting half-characters).
        Common target codes: 'hi' (Hindi), 'ta' (Tamil), 'te' (Telugu), 'mr' (Marathi), 'bn' (Bengali), 'ml' (Malayalam)
        """
        # If English/Code-mixed fallback, just return text
        if lang_code == 'en':
            return text
            
        try:
            # Fetch the dedicated language normalizer dynamically
            normalizer = self.normalizer_factory.get_normalizer(lang_code)
            return normalizer.normalize(text)
        except Exception as e:
            # Fallback for unsupported codes
            return text

    def clean_text(self, text):
        """
        Removes special characters but carefully preserves:
        - English alphanumeric strings (for Roman script code-mixing)
        - Devanagari (Hindi, Marathi)
        - Tamil
        - Telugu
        - Bengali
        - Malayalam
        - Basic grammatical punctuation
        """
        # Unicode Hex Ranges Map:
        # \u0900-\u097F (Devanagari)
        # \u0B80-\u0BFF (Tamil)
        # \u0C00-\u0C7F (Telugu)
        # \u0980-\u09FF (Bengali)
        # \u0D00-\u0D7F (Malayalam)
        
        # Regex keeps the above ranges, alphanumeric chars, whitespace, and minimal punctuation [.,!?]
        allowed_pattern = re.compile(r'[^\w\s\u0900-\u097F\u0B80-\u0BFF\u0C00-\u0C7F\u0980-\u09FF\u0D00-\u0D7F\.,!?]+', re.UNICODE)
        
        # Replace non-allowed chars with a space
        cleaned_text = allowed_pattern.sub(' ', text)
        
        # Collapse multiple structural spaces into single spaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text

    def preprocess(self, texts, lang_codes=None, max_length=128):
        """
        End-to-End Pipeline: Normalize -> Clean -> SentencePiece Tokenize -> Output Tensors.
        Returns huggingface token encoding dictionary populated with PyTorch tensors ('pt').
        """
        if isinstance(texts, str):
            texts = [texts]
            lang_codes = [lang_codes] if lang_codes else ['hi']
        elif not lang_codes:
            lang_codes = ['hi'] * len(texts)

        processed_texts = []
        for text, lang in zip(texts, lang_codes):
            # 1. Unicode Normalization
            norm_text = self.normalize_unicode(text, lang_code=lang)
            # 2. String Cleaning
            clean_txt = self.clean_text(norm_text)
            processed_texts.append(clean_txt)

        # 3. BPEN/SentencePiece Tokenization mapping to tensor arrays
        encodings = self.tokenizer.batch_encode_plus(
            processed_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        return encodings
        
if __name__ == "__main__":
    # Test cases mapping standard scripts, code-mixing, and malicious symbols
    test_complaints = [
        "This pothole is dangerous! कृपया इसे जल्दी ठीक करें।",  # Hindi script + English mix
        "veedhi deepalu pani cheyatledu, it is very dark.",       # Telugu code-mixed Roman + English
        "குடிநீர் குழாய் உடைந்துள்ளது #fixit @mayor",              # Tamil + Noisy symbols
        "Garbage hasn't been collected for 3 days!!! $^%&*_"      # English + pure noise
    ]
    lang_codes = ['hi', 'te', 'ta', 'en']
    
    print("Initializing Pipeline...")
    preprocessor = IndicBERTPreprocessor()
    
    print("\n--- Pipeline Input ---")
    for cp in test_complaints:
        print(f" -> {cp}")
        
    tensors = preprocessor.preprocess(test_complaints, lang_codes=lang_codes)
    
    print("\n--- Output PyTorch Tensors (Ready for Fine-Tuning) ---")
    print("Input IDs shape:", tensors['input_ids'].shape)
    print("Attention Mask shape:", tensors['attention_mask'].shape)
    
    print("\n--- Subword Tokenization Example (SentencePiece mapping) ---")
    # Show how SentencePiece sub-word mechanism breaks down the complex string
    tokens = preprocessor.tokenizer.convert_ids_to_tokens(tensors['input_ids'][0])
    
    # Filter padding characters out of the view
    filtered_tokens = [tok for tok in tokens if tok not in ['[PAD]', '<pad>']]
    print(filtered_tokens)
