from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "pszemraj/led-large-book-summary"
CACHE_DIR = "~/model_weights"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

tokenizer.save_pretrained(CACHE_DIR)
model.save_pretrained(CACHE_DIR)
