from transformers import pipeline

# 1. Sentiment Analysis
print("=== SENTIMENT ANALYSIS ===")
classifier = pipeline("sentiment-analysis")
result = classifier("I love building AI applications")
print(result)
print()

# 2. Zero-Shot Classification
# (summarization pipeline was removed in transformers v5.x)
print("=== ZERO-SHOT CLASSIFICATION ===")
classifier_zs = pipeline("zero-shot-classification")
result = classifier_zs(
    "I want to learn how to build neural networks",
    candidate_labels=["education", "business", "politics", "technology"],
)
print(f"Text: '{result['sequence']}'")
print(f"Labels: {result['labels']}")
print(f"Scores: {[round(s, 4) for s in result['scores']]}")
print()

# 3. Text Generation
print("=== TEXT GENERATION ===")
generator = pipeline("text-generation", model="gpt2")
result = generator("The future of AI is", max_length=50)
print(result)
