from sentence_transformers import SentenceTransformer

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# The sentences to encode
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
    "The weather is lovely today.",
    "The weather was hot yesterday.",
]

# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(sentences)
print('embedding')
print(embeddings.shape)
# print(embeddings[0])
# [4, 384]

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
print('similarities')
print(similarities)
