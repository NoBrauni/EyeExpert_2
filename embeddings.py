import os
import pickle
import torch
import string
from transformers import XLMRobertaTokenizer, XLMRobertaModel

# -------------------------------
# Initialize encoder/tokenizer
# -------------------------------
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
encoder = XLMRobertaModel.from_pretrained("xlm-roberta-base")
encoder.eval()
for param in encoder.parameters():
    param.requires_grad = False

# -------------------------------
# Function to generate full-sentence embeddings

def generate_universal_sentence_embeddings(pickle_dir, cache_path="full_sentence_embeddings.pkl",
                                           batch_size=16, device="cpu"):
    """
    Reads all dataset pickles and generates a single universal embedding cache.
    Does NOT modify the dataset files.
    """
    if os.path.exists(cache_path):
        print(f"✅ Loading existing embedding cache from {cache_path}")
        with open(cache_path, "rb") as f:
            sentence_cache = pickle.load(f)
    else:
        sentence_cache = {}

    # Iterate over all dataset pickles
    for fname in os.listdir(pickle_dir):
        if not fname.endswith(".pkl"):
            continue

        path = os.path.join(pickle_dir, fname)
        with open(path, "rb") as f:
            samples = pickle.load(f)

        # Sentences needing embeddings
        sentences_to_compute = list({s["sentence"] for s in samples if s["sentence"] not in sentence_cache})
        print(f"{fname}: {len(sentences_to_compute)} sentences need embeddings")

        for start in range(0, len(sentences_to_compute), batch_size):
            batch_sentences = sentences_to_compute[start:start + batch_size]

            # Tokenize
            encoding = tokenizer(
                batch_sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_offsets_mapping=True
            )
            offsets = encoding.pop("offset_mapping")
            encoding = {k: v.to(device) for k, v in encoding.items()}

            # Encode
            with torch.no_grad():
                outputs = encoder(**encoding)
            hidden = outputs.last_hidden_state  # [B, seq_len, hidden_dim]

            # Align tokens to words
            for i, sentence in enumerate(batch_sentences):
                token_embeds = hidden[i]
                token_offsets = offsets[i]

                words = sentence.split()
                word_vectors = []

                for word in words:
                    word_clean = word.strip(string.punctuation).lower()
                    subword_vectors = []

                    for tok_i, (start_c, end_c) in enumerate(token_offsets):
                        token_text = sentence[start_c:end_c].strip(string.punctuation).lower()
                        if token_text == word_clean:
                            subword_vectors.append(token_embeds[tok_i])

                    # fallback to first token if not matched
                    if subword_vectors:
                        word_vectors.append(torch.stack(subword_vectors).mean(dim=0))
                    else:
                        word_vectors.append(token_embeds[0])

                sentence_cache[sentence] = torch.stack(word_vectors)

            print(f"Processed {start + len(batch_sentences)}/{len(sentences_to_compute)} sentences")

    # Save universal cache
    with open(cache_path, "wb") as f:
        pickle.dump(sentence_cache, f)
    print(f"✅ Saved universal embedding cache at {cache_path}")