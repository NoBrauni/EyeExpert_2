import torch
import pickle
import os
import string
from transformers import XLMRobertaModel, XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
encoder = XLMRobertaModel.from_pretrained("xlm-roberta-base")
encoder.eval()
for param in encoder.parameters():
    param.requires_grad = False

def add_embeddings_to_pickles_fixated_words(pickle_dir, cache_path="embedding_cache.pkl", batch_size=16, device="cpu"):
    # Load existing cache if present
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        with open(cache_path, "rb") as f:
            sentence_cache = pickle.load(f)
    else:
        sentence_cache = {}

    # Iterate over pickle files
    for fname in os.listdir(pickle_dir):
        if not fname.endswith(".pkl"):
            continue

        path = os.path.join(pickle_dir, fname)
        with open(path, "rb") as f:
            samples = pickle.load(f)

        # Identify sentences that need embeddings
        sentences_to_compute = [s["sentence"] for s in samples if s["sentence"] not in sentence_cache]
        sentences_to_compute = list(set(sentences_to_compute))
        print(f"{fname}: {len(sentences_to_compute)} sentences need embeddings")

        # Compute embeddings in batches
        for start in range(0, len(sentences_to_compute), batch_size):
            batch_sentences = sentences_to_compute[start:start+batch_size]
            encoding = tokenizer(
                batch_sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_offsets_mapping=True
            )

            offsets = encoding["offset_mapping"]
            encoding = {k: v.to(device) for k, v in encoding.items() if k != "offset_mapping"}

            with torch.no_grad():
                outputs = encoder(**encoding)

            hidden = outputs.last_hidden_state  # [B, seq_len, hidden_dim]

            # Assign embeddings to fixated words
            for i, sentence in enumerate(batch_sentences):
                token_embeds = hidden[i]  # [seq_len, hidden_dim]
                token_offsets = offsets[i]
                samples_for_sentence = [s for s in samples if s["sentence"] == sentence]

                for sample in samples_for_sentence:
                    word_vectors = []
                    for word in sample["words"]:  # only fixated words
                        word_clean = word.strip(string.punctuation).lower()
                        subword_vectors = []
                        for tok_i, (start_c, end_c) in enumerate(token_offsets):
                            token_text = sentence[start_c:end_c].strip(string.punctuation).lower()
                            if token_text == word_clean:
                                subword_vectors.append(token_embeds[tok_i])
                        if subword_vectors:
                            word_vectors.append(torch.stack(subword_vectors).mean(dim=0))
                        else:
                            # fallback: first token embedding
                            word_vectors.append(token_embeds[0])
                    # Convert to tensor of shape [num_fixated_words, hidden_dim]
                    sentence_cache[sentence] = torch.stack(word_vectors)

        # Assign embeddings to all samples in this pickle
        for sample in samples:
            sample["word_embeddings"] = sentence_cache[sample["sentence"]]

        # Overwrite pickle
        with open(path, "wb") as f:
            pickle.dump(samples, f)
        print(f"Updated {len(samples)} samples with embeddings in {fname}")

    # Save cache
    with open(cache_path, "wb") as f:
        pickle.dump(sentence_cache, f)
    print(f"Saved embedding cache to {cache_path}")