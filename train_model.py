# ------------------------------- Imports -------------------------------
import os
import random
import pickle
import torch
import torch.optim as optim
from tqdm import tqdm
from embeddings import generate_universal_sentence_embeddings, normalize_sentence
from model_definition import EyeExpertM, collate_batch, LANG_TO_EXPERT, PAD_IDX

# ------------------------------- Config -------------------------------
DATA_DIR = "datasets"
EMBEDDING_CACHE = "full_sentence_embeddings.pkl"

TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

LEAVE_OUT_LANGUAGE = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
EPOCHS = 3
ALPHA = 0.31

USE_EXPERT_CURRICULUM = True
CURRICULUM_MIX_RATIO = 0.2

random.seed(42)
os.makedirs("checkpoints", exist_ok=True)

# ------------------------------- Step 1: Load dataset -------------------------------
all_samples = []
for fname in os.listdir(DATA_DIR):
    if fname.endswith(".pkl"):
        with open(os.path.join(DATA_DIR, fname), "rb") as f:
            samples = pickle.load(f)
            all_samples.extend(samples)
all_samples = [s for s in all_samples if s.get("sentence") is not None]
print(f"Loaded {len(all_samples)} samples.")

# ------------------------------- Step 2: Load or generate embeddings -------------------------------
if not os.path.exists(EMBEDDING_CACHE):
    print("No embeddings found. Generating embeddings...")
    generate_universal_sentence_embeddings(
        pickle_dir=DATA_DIR,
        cache_path=EMBEDDING_CACHE,
        batch_size=4,
        device=DEVICE,
        max_sentences=None
    )

with open(EMBEDDING_CACHE, "rb") as f:
    embedding_cache = pickle.load(f)
print(f"✅ Loaded embedding cache ({len(embedding_cache)})")

# ------------------------------- Step 3: Dataset splitting -------------------------------
def split_dataset(samples):
    if LEAVE_OUT_LANGUAGE:
        train_samples = [s for s in samples if s["lang"] != LEAVE_OUT_LANGUAGE]
        heldout = [s for s in samples if s["lang"] == LEAVE_OUT_LANGUAGE]
        n = len(heldout)
        val_samples = heldout[:n//2]
        test_samples = heldout[n//2:]
        return train_samples, val_samples, test_samples

    sentences = list({s["sentence"] for s in samples})
    random.shuffle(sentences)
    n = len(sentences)
    train_s = set(sentences[:int(TRAIN_RATIO * n)])
    val_s   = set(sentences[int(TRAIN_RATIO * n):int((TRAIN_RATIO + VAL_RATIO) * n)])
    test_s  = set(sentences[int((TRAIN_RATIO + VAL_RATIO) * n):])
    train_samples = [s for s in samples if s["sentence"] in train_s]
    val_samples   = [s for s in samples if s["sentence"] in val_s]
    test_samples  = [s for s in samples if s["sentence"] in test_s]
    return train_samples, val_samples, test_samples

train_samples, val_samples, test_samples = split_dataset(all_samples)
print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

# ------------------------------- Step 4: Expert datasets -------------------------------
expert_datasets = {}
for s in train_samples:
    lang = s["lang"].lower()
    eid = LANG_TO_EXPERT.get(lang)
    if eid is not None:
        expert_datasets.setdefault(eid, []).append(s)

def sample_curriculum_dataset(epoch, expert_datasets, mix_ratio=0.2):
    target_expert = epoch % len(expert_datasets)
    target_samples = expert_datasets[target_expert]
    other_samples = [s for eid, data in expert_datasets.items() if eid != target_expert for s in data]
    mix_size = int(len(target_samples) * mix_ratio)
    mixed_samples = random.sample(other_samples, min(mix_size, len(other_samples)))
    dataset = target_samples + mixed_samples
    random.shuffle(dataset)
    print(f"Curriculum epoch expert: {target_expert}, Samples: {len(dataset)}")
    return dataset, target_expert

# ------------------------------- Step 5: Model -------------------------------
max_fix_idx = max([max(s["scanpath"]) for s in all_samples])
embedding_size = max_fix_idx + 1

model = EyeExpertM(
    hidden_dim=256,
    encoder_dim=768,
    n_experts=5,
    max_seq_len=embedding_size,
    window_size=4,
    n_layers=2,
    attention_type="dot"
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=3e-4)

# ------------------------------- Step 6: Train / Evaluate -------------------------------
def train_epoch(model, samples, optimizer, batch_size=BATCH_SIZE, alpha=ALPHA, device=DEVICE, expert_id=0, embedding_cache=None):
    model.train()
    total_loss = 0.0
    n_batches = max(1, len(samples) // batch_size)
    pbar = tqdm(range(0, len(samples), batch_size), desc="Training", unit="batch")

    for i in pbar:
        batch_samples = samples[i:i+batch_size]
        batch = collate_batch(batch_samples, device=device, embedding_cache=embedding_cache)
        if batch is None:
            continue

        inputs, fix_targets, dur_targets, full_word_embeddings, lengths, pad_idx = batch

        optimizer.zero_grad()
        logits, dur_pred = model(inputs, fix_targets, full_word_embeddings, lengths, expert_id)
        loss_fix = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), fix_targets.view(-1), ignore_index=pad_idx
        )
        loss_dur = torch.nn.functional.mse_loss(dur_pred.view(-1), dur_targets.view(-1))
        loss = alpha * loss_fix + (1 - alpha) * loss_dur
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({"batch_loss": total_loss / ((i // batch_size) + 1)})

    return total_loss / n_batches

def evaluate(model, samples, batch_size=BATCH_SIZE, alpha=ALPHA, device=DEVICE, expert_id=0, embedding_cache=None, desc="Evaluating"):
    model.eval()
    total_loss = 0.0
    n_batches = max(1, len(samples) // batch_size)
    pbar = tqdm(range(0, len(samples), batch_size), desc=desc, unit="batch")
    with torch.no_grad():
        for i in pbar:
            batch_samples = samples[i:i+batch_size]
            batch = collate_batch(batch_samples, device=device, embedding_cache=embedding_cache)
            if batch is None:
                continue

            inputs, fix_targets, dur_targets, full_word_embeddings, lengths, pad_idx = batch
            logits, dur_pred = model(inputs, fix_targets, full_word_embeddings, lengths, expert_id)
            loss_fix = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), fix_targets.view(-1), ignore_index=pad_idx
            )
            loss_dur = torch.nn.functional.mse_loss(dur_pred.view(-1), dur_targets.view(-1))
            loss = alpha * loss_fix + (1 - alpha) * loss_dur
            total_loss += loss.item()
            pbar.set_postfix({"batch_loss": total_loss / ((i // batch_size) + 1)})

    return total_loss / n_batches

# ------------------------------- Step 7: Training loop -------------------------------
for epoch in range(EPOCHS):
    print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
    if USE_EXPERT_CURRICULUM:
        epoch_dataset, epoch_expert = sample_curriculum_dataset(epoch, expert_datasets, CURRICULUM_MIX_RATIO)
    else:
        epoch_dataset, epoch_expert = train_samples, 0

    train_loss = train_epoch(model, epoch_dataset, optimizer, expert_id=epoch_expert, embedding_cache=embedding_cache)
    val_loss = evaluate(model, val_samples, expert_id=epoch_expert, embedding_cache=embedding_cache, desc="Validation")
    test_loss = evaluate(model, test_samples, expert_id=epoch_expert, embedding_cache=embedding_cache, desc="Test")
    print(f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | Test: {test_loss:.4f}")

    torch.save(model.state_dict(), f"checkpoints/eyeexpert_epoch{epoch+1}.pt")
    print("✅ Saved checkpoint")