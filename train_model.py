import os
import random
import pickle
import torch
import torch.optim as optim
import torch.nn.functional as F

from model_definition import *
from embeddings import generate_universal_sentence_embeddings  # your embedding function

# -------------------------------
# Config
# -------------------------------
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

USE_EXPERT_CURRICULUM = True
CURRICULUM_MIX_RATIO = 0.2

BATCH_SIZE = 16
EPOCHS = 5
ALPHA = 0.31

random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# Load samples from pickle
# -------------------------------
data_dir = "datasets"

all_samples = []
for fname in os.listdir(data_dir):
    if fname.endswith(".pkl"):
        with open(os.path.join(data_dir, fname), "rb") as f:
            all_samples.extend(pickle.load(f))

print(f"Loaded {len(all_samples)} samples.")

# -------------------------------
# Load or generate embeddings
# -------------------------------
embedding_cache_path = "all_embeddings_cache.pkl"

if os.path.exists(embedding_cache_path):
    print(f"Loading embeddings from {embedding_cache_path}")
    with open(embedding_cache_path, "rb") as f:
        embedding_cache = pickle.load(f)
else:
    print(f"No embeddings cache found. Computing universal embeddings...")
    embedding_cache = generate_universal_sentence_embeddings(data_dir, cache_path=embedding_cache_path, device=device)

print("Embeddings ready.")

# -------------------------------
# Dataset split (unseen sentence split)
# -------------------------------
sentences = list({s["sentence"] for s in all_samples})
random.shuffle(sentences)
n = len(sentences)
train_s = set(sentences[:int(TRAIN_RATIO*n)])
val_s   = set(sentences[int(TRAIN_RATIO*n):int((TRAIN_RATIO+VAL_RATIO)*n)])
test_s  = set(sentences[int((TRAIN_RATIO+VAL_RATIO)*n):])

train_samples = [s for s in all_samples if s["sentence"] in train_s]
val_samples   = [s for s in all_samples if s["sentence"] in val_s]
test_samples  = [s for s in all_samples if s["sentence"] in test_s]

print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

# -------------------------------
# Curriculum logic for experts
# -------------------------------
expert_datasets = {}
for s in train_samples:
    eid = LANG_TO_EXPERT.get(s["lang"].lower(), 0)
    expert_datasets.setdefault(eid, []).append(s)

def sample_curriculum(epoch):
    target = epoch % len(expert_datasets)
    main = expert_datasets[target]
    others = [s for eid, data in expert_datasets.items() if eid != target for s in data]
    mix_n = int(len(main) * CURRICULUM_MIX_RATIO)
    mix = random.sample(others, min(mix_n, len(others)))
    data = main + mix
    random.shuffle(data)
    print(f"Curriculum expert {target}: {len(main)} + {len(mix)}")
    return data

# -------------------------------
# Collate wrapper using universal cache
# -------------------------------
def safe_collate(batch):
    return collate_batch(batch, embedding_cache, device=device)

# -------------------------------
# Training & evaluation
# -------------------------------
def train_epoch(model, samples, optimizer):
    model.train()
    total_loss = 0
    steps = 0

    for i in range(0, len(samples), BATCH_SIZE):
        batch = samples[i:i+BATCH_SIZE]
        collated = safe_collate(batch)
        if collated is None:
            continue

        inputs, fix_inputs, fix_targets, dur_targets, full_word_embeddings, lengths, expert_ids = collated

        optimizer.zero_grad()
        logits, dur_pred = model(inputs, fix_inputs, full_word_embeddings, lengths, expert_ids)

        loss_scan = F.cross_entropy(logits.view(-1, logits.size(-1)), fix_targets.view(-1), ignore_index=-1)
        mask = fix_targets != -1
        if not mask.any():
            continue
        loss_dur = F.mse_loss(dur_pred[mask], dur_targets[mask])
        loss = ALPHA * loss_scan + (1 - ALPHA) * loss_dur
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    return total_loss / max(steps, 1)

def evaluate(model, samples):
    model.eval()
    total_loss = 0
    steps = 0

    with torch.no_grad():
        for i in range(0, len(samples), BATCH_SIZE):
            batch = samples[i:i+BATCH_SIZE]
            collated = safe_collate(batch)
            if collated is None:
                continue

            inputs, fix_inputs, fix_targets, dur_targets, full_word_embeddings, lengths, expert_ids = collated
            logits, dur_pred = model(inputs, fix_inputs, full_word_embeddings, lengths, expert_ids)

            loss_scan = F.cross_entropy(logits.view(-1, logits.size(-1)), fix_targets.view(-1), ignore_index=-1)
            mask = fix_targets != -1
            if not mask.any():
                continue
            loss_dur = F.mse_loss(dur_pred[mask], dur_targets[mask])
            loss = ALPHA * loss_scan + (1 - ALPHA) * loss_dur
            total_loss += loss.item()
            steps += 1

    return total_loss / max(steps, 1)

# -------------------------------
# Model & optimizer
# -------------------------------
model = EyeExpertM(
    hidden_dim=256,
    encoder_dim=768,
    n_experts=5,
    max_seq_len=200,
    window_size=4,
    n_layers=2,
    attention_type="dot",
).to(device)

optimizer = optim.Adam(model.parameters(), lr=3e-4)

# -------------------------------
# Training loop
# -------------------------------
os.makedirs("checkpoints", exist_ok=True)

for epoch in range(EPOCHS):
    print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
    train_data = sample_curriculum(epoch) if USE_EXPERT_CURRICULUM else train_samples
    train_loss = train_epoch(model, train_data, optimizer)
    val_loss = evaluate(model, val_samples)
    test_loss = evaluate(model, test_samples)
    print(f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | Test: {test_loss:.4f}")
    torch.save(model.state_dict(), f"checkpoints/eyeexpert_epoch{epoch+1}.pt")