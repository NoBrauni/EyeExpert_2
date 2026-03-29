from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn as nn

PAD_IDX = 0

# -------------------------------
# Language → Expert mapping
# -------------------------------
LANG_TO_EXPERT = {
    # Expert 0 — Germanic
    "en": 0, "en_uk": 0, "du": 0, "ge": 0, "ge_po": 0, "ge_zu": 0,
    # Expert 1 — Nordic
    "no": 1, "da": 1, "ic": 1,
    # Expert 2 — Romance
    "sp": 2, "sp_ch": 2, "it": 2, "bp": 2,
    # Expert 3 — Slavic
    "ru": 3, "ru_mo": 3, "se": 3,
    # Expert 4 — Uralic
    "fi": 4, "ee": 4,
}

# -------------------------------
# Collate function for batching
# -------------------------------
def collate_batch(samples, embedding_cache, device="cpu"):
    """
    Prepares a batch for EyeExpertM using a shared embeddings cache.
    - samples: list of dataset dicts (no embeddings inside)
    - embedding_cache: dict {sentence: tensor[num_words, hidden_dim]}
    """
    batch_inputs, fix_inputs, fix_targets, dur_targets = [], [], [], []
    lengths, full_word_embeddings, expert_ids = [], [], []

    for sample in samples:
        # Fetch embeddings from the cache
        word_embeddings = embedding_cache[sample["sentence"]].to(device)  # [num_words, hidden_dim]
        fix_seq = torch.tensor(sample["scanpath"], dtype=torch.long, device=device)
        dur_seq = torch.tensor(sample["durations"], dtype=torch.float, device=device)
        lang = sample["lang"].lower().strip()

        if len(fix_seq) <= 1:
            continue

        # Shift for RNN input/target
        fix_input = fix_seq[:-1] + 1
        fix_target = fix_seq[1:] + 1
        dur_target = dur_seq[1:]

        # Mask invalid indices
        num_words = word_embeddings.size(0) + 1
        mask = (fix_input > 0) & (fix_input < num_words)
        fix_input, fix_target, dur_target = fix_input[mask], fix_target[mask], dur_target[mask]

        if len(fix_input) == 0:
            continue

        # Gather embeddings for fixated words
        batch_inputs.append(word_embeddings[fix_input - 1])
        fix_inputs.append(fix_input)
        fix_targets.append(fix_target)
        dur_targets.append(dur_target)
        lengths.append(len(fix_input))
        full_word_embeddings.append(word_embeddings)
        expert_ids.append(LANG_TO_EXPERT.get(lang, 0))

    if not batch_inputs:
        return None

    # Padding sequences
    padded_inputs = pad_sequence(batch_inputs, batch_first=True)
    padded_fix_inputs = pad_sequence(fix_inputs, batch_first=True, padding_value=PAD_IDX)
    padded_fix_targets = pad_sequence(fix_targets, batch_first=True, padding_value=-1)
    padded_durs = pad_sequence(dur_targets, batch_first=True)
    padded_full_words = pad_sequence(full_word_embeddings, batch_first=True)

    return (
        padded_inputs.to(device),
        padded_fix_inputs.to(device),
        padded_fix_targets.to(device),
        padded_durs.to(device),
        padded_full_words.to(device),
        lengths,
        torch.tensor(expert_ids, dtype=torch.long, device=device),
    )

# -------------------------------
# EyeExpertM Model
# -------------------------------
class EyeExpertM(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        encoder_dim=768,
        n_experts=5,
        max_seq_len=200,
        n_layers=1,
        dropout=0.1,
        attention_type="dot",
        window_size=8
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder_dim = encoder_dim
        self.n_experts = n_experts
        self.window_size = window_size
        self.attention_type = attention_type

        # Experts
        self.experts = nn.ModuleList([
            nn.GRU(
                input_size=encoder_dim + 32,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                dropout=dropout if n_layers > 1 else 0,
                batch_first=True
            ) for _ in range(n_experts)
        ])

        # Scanpath embedding
        self.scanpath_embedding = nn.Embedding(max_seq_len + 1, 32, padding_idx=PAD_IDX)

        # Output projection
        self.output_layer = nn.Linear(hidden_dim, encoder_dim)

        # Duration prediction
        self.duration_layer = nn.Linear(hidden_dim, 1)

    def forward(self, inputs, fix_inputs, full_word_embeddings, lengths, expert_ids):
        device = inputs.device

        # Scanpath embeddings
        scanpath_embeds = self.scanpath_embedding(fix_inputs)
        rnn_inputs = torch.cat([inputs, scanpath_embeds], dim=-1)

        outputs_all = []

        # Route per expert
        for expert_id in range(self.n_experts):
            mask = (expert_ids.to(device) == expert_id)
            if mask.sum() == 0:
                continue

            expert = self.experts[expert_id]
            rnn_in = rnn_inputs[mask]
            idxs = mask.nonzero(as_tuple=True)[0].tolist()
            lens = torch.tensor([lengths[i] for i in idxs], dtype=torch.long, device=device)

            packed = pack_padded_sequence(rnn_in, lens, batch_first=True, enforce_sorted=False)
            out, _ = expert(packed)
            out, _ = pad_packed_sequence(out, batch_first=True, total_length=fix_inputs.size(1))

            outputs_all.append((mask, out))

        # Reconstruct batch
        outputs = torch.zeros(rnn_inputs.size(0), rnn_inputs.size(1), self.hidden_dim,
                              device=device, dtype=rnn_inputs.dtype)
        for mask, out in outputs_all:
            outputs[mask] = out

        # Attention over full sentence
        proj = self.output_layer(outputs)
        logits = torch.matmul(proj, full_word_embeddings.transpose(1, 2))

        # Saccade mask
        cur_fix = fix_inputs.unsqueeze(-1)
        word_positions = torch.arange(full_word_embeddings.size(1), device=device).view(1, 1, -1)
        distance = torch.abs(word_positions - (cur_fix - 1))
        saccade_mask = distance <= self.window_size
        logits = logits.masked_fill(~saccade_mask, -1e9)

        # Word mask (padding)
        word_mask = (full_word_embeddings.abs().sum(-1) != 0)
        logits = logits.masked_fill(~word_mask.unsqueeze(1), -1e9)

        # Duration prediction
        dur_pred = self.duration_layer(outputs).squeeze(-1)

        return logits, dur_pred