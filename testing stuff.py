import pyreadr
import pandas as pd
from rapidfuzz import process, fuzz
from rapidfuzz.fuzz_py import partial_ratio


df1 = pyreadr.read_r('joint_l1_fixation_version2.0_w1.rda')['joint.fix']
df2 = pyreadr.read_r('joint_fix_trimmed_l1_wave2_MinusCh_version2.0.RDA')['joint.fix']

df = pd.concat([df1, df2], join="inner", ignore_index=True)
sent_df = pd.read_csv('sentences.csv')
sentences_list = sent_df['sentence'].dropna().astype(str).tolist()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
en = df[
    (df['lang'] == 'en') &
    (df['blink'] == 0) &
    (df['dur'] >= 50)
].sort_values(['subid', 'trialid', 'fixid'])

sent_level = en.groupby(['subid', 'trialid', 'sentnum']).agg({
    'word': list,
    'dur': list,
    'ianum': list,
    'lang': 'first',
    'sent': 'first',

}).reset_index()

# Reset scanpath per sentence
def renumber_scanpath(sp):
    """Reassign scanpath numbers starting at 1, preserving order within sentence"""
    unique_order = sorted(set(sp), key=sp.index)  # unique fixations in original order
    mapping = {old: new for new, old in enumerate(unique_order, 1)}
    return [mapping[x] for x in sp]

def map_ianum_to_sentence_positions(ianum_list, full_sentence):
    """Map each ianum to its word position in the full sentence (1-based)."""

    if full_sentence is None or pd.isna(full_sentence):
        return [None] * len(ianum_list)

    # Tokenize full sentence (simple split; adjust if needed)
    words = str(full_sentence).strip().split()

    # Build mapping: ianum → position (1-based)
    # Assumes ianum corresponds to word order in sentence
    ia_to_pos = {i + 1: i + 1 for i in range(len(words))}

    # Map scanpath
    return [ia_to_pos.get(int(ia), None) for ia in ianum_list]

def fuzzy_match(sent, sentences_list, score_cutoff=80):
    """Return the best fuzzy match for sent in sentences_list, or None if no good match"""
    if pd.isna(sent) or not str(sent).strip():
        return None

    sent_str = str(sent).strip()
    match = process.extractOne(
        sent_str,
        sentences_list,
        scorer=fuzz.partial_ratio
    )

    if match is None:
        print("NO MATCH for:", sent_str)
        return None

    # Take first two elements in case match has index too
    matched_text, score = match[:2]

    if score >= score_cutoff:
        return matched_text
    else:
        print("NO MATCH for:", sent_str)
        return None

sent_level['full_sentence'] = sent_level['sent'].apply(
    lambda s: fuzzy_match(s, sentences_list, score_cutoff=80)
)

sent_level['scanpath'] = sent_level.apply(
    lambda row: map_ianum_to_sentence_positions(
        row['ianum'],
        row['full_sentence']
    ),
    axis=1
)
for i, row in sent_level.head(10).iterrows():
    print(f"Sentence {i+1}:")
    print("Words:", row['word'])
    print("Durations:", row['dur'])
    print("Scanpath:", row['ianum'])
    print("scanpath_renumbered", row['scanpath'])
    print("SubID:", row['subid'], "Sent ID:", row['sentnum'], "Lang:", row['lang'])
    print("Sentence truncated:", row['sent'])
    print("sentence full:", row['full_sentence'])

    print("FULL:", row['full_sentence'])
    print("IANUM:", row['ianum'])
    print("SCANPATH:", row['scanpath'])
    print("---")

"""
df = df[df['blink'] == 0]
df = df.sort_values(['subid', 'trialid', 'fixid'])
df['group_id'] = (
    df['subid'].astype(str) + "_" +
    df['trialid'].astype(str) + "_" +
    df['sent'].astype(str)
)
groups = df.groupby('group_id')


def match_sentence(truncated_sentence):
    if pd.isna(truncated_sentence):
        return None

    truncated_sentence = str(truncated_sentence).strip()

    match = process.extractOne(
        truncated_sentence,
        sentences_list,
        scorer=fuzz.partial_ratio  # good for prefix/fragments
    )

    if match is None:
        return None

    sentence, score, _ = match

    # Optional threshold to avoid bad matches
    if score >= 80:
        return sentence

    return None


# -----------------------------
# Apply matching
# -----------------------------
print("Applying sentence matching...")
df['full_sentence'] = df['sent'].apply(match_sentence)

# -----------------------------
# Validation
# -----------------------------

# Match rate
match_rate = df['full_sentence'].notna().mean()
print("\nMatch rate:", match_rate)

# Missing matches
missing = df[df['full_sentence'].isna()]
print("\nNumber of unmatched rows:", len(missing))

if len(missing) > 0:
    print("\nSample unmatched truncated sentences:")
    print(missing['sent'].drop_duplicates().head(20))

# Check correctness of matches
df['prefix_check'] = df.apply(
    lambda r: str(r['full_sentence']).startswith(str(r['sent']))
    if pd.notna(r['full_sentence']) else False,
    axis=1
)

print("\nPrefix correctness:")
print(df['prefix_check'].value_counts())

# -----------------------------
# Inspect some examples
# -----------------------------

print("\n--- Example matches ---")
examples = df[['sent', 'full_sentence']].drop_duplicates().dropna().sample(10)
print(examples)

print("\n--- Example mismatches ---")
bad_examples = df[df['full_sentence'].isna()][['sent']].drop_duplicates().head(10)

"""