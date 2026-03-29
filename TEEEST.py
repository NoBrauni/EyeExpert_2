import pyreadr

# Paths to your RDA files
rda_files = [
    "joint_l1_fixation_version2.0_w1.rda",
    "joint_fix_trimmed_l1_wave2_MinusCh_version2.0.RDA"
]

all_langs = set()

for f in rda_files:
    result = pyreadr.read_r(f)
    # Assuming the fixation table is called 'joint.fix' in the RDA
    df = result["joint.fix"]

    if "lang" in df.columns:
        langs = df["lang"].dropna().unique()
        all_langs.update(langs)
    else:
        all_langs.add("unknown")

print("Languages found across files:", sorted(all_langs))

