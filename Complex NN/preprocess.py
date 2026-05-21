import pandas as pd

# -------------------------
# LOAD DATA
# -------------------------
pokemon = pd.read_csv("pokemon.csv").rename(columns={"#": "PokemonID"})
combats = pd.read_csv("combats.csv")

# -------------------------
# TARGET
# -------------------------
combats["target"] = (
    combats["Winner"] == combats["Second_pokemon"]
).astype(int)

y = combats["target"].copy()
combats = combats.drop(columns=["Winner"])

# -------------------------
# MERGE POKEMON DATA (FOR TRAINING ONLY)
# -------------------------
p1 = combats.merge(
    pokemon,
    left_on="First_pokemon",
    right_on="PokemonID"
).add_prefix("p1_")

p2 = combats.merge(
    pokemon,
    left_on="Second_pokemon",
    right_on="PokemonID"
).add_prefix("p2_")

df = pd.concat([p1, p2], axis=1).reset_index(drop=True)
df["target"] = y.values

# -------------------------
# CLEANUP
# -------------------------
df = df.drop(columns=[
    "p1_Name",
    "p2_Name",
    "p1_PokemonID",
    "p2_PokemonID"
])

# -------------------------
# ONE-HOT ENCODING
# -------------------------
df = pd.get_dummies(
    df,
    columns=[
        "p1_Type 1",
        "p1_Type 2",
        "p2_Type 1",
        "p2_Type 2"
    ]
).fillna(0)

# =========================================================
# 1. SAVE CLEAN POKEMON TABLE (FOR PREDICTION)
# =========================================================
pokemon_clean = pd.get_dummies(
    pokemon,
    columns=["Type 1", "Type 2"]
).fillna(0)

pokemon_clean.to_csv("pokemon_processed.csv", index=False)

# =========================================================
# 2. BUILD TRAINING FEATURES (FIXED)
# =========================================================

IGNORE = {
    "First_pokemon",
    "Second_pokemon",
    "Winner",
    "target",
    "PokemonID"
}

features = []

for col in df.columns:
    if not col.startswith("p1_"):
        continue

    base = col.replace("p1_", "")

    # ❌ skip IDs / metadata
    if base in IGNORE:
        continue

    p2_col = col.replace("p1_", "p2_")

    if p2_col not in df.columns:
        continue

    diff_name = base + "_diff"

    df[diff_name] = (
        df[col].astype(float) - df[p2_col].astype(float)
    )

    features.append(diff_name)

final_df = df[features + ["target"]]

# =========================================================
# SYMMETRY AUGMENTATION
# =========================================================
swapped = final_df.copy()
swapped[features] = -swapped[features]
swapped["target"] = 1 - swapped["target"]

final_df = pd.concat([final_df, swapped], ignore_index=True)

# =========================================================
# SAVE TRAINING DATA
# =========================================================
final_df.to_csv("train_data.csv", index=False)

pd.Series(features).to_csv(
    "feature_names.csv",
    index=False,
    header=False
)

print("Preprocessing complete")
print("Saved:")
print("- train_data.csv")
print("- feature_names.csv")
print("- pokemon_processed.csv")