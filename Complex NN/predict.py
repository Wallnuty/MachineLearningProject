import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

# -------------------------
# MODEL
# -------------------------
class BattleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# LOAD ARTIFACTS
# -------------------------
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BattleNN(len(feature_names))
model.load_state_dict(torch.load("battle_model.pth", map_location=device))
model.to(device)
model.eval()

# -------------------------
# LOAD CLEAN POKEMON DATA
# -------------------------
pokemon_df = pd.read_csv("pokemon_processed.csv")


# -------------------------
# FEATURE BUILDER (CORRECT VERSION)
# -------------------------
def build_features(p1_id, p2_id):

    p1 = pokemon_df[pokemon_df["PokemonID"] == p1_id].iloc[0]
    p2 = pokemon_df[pokemon_df["PokemonID"] == p2_id].iloc[0]

    features = {}

    for f in feature_names:

        col = f.replace("_diff", "")

        # strict check (fail fast instead of hiding bugs)
        if col not in pokemon_df.columns:
            raise ValueError(f"Missing column in pokemon data: {col}")

        features[f] = float(p1[col]) - float(p2[col])

    return np.array([features[f] for f in feature_names], dtype=np.float32)

# -------------------------
# PREDICT FUNCTION
# -------------------------
def predict(p1_id, p2_id):

    x = build_features(p1_id, p2_id)

    x = scaler.transform(x.reshape(1, -1))
    x = torch.tensor(x, dtype=torch.float32).to(device)

    with torch.no_grad():
        logit = model(x)
        prob_second = torch.sigmoid(logit).item()

    prob_first = 1 - prob_second

    if prob_second > 0.5:
        winner = "Second Pokemon"
        confidence = prob_second
    else:
        winner = "First Pokemon"
        confidence = prob_first

    return winner, confidence


# -------------------------
# EXAMPLE
# -------------------------
if __name__ == "__main__":
    first_pokemon_id = int(input("Enter the ID of the first pokemon: "))
    second_pokemon_id = int(input("Enter the ID of the second pokemon: "))
    winner, prob = predict(first_pokemon_id, second_pokemon_id)

    print("Winner:", winner)
    print("Confidence:", prob)