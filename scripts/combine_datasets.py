import pandas as pd

animals = pd.read_csv("../features_animal_sounds.csv")
other = pd.read_csv("../features_other_sounds.csv")

animals["category"] = "animal"
other["category"] = "other"

combined = pd.concat([animals, other], ignore_index=True)

combined.to_csv("features_all.csv", index=False)

print("Combined dataset created")
print("Shape:", combined.shape)