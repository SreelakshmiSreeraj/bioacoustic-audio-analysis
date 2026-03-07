import pandas as pd

animals = pd.read_csv("../features_animal_sounds.csv")
other = pd.read_csv("../features_other_sounds.csv")

print("Animal dataset shape:", animals.shape)
print("Other sounds dataset shape:", other.shape)

print("\nAnimal labels:")
print(animals["label"].value_counts())

print("\nOther sound labels:")
print(other["label"].value_counts())