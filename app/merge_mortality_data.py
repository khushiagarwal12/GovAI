import pandas as pd
import glob

files = glob.glob("cleaned_data/*.csv")
all_data = []

for f in files:
    df = pd.read_csv(f)
    all_data.append(df)

final_df = pd.concat(all_data, ignore_index=True)
final_df.to_csv("mortality_combined.csv", index=False)

print("✅ Combined dataset created:", final_df.shape)
