import pandas as pd

# Read your big combined file
df = pd.read_csv("mortality_combined.csv", encoding='utf-8', on_bad_lines='skip')

# Drop completely empty rows
df = df.dropna(how='all')

# Remove duplicate unnamed columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Keep only relevant columns
keep_cols = [c for c in df.columns if "City Name" in c or "Total No. of Live Births" in c or "No. of Deaths" in c]
df = df[keep_cols]

# Clean column names (remove extra spaces and brackets)
df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()

# Melt the dataset (wide → long)
long_df = df.melt(id_vars=["City Name"], var_name="MetricYear", value_name="Value")

# Extract metric name and year from column text
long_df["Metric"] = long_df["MetricYear"].str.extract(r"^(.*?)\s*\(in Thousands\)")
long_df["Year"] = long_df["MetricYear"].str.extract(r"(\d{4}-\d{2,4})")

# Clean up and pivot back into tidy table
final_df = long_df.pivot_table(index=["City Name", "Year"], columns="Metric", values="Value", aggfunc="first").reset_index()

# Save clean version
final_df.to_csv("cleaned_mortality_final.csv", index=False)

print("✅ Cleaned tidy dataset saved as cleaned_mortality_final.csv")
print(final_df.head(10))
