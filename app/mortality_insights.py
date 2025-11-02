import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("cleaned_mortality_final.csv")

# Clean column names and strip spaces
df.columns = df.columns.str.strip()

# Convert numeric columns safely
numeric_cols = [
    "No. of Deaths - Children (1-5 years)",
    "No. of Deaths - Female",
    "No. of Deaths - Infants (0-1 year)",
    "No. of Deaths - Male",
    "No. of Deaths - Total",
    "No. of Deaths - age above 5 years",
    "Total No. of Live Births",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows missing critical info
df = df.dropna(subset=["City Name", "Year", "No. of Deaths - Total", "Total No. of Live Births"])

# ---- 1️⃣ Mortality Trend per City ----
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="Year", y="No. of Deaths - Total", hue="City Name", marker="o")
plt.title("City-wise Mortality Trend Over Years")
plt.xlabel("Year")
plt.ylabel("Total Deaths (in Thousands)")
plt.legend(title="City", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ---- 2️⃣ Live Births vs Deaths ----
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="Total No. of Live Births", y="No. of Deaths - Total", hue="City Name", s=80)
plt.title("Live Births vs Total Deaths per City")
plt.xlabel("Live Births (in Thousands)")
plt.ylabel("Total Deaths (in Thousands)")
plt.legend(title="City", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ---- 3️⃣ Gender-wise Mortality ----
gender_df = (
    df.groupby("City Name")[["No. of Deaths - Male", "No. of Deaths - Female"]]
    .mean()
    .reset_index()
    .melt(id_vars="City Name", var_name="Gender", value_name="Deaths")
)

plt.figure(figsize=(12, 6))
sns.barplot(data=gender_df, x="City Name", y="Deaths", hue="Gender")
plt.title("Gender-wise Average Mortality per City")
plt.xticks(rotation=45)
plt.ylabel("Average Deaths (in Thousands)")
plt.tight_layout()
plt.show()

# ---- 4️⃣ Infant Mortality Trend ----
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="Year", y="No. of Deaths - Infants (0-1 year)", hue="City Name", marker="o")
plt.title("Infant Mortality Trend Over Years")
plt.xlabel("Year")
plt.ylabel("Infant Deaths (in Thousands)")
plt.legend(title="City", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ---- 5️⃣ Top and Bottom Cities by Average Mortality ----
avg_mortality = df.groupby("City Name")["No. of Deaths - Total"].mean().sort_values(ascending=False)

print("\n🏙️ Top 5 Cities by Average Mortality:")
print(avg_mortality.head(5))

print("\n🌱 Bottom 5 Cities by Average Mortality:")
print(avg_mortality.tail(5))
