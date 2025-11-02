import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("cleaned_mortality_final.csv")

# Remove rows where any column has 'CRS System stared' or other invalid strings
df = df.replace("CRS System stared", pd.NA)

# Convert all numeric columns to float, forcing errors to NaN
for col in df.columns:
    if col != "City Name":
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with missing key values
df = df.dropna(subset=[
    "Total No. of Live Births",
    "No. of Deaths - Total",
    "No. of Deaths - Male",
    "No. of Deaths - Female",
    "No. of Deaths - Infants (0-1 year)",
    "No. of Deaths - Children (1-5 years)"
])

# Define features and target
X = df[[
    "Total No. of Live Births",
    "No. of Deaths - Male",
    "No. of Deaths - Female",
    "No. of Deaths - Infants (0-1 year)",
    "No. of Deaths - Children (1-5 years)"
]]
y = df["No. of Deaths - Total"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Model trained successfully!")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualization
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Mortality")
plt.ylabel("Predicted Mortality")
plt.title("Actual vs Predicted Mortality Rates")
plt.show()

# Interactive prediction
print("\n🔮 Predict future mortality rate:")
live_births = float(input("Enter total live births (in thousands): "))
male_deaths = float(input("Enter male deaths (in thousands): "))
female_deaths = float(input("Enter female deaths (in thousands): "))
infant_deaths = float(input("Enter infant deaths (in thousands): "))
child_deaths = float(input("Enter child deaths (in thousands): "))

new_data = pd.DataFrame([[live_births, male_deaths, female_deaths, infant_deaths, child_deaths]],
                        columns=X.columns)
predicted = model.predict(new_data)[0]
print(f"\n✅ Predicted Total Deaths: {predicted:.2f} (in thousands)")
