
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# AI-Powered Disaster Response System

print("=" * 55)
print(" AI-POWERED DISASTER RESPONSE RESOURCE ALLOCATION")
print("=" * 55)


# Sample disaster data
data = {
    "severity": [9, 7, 5, 8, 6, 4, 9, 7, 5, 8],
    "population": [5000, 3000, 1500, 4500, 2500, 1000, 6000, 3500, 1800, 4000],
    "urgency": [9, 8, 5, 9, 6, 4, 10, 7, 5, 8],
    "resources_needed": [90, 70, 40, 85, 55, 30, 95, 75, 45, 80]
}

df = pd.DataFrame(data)


# Create priority labels
def calculate_priority(row):
    score = (
        row["severity"] * 0.4
        + row["urgency"] * 0.4
        + (row["population"] / 6000) * 10 * 0.2
    )

    if score >= 7.5:
        return 2       # High
    elif score >= 5:
        return 1       # Medium
    else:
        return 0       # Low


df["priority"] = df.apply(calculate_priority, axis=1)


# Features used by the AI model
X = df[["severity", "population", "urgency"]]
y = df["priority"]


# Train Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)


# Take information from the user

print("\nEnter disaster information:")
print("(Use values between 1 and 10 for severity and urgency.)")

try:
    severity = int(input("Severity (1-10): "))
    population = int(input("Affected population: "))
    urgency = int(input("Urgency (1-10): "))

    if severity < 1 or severity > 10:
        raise ValueError("Severity must be between 1 and 10.")

    if urgency < 1 or urgency > 10:
        raise ValueError("Urgency must be between 1 and 10.")

    if population <= 0:
        raise ValueError("Population must be greater than 0.")


    # Predict priority using AI model
    input_data = pd.DataFrame(
        [[severity, population, urgency]],
        columns=["severity", "population", "urgency"]
    )

    prediction = model.predict(input_data)[0]


    # Convert prediction to priority name
    priority_names = {
        0: "LOW",
        1: "MEDIUM",
        2: "HIGH"
    }

    priority = priority_names[prediction]


    # Resource allocation

    if priority == "HIGH":
        food = 80
        water = 100
        medical = 70
        rescue = 60

    elif priority == "MEDIUM":
        food = 50
        water = 60
        medical = 40
        rescue = 35

    else:
        food = 30
        water = 40
        medical = 20
        rescue = 15


    # Adjust resources according to population
    population_factor = min(population / 5000, 2)

    food = int(food * population_factor)
    water = int(water * population_factor)
    medical = int(medical * population_factor)
    rescue = int(rescue * population_factor)

    # Display result

    print("\n" + "=" * 55)
    print("DISASTER RESPONSE ANALYSIS")
    print("=" * 55)

    print(f"Severity             : {severity}/10")
    print(f"Affected Population  : {population}")
    print(f"Urgency              : {urgency}/10")
    print(f"Priority Level        : {priority}")

    print("\nRecommended Resources:")
    print(f"Food Units            : {food}")
    print(f"Water Units           : {water}")
    print(f"Medical Units         : {medical}")
    print(f"Rescue Teams          : {rescue}")

    print("\n" + "=" * 55)
    print("Resources have been allocated based on predicted priority.")
    print("=" * 55)


except ValueError as e:
    print("\nInvalid input:", e)
