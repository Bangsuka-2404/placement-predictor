import numpy as np
import pandas as pd

np.random.seed(42)

N = 500

# Generate realistic student features
cgpa = np.round(np.random.uniform(5.0, 10.0, N), 2)
projects = np.random.randint(0, 6, N)               # 0 to 5
skills_score = np.random.randint(30, 101, N)        # 30 to 100
aptitude_score = np.random.randint(20, 101, N)      # 20 to 100
internships = np.random.randint(0, 4, N)            # 0 to 3

# Placement probability logic (rule-based)
placement_score = (
    (cgpa * 10) * 0.35 +           # CGPA impact
    (skills_score) * 0.30 +        # skills impact
    (aptitude_score) * 0.20 +      # aptitude impact
    (projects * 5) * 0.10 +        # projects impact
    (internships * 8) * 0.05       # internship impact
)

# Add noise for realism
noise = np.random.normal(0, 8, N)
placement_score = placement_score + noise

# Convert into placed/not placed
placed = (placement_score >= 65).astype(int)

df = pd.DataFrame({
    "cgpa": cgpa,
    "projects": projects,
    "skills_score": skills_score,
    "aptitude_score": aptitude_score,
    "internships": internships,
    "placed": placed
})

df.to_csv("placement_data.csv", index=False)

print("✅ placement_data.csv generated successfully with 500 rows!")
print(df.head(10))
