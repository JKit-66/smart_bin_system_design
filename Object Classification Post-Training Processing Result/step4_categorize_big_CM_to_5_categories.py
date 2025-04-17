import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

cm = np.array([
[108, 3, 0, 0, 0, 0, 0, 1, 0, 3, 0, 8, 4, 0, 2, 0, 0, 8, 4, 0, 0, 0, 0, 0, 0, 0, 13],
[0, 90, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 24, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 28],
[0, 0, 181, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 67, 0, 0, 0, 0, 46, 0, 0, 0, 19],
[0, 1, 0, 262, 0, 0, 0, 0, 0, 1, 0, 0, 1, 13, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 26],
[0, 0, 6, 0, 96, 16, 0, 0, 10, 0, 0, 1, 1, 0, 22, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 51],
[1, 0, 2, 0, 2, 144, 0, 1, 5, 0, 2, 0, 0, 0, 10, 0, 0, 0, 3, 0, 9, 0, 0, 0, 0, 9, 53],
[0, 0, 0, 1, 0, 0, 84, 3, 0, 3, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1, 0, 0, 0, 0, 15],
[9, 0, 0, 0, 0, 0, 6, 81, 0, 5, 0, 0, 0, 0, 0, 0, 11, 13, 0, 0, 0, 0, 0, 0, 0, 0, 16],
[0, 0, 0, 1, 0, 0, 0, 0, 274, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 19, 0, 0, 0, 0, 2, 154],
[0, 0, 0, 0, 0, 0, 2, 0, 1, 120, 0, 28, 2, 0, 0, 0, 4, 0, 1, 0, 0, 0, 4, 0, 0, 0, 24],
[0, 0, 0, 1, 0, 0, 0, 0, 4, 0, 97, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6],
[0, 7, 0, 0, 0, 0, 0, 0, 0, 1, 0, 96, 0, 3, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 7],
[3, 0, 0, 0, 0, 2, 0, 1, 0, 0, 4, 0, 92, 0, 0, 0, 1, 9, 0, 0, 0, 1, 0, 0, 0, 4, 15],
[0, 14, 3, 0, 0, 0, 0, 0, 0, 2, 0, 4, 1, 135, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 28],
[0, 0, 0, 1, 0, 3, 0, 0, 19, 0, 0, 0, 0, 0, 116, 0, 0, 0, 0, 0, 23, 0, 0, 0, 0, 1, 41],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 235, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 34],
[3, 0, 0, 0, 0, 0, 15, 12, 0, 1, 0, 0, 0, 0, 0, 0, 68, 0, 2, 0, 0, 0, 0, 0, 0, 0, 45],
[2, 1, 0, 0, 0, 0, 2, 31, 0, 8, 0, 0, 2, 0, 0, 0, 0, 50, 4, 0, 0, 0, 6, 0, 0, 0, 23],
[21, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 6, 0, 97, 8, 0, 0, 0, 0, 0, 0, 13],
[0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 26, 1, 0, 0, 12, 4, 0, 112, 0, 14, 0, 0, 0, 0, 52],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 138, 0, 0, 0, 0, 0, 17],
[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 385, 0, 0, 0, 0, 10],
[0, 5, 0, 0, 0, 0, 0, 0, 0, 3, 1, 40, 0, 1, 2, 0, 0, 7, 0, 0, 2, 0, 233, 0, 0, 0, 68],
[0, 0, 1, 0, 0, 0, 0, 0, 3, 0, 0, 0, 17, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 2, 17],
[2, 1, 0, 0, 0, 0, 2, 3, 0, 0, 0, 1, 1, 2, 0, 23, 0, 5, 1, 2, 0, 12, 5, 0, 22, 3, 27],
[0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 107, 60],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]    
])


category = {
    'shoe': 21, 'paperBox': 11, 'pastry': 14, 'penPencil': 15, 'milkCarton': 9, 'cutlery': 3,
    'crumpledPaper': 2, 'eggShell': 4, 'glassBottle': 6, 'plasticContainer': 17, 'paperEnvelope': 13,
    'paperCup': 12, 'fruit': 5, 'noodlePasta': 10, 'plasticLid': 18, 'plasticMilkBottle': 19,
    'sandwich': 20, 'tissueCore': 23, 'vape': 24, 'vegeScraps': 25, 'glassJar': 7, 'meat': 8,
    'book': 1, 'plasticBottle': 16, 'alCan': 0, 'softPlastic': 22, 'background': 26
}

# Define the main categories
main_category = {
    "Paper": ["crumpledPaper", "paperEnvelope", "paperBox", "tissueCore", "book"],
    "Mixed Recycling": ["alCan", "plasticBottle", "plasticContainer", "glassBottle", "glassJar", "plasticLid", "plasticMilkBottle"],
    "Food Waste": ["vegeScraps", "noodlePasta", "meat", "sandwich", "eggShell", "pastry", "fruit"],
    "General Waste": ["softPlastic", "paperCup", "penPencil", "cutlery", "shoe", "milkCarton"],
    "Prohibited Item": ["vape"]
}

# Add Background to main_category
main_category["Background"] = ["background"]
# Define the order of categories for the final matrix
main_categories = ["Paper", "Mixed Recycling", "Food Waste", "General Waste", "Prohibited Item", "Background"]

# Compute indices for each main category
indices_list = [[category[item] for item in main_category[cat]] for cat in main_categories]

# Initialize the aggregated 6x6 matrix
aggregated_cm = np.zeros((6, 6), dtype=int)

# Aggregate the matrix
for i in range(6):
    for j in range(6):
        aggregated_cm[i, j] = cm[np.ix_(indices_list[i], indices_list[j])].sum()

matrix_list = aggregated_cm.tolist()
for row in matrix_list:
    print(f"{row},")

# Normalize the aggregated matrix across rows
row_sums = aggregated_cm.sum(axis=1, keepdims=True)
# Avoid division by zero by setting sums of 0 to 1 (though Background row stays 0 in output)
row_sums[row_sums == 0] = 1  # This ensures no NaN values, but normalized row remains 0
normalized_cm = aggregated_cm / row_sums

# Display as DataFrames for clarity
print("Aggregated 6x6 Confusion Matrix (Raw Counts):")
aggregated_df = pd.DataFrame(aggregated_cm, index=main_categories, columns=main_categories)
print(aggregated_df)

#print("\nNormalized Aggregated 6x6 Confusion Matrix:")
normalized_df = pd.DataFrame(normalized_cm, index=main_categories, columns=main_categories)
#print(normalized_df.round(4))  # Round to 4 decimal places
