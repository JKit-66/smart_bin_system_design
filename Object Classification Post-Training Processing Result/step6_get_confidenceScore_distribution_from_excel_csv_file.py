import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define the category mapping for class indices
category = {
    'shoe': 21, 'paperBox': 11, 'pastry': 14, 'penPencil': 15, 'milkCarton': 9, 'cutlery': 3,
    'crumpledPaper': 2, 'eggShell': 4, 'glassBottle': 6, 'plasticContainer': 17, 'paperEnvelope': 13,
    'paperCup': 12, 'fruit': 5, 'noodlePasta': 10, 'plasticLid': 18, 'plasticMilkBottle': 19,
    'sandwich': 20, 'tissueCore': 23, 'vape': 24, 'vegeScraps': 25, 'glassJar': 7, 'meat': 8,
    'book': 1, 'plasticBottle': 16, 'alCan': 0, 'softPlastic': 22, 'background': 26
}

# Define the main category mapping
main_category = {
    "Paper": ["crumpledPaper", "paperEnvelope", "paperBox", "tissueCore", "book"],
    "Mixed Recycling": ["alCan", "plasticBottle", "plasticContainer", "glassBottle", "glassJar", "plasticLid", "plasticMilkBottle"],
    "Food Waste": ["vegeScraps", "noodlePasta", "meat", "sandwich", "eggShell", "pastry", "fruit"],
    "General Waste": ["softPlastic", "paperCup", "penPencil", "cutlery", "shoe", "milkCarton"],
    "Prohibited Item": ["vape"]
}

# Create a list of class names sorted by their indices (0 to 25)
class_names = [''] * 26
for class_name, index in category.items():
    if index < 26:  # Ignore 'background' (index 26)
        class_names[index] = class_name

# Initialize lists for each bin, with 26 entries (one for each class, initially 0)
bin_1 = [0] * 26  # 1-40
bin_2 = [0] * 26  # 40-60
bin_3 = [0] * 26  # 60-70
bin_4 = [0] * 26  # 70-80
bin_5 = [0] * 26  # 80-90
bin_6 = [0] * 26  # 90-100

# Path to your text file (replace with your actual file path)
file_path = "path/to/_v9.txt"

# Read and parse the file
with open(file_path, 'r') as file:
    lines = file.readlines()

current_class = None
in_true_positive = False
in_false_positive = False
true_counts = {}
false_counts = {}

for line in lines:
    line = line.strip()

    # Identify the class
    if line.startswith("Analysis for Class:"):
        if current_class is not None:
            # Process the previous class
            combined = {
                '1-40': true_counts.get('1-40', 0) + false_counts.get('1-40', 0),
                '40-60': true_counts.get('40-60', 0) + false_counts.get('40-60', 0),
                '60-70': true_counts.get('60-70', 0) + false_counts.get('60-70', 0),
                '70-80': true_counts.get('70-80', 0) + false_counts.get('70-80', 0),
                '80-90': true_counts.get('80-90', 0) + false_counts.get('80-90', 0),
                '90-100': true_counts.get('90-100', 0) + false_counts.get('90-100', 0)
            }
            class_index = category.get(current_class)
            if class_index is not None and class_index < 26:
                bin_1[class_index] = combined['1-40']
                bin_2[class_index] = combined['40-60']
                bin_3[class_index] = combined['60-70']
                bin_4[class_index] = combined['70-80']
                bin_5[class_index] = combined['80-90']
                bin_6[class_index] = combined['90-100']

        # Reset for the new class
        current_class = line.split("Analysis for Class:")[1].strip()
        in_true_positive = False
        in_false_positive = False
        true_counts = {}
        false_counts = {}

    # Identify True Positive Detections section
    elif line == "True Positive Detections:":
        in_true_positive = True
        in_false_positive = False

    # Identify False Positive Detections section
    elif line == "False Positive Detections:":
        in_true_positive = False
        in_false_positive = True

    # Parse confidence distribution
    elif line.startswith(("1-40:", "40-60:", "60-70:", "70-80:", "80-90:", "90-100:")):
        bin_name, count = line.split(":")
        bin_name = bin_name.strip()
        count = int(count.strip())
        if in_true_positive:
            true_counts[bin_name] = count
        elif in_false_positive:
            false_counts[bin_name] = count

# Process the last class
if current_class is not None:
    combined = {
        '1-40': true_counts.get('1-40', 0),
        '40-60': true_counts.get('40-60', 0),
        '60-70': true_counts.get('60-70', 0),
        '70-80': true_counts.get('70-80', 0),
        '80-90': true_counts.get('80-90', 0),
        '90-100': true_counts.get('90-100', 0)         # + false_counts.get('90-100', 0)
    }
    class_index = category.get(current_class)
    if class_index is not None and class_index < 26:
        bin_1[class_index] = combined['1-40']
        bin_2[class_index] = combined['40-60']
        bin_3[class_index] = combined['60-70']
        bin_4[class_index] = combined['70-80']
        bin_5[class_index] = combined['80-90']
        bin_6[class_index] = combined['90-100']

# Create the per-class DataFrame
data = pd.DataFrame({
    'Class': class_names,
    '1-40': bin_1,  # 1-40
    '40-60': bin_2,  # 40-60
    '60-70': bin_3,  # 60-70
    '70-80': bin_4,  # 70-80
    '80-90': bin_5,  # 80-90
    '90-100': bin_6   # 90-100
})

# Print the per-class DataFrame
print("Per-Class DataFrame:")
print(data)

# Create a reverse mapping from class to main category
class_to_category = {}
for main_cat, classes in main_category.items():
    for cls in classes:
        class_to_category[cls] = main_cat

# Initialize lists for the 4 categories
categories = ['Mixed Recycling', 'Paper', 'General Waste', 'Food Waste', 'Prohibited Item']
cat_bin_1 = [0] * 5
cat_bin_2 = [0] * 5
cat_bin_3 = [0] * 5
cat_bin_4 = [0] * 5
cat_bin_5 = [0] * 5
cat_bin_6 = [0] * 5

# Sum counts by category
for idx, class_name in enumerate(class_names):
    main_cat = class_to_category.get(class_name)
    if main_cat in categories:  # Only process the 4 specified categories
        cat_idx = categories.index(main_cat)
        cat_bin_1[cat_idx] += bin_1[idx]
        cat_bin_2[cat_idx] += bin_2[idx]
        cat_bin_3[cat_idx] += bin_3[idx]
        cat_bin_4[cat_idx] += bin_4[idx]
        cat_bin_5[cat_idx] += bin_5[idx]
        cat_bin_6[cat_idx] += bin_6[idx]

# Create the category DataFrame
data_categories = pd.DataFrame({
    'Category': categories,
    '1-40': cat_bin_1,  # 1-40
    '40-60': cat_bin_2,  # 40-60
    '60-70': cat_bin_3,  # 60-70
    '70-80': cat_bin_4,  # 70-80
    '80-90': cat_bin_5,  # 80-90
    '90-100': cat_bin_6   # 90-100
})


# Print the category DataFrame
print("\nCategory-Summarized DataFrame:")
print(data_categories)

# Plot heatmap for per-class DataFrame
heatmap_data = data.set_index('Class')
plt.figure(figsize=(12, 10))
sns.heatmap(heatmap_data, cmap='Blues', annot=True, fmt='g', linewidths=.5,
            cbar_kws={'label': 'Count'},
            xticklabels=['[0-40]', '(40-60]', '(60-70]', '(70-80]', '(80-90]', '(90-100]'],
            yticklabels=heatmap_data.index)
plt.title('Distribution of Confidence Scores Across 26 Classes')
plt.xlabel('Score Bins')
plt.ylabel('Classes')
plt.show()

# Plot heatmap for category-summarized DataFrame
heatmap_data_categories = data_categories.set_index('Category')
plt.figure(figsize=(12, 4))  # Smaller height since there are only 4 rows
sns.heatmap(heatmap_data_categories, cmap='Blues', annot=True, fmt='g', linewidths=.5,
            cbar_kws={'label': 'Count'},
            xticklabels=['[0-40]', '(40-60]', '(60-70]', '(70-80]', '(80-90]', '(90-100]'],
            yticklabels=heatmap_data_categories.index)
plt.title('Distribution of Confidence Scores Across 4 Categories')
plt.xlabel('Score Bins')
plt.ylabel('Categories')
plt.show()