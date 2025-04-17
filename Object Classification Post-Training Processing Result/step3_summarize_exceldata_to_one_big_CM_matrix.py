import numpy as np


data_from_file = "path/to/_v9.txt"


# Define the category dictionary
category = {
    'shoe': 21, 'paperBox': 11, 'pastry': 14, 'penPencil': 15, 'milkCarton': 9, 'cutlery': 3,
    'crumpledPaper': 2, 'eggShell': 4, 'glassBottle': 6, 'plasticContainer': 17, 'paperEnvelope': 13,
    'paperCup': 12, 'fruit': 5, 'noodlePasta': 10, 'plasticLid': 18, 'plasticMilkBottle': 19,
    'sandwich': 20, 'tissueCore': 23, 'vape': 24, 'vegeScraps': 25, 'glassJar': 7, 'meat': 8,
    'book': 1, 'plasticBottle': 16, 'alCan': 0, 'softPlastic': 22, 'background': 26
}

def parse_paragraph(paragraph):
    """
    Parse a single paragraph from the text file to extract true class name,
    true positive count, background count, and false detection counts.
    """
    lines = paragraph.strip().split('\n')
    # Extract true class name
    true_class_name = lines[0].split(': ')[1]
    # Extract true positive count
    tp_line = [line for line in lines if 'Total True Positive Detections:' in line][0]
    tp_count = int(tp_line.split(': ')[1])
    # Extract background count
    background_line = [line for line in lines if 'Total Class under General:' in line][0]
    background_count = int(background_line.split(': ')[1])
    # Locate start of false detections
    false_detections_start = lines.index('Top Most Common False Detections:') + 1
    # Check if there are no false detections
    if lines[false_detections_start].strip() == 'No false detections.':
        false_detections = {}
    else:
        false_detections = {}
        # Parse each false detection line
        for line in lines[false_detections_start:]:
            if line.strip():  # Skip empty lines
                parts = line.strip().split(': ')
                pred_class_name = parts[0]
                count = int(parts[1].split(' ')[0])
                false_detections[pred_class_name] = count
    return true_class_name, tp_count, background_count, false_detections

# Read the text file (replace 'path_to_file.txt' with your actual file path)
with open(data_from_file, 'r') as file:
    text = file.read()

# Split the text into paragraphs
paragraphs = text.strip().split('\n\n')

# Initialize a 27x27 confusion matrix with zeros
cm = np.zeros((27, 27), dtype=int)

# Process each paragraph
for paragraph in paragraphs:
    true_class_name, tp_count, background_count, false_detections = parse_paragraph(paragraph)
    # Get the index of the true class
    true_idx = category[true_class_name]
    # Set true positive count on the diagonal
    cm[true_idx, true_idx] = tp_count
    # Set false detection counts in the corresponding predicted class columns
    for pred_class_name, count in false_detections.items():
        pred_idx = category[pred_class_name]
        cm[true_idx, pred_idx] = count
    # Set background count in the last column
    cm[true_idx, 26] = background_count

# The row for 'background' (index 26) remains all zeros since no paragraph exists for it

# Print the matrix (optional: adjust formatting as needed)
print("Confusion Matrix:")
for row in cm:
    print(f'{row.tolist()},')

