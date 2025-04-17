import pandas as pd
from collections import Counter
import ast

theoretical_ans = {'shoe': 390, 
                   'paperBox': 106, 
                   'pastry': 227, 
                   'penPencil': 279, 
                   'milkCarton': 154, 
                   'cutlery': 275,
                   'crumpledPaper': 293, 
                   'eggShell':106, 
                   'glassBottle': 100, 
                   'plasticContainer': 100,
                   'paperEnvelope': 223, 
                   'paperCup': 100, 
                   'fruit': 107, 
                   'noodlePasta': 100, 
                   'plasticLid': 159, 
                   'plasticMilkBottle': 159, 
                   'sandwich': 102,
                   'tissueCore': 101, 
                   'vape': 122, 
                   'vegeScraps': 120, 
                   'glassJar': 126, 
                   'meat': 123, 
                   'book': 128, 
                   'plasticBottle': 107, 
                   'alCan': 132, 
                   'softPlastic': 103
                   }

# Function to analyze confidence scores
def analyze_confidences(confidences):
    """Compute mean confidence and categorize scores into bins."""
    if not confidences:
        return None, None
    mean_conf = sum(confidences) / len(confidences)
    conf_percent = [conf * 100 for conf in confidences]  # Convert to percentages
    bins = [(1, 40), (40, 60), (60, 70), (70, 80), (80, 90), (90, 100)]
    bin_counts = {f"{low}-{high}": 0 for low, high in bins}
    for conf in conf_percent:
        for low, high in bins:
            if low <= conf < high:
                bin_counts[f"{low}-{high}"] += 1
                break
    return mean_conf, bin_counts

# Read the CSV file
df = pd.read_csv("path/to/DetectionResults_fd88efb124b04c15a39e4e5b951518d3_v9.csv")  #testEnv FINAL_DetectionResults_6fe69b57a99a4c17b777f816db57e04d
threshold = 0.4

# Group data by 'Class' (true class)
for class_name, group in df.groupby('Class'):
    total_units = len(group)
    if total_units != 100:
        #print(f"Warning: Class '{class_name}' has {total_units} rows instead of 100.")
        pass
    
    # Initialize variables
    detected_count = 0
    max_confidences = []
    num_items_list = []
    false_detections = []
    true_positive_confidences = []  # Confidence scores where detected class matches true class
    false_positive_confidences = []  # Confidence scores where detected class does not match true class
    
    # Process each row in the group
    for _, row in group.iterrows():
        num_items = row['Number of Item Detected']
        detected_classes = ast.literal_eval(row['Detected Classes Name'])
        confidence_scores = ast.literal_eval(row['Confidence Score'])
        true_class = row['Class']
        
        num_items_list.append(num_items)
        
        # Analyze all detections in the row
        for cls, conf in zip(detected_classes, confidence_scores):
            if cls == true_class:
                true_positive_confidences.append(conf)
            else:
                false_positive_confidences.append(conf)

            if (conf > threshold) and (cls != true_class):
                false_detections.append(cls)

        #print(false_detections)

        filtered_classes = [cls for cls, conf in zip(detected_classes, confidence_scores) if conf > threshold]
        filtered_confidences = [conf for cls, conf in zip(detected_classes, confidence_scores) if conf > threshold]
        # Existing logic: Check if true class is detected and collect max confidence
        for true_class_row in filtered_classes:
            if true_class in true_class_row:
                detected_count += 1
                true_class_confidences = [conf for cls, conf in zip(filtered_classes, filtered_confidences) if cls == true_class]
                max_confidence = max(true_class_confidences)
                max_confidences.append(max_confidence)
        #print('fe', true_class, filtered_classes)
    #print(detected_count)
    # Perform analysis
    theoretical_accuracy = detected_count/ theoretical_ans[true_class]
    detection_accuracy = detected_count / total_units
    avg_max_confidence = sum(max_confidences) / detected_count if detected_count > 0 else None
    avg_num_items = sum(num_items_list) / total_units
    false_detection_counter = Counter(false_detections)
    most_common_false = false_detection_counter.most_common(30)
    
    # Analyze confidence scores for true positive and false positive detections
    tp_mean, tp_bin_counts = analyze_confidences(true_positive_confidences)
    fp_mean, fp_bin_counts = analyze_confidences(false_positive_confidences)
    
    # Output results
    print(f"Analysis for Class: {class_name}")
    print(f'Total True Positive Detections: {detected_count}')

    #print(f"Total Units: {total_units}")
    #print(f'Total Detections: {sum(num_items_list)}')
    #print(f'Theoretical Accuracy: {theoretical_accuracy}; ({detected_count} out of {theoretical_ans[true_class]})')
    
    '''#print(f"Detection Accuracy: {detection_accuracy:.2f} ({detected_count}/{total_units} units)")'''
    print(f"Total Class under General: {sum(num_items_list) - detected_count - len(false_detections)}")
    
    if avg_max_confidence is not None:
        print(f"Average Max Confidence for True Class: {avg_max_confidence:.4f}")
    else:
        print("True class was never detected.")
    print(f"Average Number of Items Detected: {avg_num_items:.2f}")
    
    # True Positive Detections
    print(f"True Positive Detections:")
    if tp_mean is not None:
        print(f"  Mean Confidence: {tp_mean:.4f} ({tp_mean*100:.2f}%)")
        print(f"  Confidence Distribution:")
        for bin_range, count in tp_bin_counts.items():
            print(f"    {bin_range}: {count}")
    else:
        print("  No true positive detections.")
    
    # False Positive Detections
    print(f"False Positive Detections:")
    if fp_mean is not None:
        print(f"  Mean Confidence: {fp_mean:.4f} ({fp_mean*100:.2f}%)")
        print(f"  Confidence Distribution:")
        for bin_range, count in fp_bin_counts.items():
            print(f"    {bin_range}: {count}")
    else:
        print("  No false positive detections.")
    
    print("Top Most Common False Detections:")
    if most_common_false:
        for cls, count in most_common_false:
            print(f"  {cls}: {count} times")
    else:
        print("  No false detections.")
    print("\n")

print("Analysis complete.")