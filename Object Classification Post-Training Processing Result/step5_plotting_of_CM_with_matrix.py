import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

conf_tre = 0.4
# Confusion matrix
cm = np.array([
[108, 3, 0, 0, 0, 0, 0, 1, 0, 3, 0, 8, 4, 0, 2, 0, 0, 8, 4, 0, 0, 0, 0, 0, 0, 0, 13],
[0, 90, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 24, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 28],
[0, 0, 181, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 67, 0, 0, 0, 0, 46, 0, 0, 0, 19],
[0, 1, 0, 262, 0, 0, 0, 0, 0, 1, 0, 0, 1, 13, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 26],
[0, 0, 6, 0, 96, 16, 0, 0, 10, 0, 0, 1, 1, 0, 22, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 51],
[0, 0, 2, 0, 3, 174, 0, 1, 7, 0, 3, 0, 0, 0, 10, 0, 0, 0, 2, 0, 14, 0, 0, 0, 0, 5, 81],
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

names = ['aluminiumCan', 'book', 'crumpledPaper', 'cutlery',
         'eggShell', 'fruit', 'glassBottle', 'glassJar', 'meat',
         'milkCarton', 'noodlePasta', 'paperBox', 'paperCup',
         'paperEnvelope', 'pastry', 'penPencil', 'plasticBottle',
         'plasticContainer', 'plasticLid', 'plasticMilkBottle',
         'sandwich','shoe', 'softPlastic', 'tissueCore',
         'vape', 'vegeScraps', 'background']

# Plot confusion matrix
plt.figure(figsize=(16, 12))
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#cmn = cm
sns.heatmap(cmn, annot=True, cmap="Blues", fmt=".2f", cbar=True, xticklabels=names, yticklabels=names)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title(f"Confusion Matrix for YOLO: Vision Object Detection, Confidence Threshold : {conf_tre}")
plt.savefig("norm_yolo_cm.pdf", format="pdf", bbox_inches="tight")
plt.show()





cm = np.array([

[580, 72, 11, 72, 0, 99],
[13, 783, 8, 75, 0, 177],
[9, 1, 1311, 7, 0, 502],
[88, 28, 17, 1341, 0, 177],
[4, 15, 3, 41, 22, 27],
[0, 0, 0, 0, 0, 0]

])




names = ["Paper", "Mixed Recycling", "Food Waste", "General Waste", "Prohibited", "Background"]

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#cmn = cm
sns.heatmap(cmn, annot=True, cmap="Blues", fmt=".2f", cbar=True, xticklabels=names, yticklabels=names)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title(f"Confusion Matrix for YOLO: Vision Object Detection, Confidence Threshold : {conf_tre}")
plt.savefig("norm_yolo_cm.pdf", format="pdf", bbox_inches="tight")
plt.show()
