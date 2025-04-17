# New number to insert
import os
import shutil
from collections import defaultdict

category = {
    'shoe': 0,
    'paperBox': 1,
    'pastry': 2,
    'penPencil': 3,
    'milkCarton': 4,
    'cutlery':5,
    'crumpledPaper':6,
    'eggShell':7,
    'glassBottle':8,
    'plasticContainer':9,
    'paperEnvelope':10,
    'paperCup':11,  
    'fruit':12,
    'noodlePasta':13,
    'plasticLid':14,
    'plasticMilkBottle':15,
    'sandwich':16,
    'tissueCore': 17,
    'vape': 18,
    'vegeScraps': 19,
    'glassJar':20,
    'meat':21,
    'book':22,
    'plasticBottle':23,
    'alCan':24,
    'softPlastic':25
    }

#'plasticBottle':11

tasks = {
    'test': 0,
    'train': 1,
    'valid': 2
    }

def count_files_in_directory(directory):
    """Count the number of files in a directory."""
    if os.path.exists(directory):
        return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
    else:
        return 0

def ask_for_confirmation(mode):
    prompt=f"Do you want to proceed with {mode}? (yes/no):"
    while True:
        user_input = input(prompt).strip().lower()
        if user_input in ['yes', 'y']:
            return True  # Proceed
        elif user_input in ['no', 'n']:
            return False  # Do not proceed
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

def print_summary(combined_dir):
    """Print the total number of files in the combined directory for each subset and type."""
    for subset in ["train", "valid", "test"]:
        images_dir = os.path.join(combined_dir, subset, "images")
        labels_dir = os.path.join(combined_dir, subset, "labels")

        num_images = count_files_in_directory(images_dir)
        num_labels = count_files_in_directory(labels_dir)

        print(f"{subset.capitalize()}:")
        print(f"  Images: {num_images}")
        print(f"  Labels: {num_labels}")


file_direc = 'path/to/dataset_folder'
base_dir = file_direc
#file_direc = 'dataset'
files = os.listdir(file_direc)

if ask_for_confirmation('changing txt label files'):
    for task in tasks:
        for cats in files:
            if cats in category:
                new_number = category[cats]  # Get new class ID
                print(new_number)

                dirs = os.path.join(file_direc, cats, task, 'labels')
                ans = os.listdir(dirs)

                for file_name in ans:
                    file_path = os.path.join(dirs, file_name)

                    try:
                        with open(file_path, 'r') as file:
                            content = file.readlines()

                        # Modify only the first number (Class ID) of each line
                        modified_content = []
                        for line in content:
                            parts = line.split()  # Split by whitespace
                            if parts:  # Ensure line is not empty
                                parts[0] = str(new_number)  # Change first number (Class ID)
                            modified_content.append(' '.join(parts) + '\n')  # Rejoin the line

                        # Write back the modified content
                        with open(file_path, 'w') as file:
                            file.writelines(modified_content)
                    
                    except FileNotFoundError:
                        print(f"File not found: {file_path}")
                        continue  # Skip to the next file if the current one is missing

                print(f'{cats}-{task}- done')

    print("First number replaced successfully.")
else:
    print("Operation cancelled.")
    # Optionally, exit or handle the cancellation


def merge_files(category):
    category_path = os.path.join(base_dir, category)
    
    for subset in ["test", "train", "valid"]:
        img_src = os.path.join(category_path, subset, "images")
        lbl_src = os.path.join(category_path, subset, "labels")

        img_dst = os.path.join(combined_dir, subset, "images")
        lbl_dst = os.path.join(combined_dir, subset, "labels")

        # Copy images
        for file in os.listdir(img_src):
            src_file_path = os.path.join(img_src, file)
            try:
                shutil.copy(src_file_path, os.path.join(img_dst, file))
            except FileNotFoundError:
                print(f'Error: {src_file_path} - File not found !')
            except OSError as e:
                print(f"Error: Failed to copy {file} - {e}")

        # Copy labels
        if os.path.exists(lbl_src):
            for file in os.listdir(lbl_src):    
                shutil.copy(os.path.join(lbl_src, file), os.path.join(lbl_dst, file))

    print(f'{subset}-done')

# Example usage
if ask_for_confirmation('combining all files'):
    print("Proceeding with the code...")
    
    # Define the destination directory
    combined_dir  = os.path.join(base_dir, 'combined')

    for subset in ["test", "train", "valid"]:
        os.makedirs(os.path.join(combined_dir, subset, "images"), exist_ok=True)
        os.makedirs(os.path.join(combined_dir, subset, "labels"), exist_ok=True)

    # Dictionary to store file counts for each category
    category_counts = {cat: {"images": 0, "labels": 0} for cat in category.keys()}

    for cat in category.keys():
        merge_files(cat)
    
        # Count files for the current category
        for subset in ["test", "train", "valid"]:
            img_dir = os.path.join(combined_dir, subset, "images")
            lbl_dir = os.path.join(combined_dir, subset, "labels")

            category_counts[cat]["images"] += count_files_in_directory(img_dir)
            category_counts[cat]["labels"] += count_files_in_directory(lbl_dir)

        
    # Calculate the total number of images and labels across all categories
    total_images = sum(category_counts[cat]["images"] for cat in category.keys())
    total_labels = sum(category_counts[cat]["labels"] for cat in category.keys())
    print(category_counts)
    print(f"Total images across all categories: {total_images}")
    print(f"Total labels across all categories: {total_labels}")

    print_summary(combined_dir)

    print("Files merged successfully into the 'combined' directory.")


else:
    print("Operation cancelled.")
    # Optionally, exit or handle the cancellation



def count_files_in_directory_1(directory):
    """Count the number of files in a directory efficiently."""
    if os.path.exists(directory):
        return sum(1 for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)))
    else:
        return 0
    

if ask_for_confirmation('checking number of files'):
    combined_dir = os.path.join(base_dir, 'combined')

    # Dictionary to store file counts for each subset and type
    subset_class_counts = defaultdict(lambda: defaultdict(lambda: {"images": 0, "labels": 0}))

    # Dictionary to store file counts for the combined folder
    combined_counts = defaultdict(lambda: {"images": 0, "labels": 0})
        
    for subset in ["test", "train", "valid"]:
        for cat in category.keys():
            cat_dir = os.path.join(base_dir, cat)
            img_dir = os.path.join(cat_dir, subset, "images")
            lbl_dir = os.path.join(cat_dir, subset, "labels")

            # Count files for the current category
            subset_class_counts[subset][cat]["images"] = count_files_in_directory_1(img_dir)
            subset_class_counts[subset][cat]["labels"] = count_files_in_directory_1(lbl_dir)

    # Count files in the combined folder under train, valid, test
    for subset in ["test", "train", "valid"]:
        img_dir = os.path.join(combined_dir, subset, "images")
        lbl_dir = os.path.join(combined_dir, subset, "labels")

        # Count files in the combined folder
        combined_counts[subset]["images"] = count_files_in_directory_1(img_dir)
        combined_counts[subset]["labels"] = count_files_in_directory_1(lbl_dir)

    # Print the total number of files for each class under each subset
    print("File counts for each category:")
    for subset in ["test", "train", "valid"]:
        print(f"Subset: {subset}")
        for cat in category.keys():
            print(f"  Category: {cat}")
            print(f"    Total images: {subset_class_counts[subset][cat]['images']}")
            print(f"    Total labels: {subset_class_counts[subset][cat]['labels']}")
        print()

    # Print the total number of files in the combined folder
    print("File counts in the combined folder:")
    for subset in ["test", "train", "valid"]:
        print(f"Subset: {subset}")
        print(f"  Total images: {combined_counts[subset]['images']}")
        print(f"  Total labels: {combined_counts[subset]['labels']}")
        print()


    # Calculate the total number of images and labels across all subsets
    total_images = sum(combined_counts[subset]["images"] for subset in combined_counts)
    total_labels = sum(combined_counts[subset]["labels"] for subset in combined_counts)

    print(f"Total images across all categories: {total_images}")
    print(f"Total labels across all categories: {total_labels}")

else:
    print("Operation cancelled.")