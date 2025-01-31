import os
label_directory2 = "people_with_guns/valid/labels"
label_directory1 = "people_with_guns/train/labels"
label_directory3 = "people_with_guns/test/labels"


# Label file direath/to/your/labels"

# Process all label files in the directory
for filename in os.listdir(label_directory):
    if filename.endswith(".txt"):  # YOLO label files have a .txt extension
        file_path = os.path.join(label_directory, filename)

        with open(file_path, "r") as file:
            lines = file.readlines()

        # Change the class index to 1 in each line
        updated_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0:
                parts[0] = "1"  # Set the first value (class index) to 1
            updated_lines.append(" ".join(parts))

        # Overwrite the file with the updated content
        with open(file_path, "w") as file:
            file.write("\n".join(updated_lines) + "\n")

print("All class indices have been changed from 0 to 1.")

