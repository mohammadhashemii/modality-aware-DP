import json
import re

# Load the original JSON file
input_file = 'fashion_indio/data/val_data.json'
output_file = 'preprocess_val_data.json'

# Function to remove class label from the product title
def remove_class_label_from_title(title, class_label):
    # Use regex to find and remove the class label followed by optional space
    pattern = re.compile(rf"\\b{re.escape(class_label)}\\b", re.IGNORECASE)
    return pattern.sub("", title).strip()

try:
    # Read the input JSON file
    with open(input_file, 'r') as infile:
        data = [json.loads(line) for line in infile]

    # Process the data to remove the class label from the product title
    for item in data:
        item['product_title'] = remove_class_label_from_title(item['product_title'], item['class_label'])

    # Save the processed data to a new JSON file
    with open(output_file, 'w') as outfile:
        for item in data:
            outfile.write(json.dumps(item) + '\n')

    print(f"Processed data has been saved to {output_file}")

except Exception as e:
    print(f"An error occurred: {e}")
