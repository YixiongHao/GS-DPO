import json

# Load json file
file = "evaluation_dataset.json"

with open(file, "r") as f:
    dataset = json.load(f)

# Create a flattened version for easier use
flattened_dataset = {
    "metadata": dataset["metadata"],
    "questions": {'benign': [], 'harmful': []},
}

# Flatten benign questions
for question_data in dataset["benign_questions"]:
    for variation in question_data["variations"]:
        flattened_dataset["questions"]['benign'].append(variation)

# Flatten harmful questions
for question_data in dataset["harmful_questions"]:
    for variation in question_data["variations"]:
        flattened_dataset["questions"]['harmful'].append(variation)

# Save the flattened dataset
flattened_output = "eval_data_flattended.json"
with open(flattened_output, "w") as f:
    json.dump(flattened_dataset, f, indent=2)