import pandas as pd
import torch
from transformers import pipeline

# Load the data
data = pd.read_csv('ekdnam_train.csv')

# Define the count of rows to be accessed
count = 100

# Access the first 100 rows of the 'X' column
selected_data = data.loc[:count-1, 'X']

# Initialize the text generation pipeline
generate_text = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

# Define the prompt template
prompt_template = "Given the discharge summary, find the following things about the patient in a structured way. 1. Medication:, 2. Type 3. Stage 4. Admission_Reason. Here is the discharge summary: "

# Initialize a list to store the data and responses
results = []

# Loop through the selected data
for d in selected_data:
    # Concatenate the prompt template with the discharge summary
    full_prompt = prompt_template + d

    # Generate text using the model
    res = generate_text(full_prompt)

    # Append the actual data and generated text to the results list
    results.append({'Actual Data': d, 'Generated Response': res[0]["generated_text"]})

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Write the DataFrame to a CSV file
results_df.to_csv('results/output.csv', index=False)
