import re

# Define the input and output filenames
input_filename = 'test_gen_tacro_corrected.txt'
output_filename = 'experiment_ids.txt'

try:
    # Open and read the content of the source file
    with open(input_filename, 'r') as f:
        content = f.read()

    # Use a regular expression to find all numbers following "Experiment ID: "
    # \d+ matches one or more digits
    experiment_ids = re.findall(r'Experiment ID: (\d+)', content)

    # Open the output file and write each ID on a new line
    with open(output_filename, 'w') as f:
        for eid in experiment_ids:
            f.write(eid + '\n')

    print(f"✅ Success! Found {len(experiment_ids)} Experiment IDs.")
    print(f"They have been saved to the file: '{output_filename}'")

except FileNotFoundError:
    print(f"❌ Error: The file '{input_filename}' was not found.")
    print("Please make sure the script is in the same directory as the text file.")