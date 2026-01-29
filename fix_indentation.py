import sys
import os

def tabs_to_spaces(filepath):
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Check if tabs exist
        if '\t' in content:
            print(f"Converting tabs to spaces in {filepath}")
            new_content = content.replace('\t', '    ')
            with open(filepath, 'w') as f:
                f.write(new_content)
        else:
            print(f"No tabs found in {filepath}")

    except Exception as e:
        print(f"Error processing {filepath}: {e}")

files = [
    'lib/base_models.py',
    'lib/latent_ode.py',
    'lib/ode_rnn.py',
    'lib/rnn_baselines.py',
    'lib/encoder_decoder.py',
    'run_models.py',
    'test_model.py',
    'analyse_std.py'
]

for file in files:
    if os.path.exists(file):
        tabs_to_spaces(file)
    else:
        print(f"File not found: {file}")
