import os
import argparse

def extract_code_from_md(md_filepath, output_dir=None):
    """
    Parses a Markdown file and extracts the Python and R code blocks 
    into separate files.
    """
    if not os.path.exists(md_filepath):
        print(f"❌ Error: File '{md_filepath}' not found.")
        return

    # Default to the same directory as the markdown file if not specified
    if output_dir is None:
        output_dir = os.path.dirname(md_filepath) or "."
    
    os.makedirs(output_dir, exist_ok=True)

    # Generate output filenames based on the original markdown filename
    base_name = os.path.splitext(os.path.basename(md_filepath))[0]
    py_out_path = os.path.join(output_dir, f"{base_name}.py")
    r_out_path = os.path.join(output_dir, f"{base_name}.R")

    py_code = []
    r_code = []

    in_python_block = False
    in_r_block = False

    with open(md_filepath, "r", encoding="utf-8") as f:
        for line in f:
            stripped_line = line.strip()

            # Check for the END of a code block
            if stripped_line == "```" and (in_python_block or in_r_block):
                in_python_block = False
                in_r_block = False
                continue

            # Check for the START of a Python block
            if stripped_line.lower().startswith("```python") and not in_python_block:
                in_python_block = True
                continue

            # Check for the START of an R block
            if (stripped_line.lower().startswith("```r") or stripped_line.lower().startswith("``` r")) and not in_r_block:
                in_r_block = True
                continue

            # Accumulate lines if we are inside a block
            if in_python_block:
                py_code.append(line)
            elif in_r_block:
                r_code.append(line)

    # Save the extracted Python code
    if py_code:
        with open(py_out_path, "w", encoding="utf-8") as f:
            f.writelines(py_code)
        print(f"✅ Extracted Python script: {os.path.abspath(py_out_path)}")
    else:
        print("⚠️ No Python code block found in the Markdown file.")

    # Save the extracted R code
    if r_code:
        with open(r_out_path, "w", encoding="utf-8") as f:
            f.writelines(r_code)
        print(f"✅ Extracted R script: {os.path.abspath(r_out_path)}")
    else:
        print("⚠️ No R code block found in the Markdown file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Python and R code blocks from a Markdown file.")
    parser.add_argument("md_file", type=str, help="Path to the input .md file")
    parser.add_argument("--out_dir", type=str, default=None, help="Directory to save the extracted scripts (defaults to same folder as .md file)")
    
    args = parser.parse_args()
    
    print(f"Processing '{args.md_file}'...")
    extract_code_from_md(args.md_file, args.out_dir)