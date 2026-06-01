import streamlit as st
from google import genai
import os
from datetime import datetime

# ==========================================
# 1. Page Configuration & UI Setup
# ==========================================
st.set_page_config(page_title="Pipeline-Safe PK Generator", page_icon="⚙️", layout="wide")

st.title("⚙️ Pipeline-Safe PK Script Generator")
st.markdown("This tool uses your existing Tacrolimus scripts as strict templates to ensure downstream pipeline compatibility.")

api_key = st.text_input("Enter your Gemini API Key:", type="password")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. New Monolix (mlxtran) Model")
    mlxtran_input = st.text_area("Paste the NEW .txt file content here:", height=300)

with col2:
    st.subheader("2. New mrgsolve Block")
    mrgsolve_input = st.text_area("Paste the NEW [PROB] to $CAPTURE block here:", height=300)

# ==========================================
# 2. Template Loading
# ==========================================
# Load the user's provided files to use as strict skeletons
try:
    with open("gen_tacro.py", "r") as f:
        py_template = f.read()
    with open("all_run_tacro.r", "r") as f:
        r_template = f.read()
    templates_loaded = True
except FileNotFoundError:
    st.error("⚠️ Could not find `gen_tacro.py` or `all_run_tacro.r`. Please ensure they are in the same folder as this app.")
    templates_loaded = False

# ==========================================
# 3. Prompt Template Definition
# ==========================================
def build_prompt(mlxtran_code, mrgsolve_code, py_temp, r_temp):
    return f"""
    You are an expert Pipeline Engineer and Pharmacometrician. 
    Your task is to update an existing simulation and estimation pipeline with a NEW pharmacokinetic model.

    CRITICAL CONSTRAINT: These scripts are part of an automated pipeline. You MUST preserve the exact architecture, argument parsing (`argparse`), file naming conventions, random splits (`train_test_split`), mapbayr loops, and CSV column structures of the provided templates. 

    DO NOT change how the scripts take inputs or save outputs. 
    YOUR ONLY JOB is to swap out the underlying mathematical model (ODEs, parameters, covariates, compartments) to match the new inputs.

    ---------------------------------------------------------
    === NEW MODEL INPUTS ===
    ---------------------------------------------------------
    Input 1: New Monolix (mlxtran) Model
    ```text
    {mlxtran_code}
    Input 2: New mrgsolve Code Block
    Plaintext

    {mrgsolve_code}

    === PYTHON TEMPLATE (gen_tacro.py) ===
    Use this exact skeleton. Keep argparse, train_test_split, and save logic.
    Update the POPULATION_PARAMS, IPV_OMEGA, __init__, forward (ODEs), and _sample_individual_parameters to reflect the NEW model inputs.
    Python

    {py_temp}

    === R TEMPLATE (all_run_tacro.r) ===
    Use this exact skeleton. Keep argparse, furrr parallelization, plot saving, and CSV saving logic.
    Update the column_mapping, setIndividualParameterModel, code_tac (inline mrgsolve), mod_tac_updatedmapping, and the get_param extraction to reflect the NEW model inputs.
    R

    {r_temp}
    Formatting Rules:
    * Output exactly two code blocks: one for the new Python script, one for the new R script.
    * Ensure all new parameters map perfectly between the PyTorch class and the mrgsolve block.
    * Do not add conversational filler. Output the code blocks directly."""

if st.button("Generate Pipeline-Safe Scripts", type="primary"):
    if not api_key:
        st.error("Please enter your Gemini API Key.")
    elif not mlxtran_input or not mrgsolve_input:
        st.error("Please provide both the mlxtran and mrgsolve code.")
    elif not templates_loaded:
        st.error("Templates not loaded. Cannot proceed.")
    else:
        with st.spinner("Injecting new math into pipeline templates... This will take about 60 seconds."):
            try:
                client = genai.Client(api_key=api_key)
                final_prompt = build_prompt(mlxtran_input, mrgsolve_input, py_template, r_template)
                            # Using Gemini Pro as it excels at following complex system constraints and large contexts
                try:
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=final_prompt,
                    )
                except:
                    response = client.models.generate_content(
                        model='gemma-4-26b-a4b-it',
                        contents=final_prompt,
                    )
                llm_output = response.text
                
                st.success("Scripts generated successfully!")
                st.markdown("### Generated Output")
                st.markdown(llm_output)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_dir = "generated_scripts"
                os.makedirs(save_dir, exist_ok=True)
                
                file_path = os.path.join(save_dir, f"pipeline_scripts_{timestamp}.md")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(llm_output)
                    
                st.info(f"💾 Output saved to your computer at: `{os.path.abspath(file_path)}`")
                            
            except Exception as e:
                st.error(f"An error occurred: {e}")