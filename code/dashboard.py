import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
import json
import subprocess
import time

def get_available_experiments():
    base_dir = "results"
    if not os.path.exists(base_dir):
        return []
    
    experiments = []
    for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path):
            if os.path.exists(os.path.join(full_path, "predictions.csv")):
                experiments.append(entry)
    
    return sorted(experiments)

def get_patients_for_experiment(exp_id):
    pred_path = os.path.join("results", exp_id, "predictions.csv")
    if not os.path.exists(pred_path):
        return []
    
    df = pd.read_csv(pred_path)
    if 'patient_id' in df.columns:
        return sorted(df['patient_id'].unique().tolist())
    return []

def get_model_config(exp_id):
    config_path = os.path.join("results", exp_id, "drug_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            try:
                data = json.load(f)
                return json.dumps(data, indent=2)
            except json.JSONDecodeError:
                return '{"error": "Could not parse drug_config.json"}'
    else:
        return '{\n  "info": "Legacy Tacrolimus Model",\n  "note": "Config file not found - assuming 2-compartment oral model"\n}'

from matplotlib.figure import Figure

def plot_patient(exp_id, patient_id, plot_horizon):
    exp_dir = os.path.join("results", exp_id)
    pred_path = os.path.join(exp_dir, "predictions.csv")
    obs_path = os.path.join(exp_dir, "observations.csv")
    true_path = os.path.join(exp_dir, "virtual_cohort_test.csv")
    
    if not os.path.exists(pred_path):
        return None
        
    fig = Figure(figsize=(10, 6), dpi=120)
    ax = fig.add_subplot(111)
    
    # 1 & 2. Plot True Trajectory, Encoder Inputs, and Verification Points
    if os.path.exists(true_path) and os.path.exists(obs_path):
        obs_df = pd.read_csv(obs_path)
        patient_obs = obs_df[obs_df['patient_id'] == int(patient_id)]
        
        true_df = pd.read_csv(true_path)
        true_df['DV'] = pd.to_numeric(true_df['DV'], errors='coerce')
        true_df = true_df.dropna(subset=['TIME', 'DV'])
        # FIX: Filter by mdv=0 instead of ST=1 because batched_PK sets ST=0
        if 'mdv' in true_df.columns:
            patient_true = true_df[(true_df['ID'] == int(patient_id)) & (true_df['mdv'] == 0)]
        else:
            patient_true = true_df[(true_df['ID'] == int(patient_id))]
        
        patient_true = patient_true[patient_true['TIME'] >= 0] 
        
        # Split into encoded vs verification using tolerance for floats against observations.csv
        tol = 1e-4
        obs_times = patient_obs['time'].values if not patient_obs.empty else []
        def is_encoded(t):
            return any(abs(t - st) < tol for st in obs_times)
            
        encoded_mask = patient_true['TIME'].apply(is_encoded)
        verification_points = patient_true[~encoded_mask]
        
        if not patient_true.empty:
            ax.plot(patient_true['TIME'], patient_true['DV'], 
                    color='blue', linestyle='-', alpha=0.3, label='Ground Truth (Simulated Trajectory)')         
        
        # Plot the actual noisy observations that went into the model
        if not patient_obs.empty:
            ax.scatter(patient_obs['time'], patient_obs['value'], 
                       color='red', marker='X', s=100, zorder=5, label='Encoder Input Points (Observations)')
                       
        if not verification_points.empty:
            ax.scatter(verification_points['TIME'], verification_points['DV'], 
                       color='orange', marker='o', s=50, zorder=4, label='Verification Points (Ground Truth)')

    # 3. Plot Model Prediction
    pred_df = pd.read_csv(pred_path)
    patient_pred = pred_df[pred_df['patient_id'] == int(patient_id)]
    
    if not patient_pred.empty:
        ax.plot(patient_pred['time'], patient_pred['prediction'], 
                color='orange', linewidth=2.5, linestyle='--', label='Latent ODE Prediction')
        
        if 'lower_ci' in patient_pred.columns and 'upper_ci' in patient_pred.columns:
            ax.fill_between(patient_pred['time'], patient_pred['lower_ci'], patient_pred['upper_ci'], 
                            color='orange', alpha=0.2, label='Confidence Interval')

    ax.set_title(f"PK Profile for Patient {patient_id} (Experiment {exp_id})", fontsize=14)
    ax.set_xlabel("Time (h)", fontsize=12)
    ax.set_ylabel("Concentration (mg/L)", fontsize=12)
    
    horizon_val = 12 if "12h" in plot_horizon else 24
    ax.set_xlim(0, horizon_val)
    
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='best')
    
    fig.tight_layout()
    return fig

def get_latent_plot(exp_id):
    pca_plot = os.path.join("results", exp_id, "latent_pca.png")
    if os.path.exists(pca_plot):
        return pca_plot
    return None

def update_patient_dropdown(exp_id):
    patients = get_patients_for_experiment(exp_id)
    if not patients:
        return gr.update(choices=[], value=None)
    return gr.update(choices=patients, value=patients[0])

def update_dashboard(exp_id, patient_id, plot_horizon):
    if not exp_id or not patient_id:
        config_text = get_model_config(exp_id) if exp_id else "{}"
        return None, None, config_text
    
    fig = plot_patient(exp_id, patient_id, plot_horizon)
    latent_img = get_latent_plot(exp_id)
    config_text = get_model_config(exp_id)
    
    return fig, latent_img, config_text

def toggle_custom_inputs(gen_type):
    if gen_type == "Custom Generic PK":
        return [gr.update(visible=True)] * 10
    return [gr.update(visible=False)] * 10

def train_model_ui(exp_id, gen_type, compartments, absorption, distribution, pop_cl, pop_vc, pop_q, pop_vp, res_prop, res_add, covariates, num_patients, niters, use_film):
    if not exp_id:
        yield "Error: Experiment ID cannot be empty.\n", gr.update()
        return
        
    output_text = f"--- Starting Pipeline for Experiment {exp_id} ---\n\n"
    yield output_text, gr.update()
    
    # 1. Generate Data
    if gen_type == "Custom Generic PK":
        gen_cmd = ["python3", "gen_generic_pk.py", "--exp", str(exp_id), 
                   "--num_patients", str(num_patients), 
                   "--compartments", str(compartments), 
                   "--absorption", absorption,
                   "--distribution", distribution,
                   "--pop_cl", str(pop_cl),
                   "--pop_vc", str(pop_vc),
                   "--pop_q", str(pop_q),
                   "--pop_vp", str(pop_vp),
                   "--res_prop", str(res_prop),
                   "--res_add", str(res_add),
                   "--covariates", str(covariates)]
    else:
        gen_script = "gen_tacro_film.py" if use_film else "gen_tacro.py"
        gen_cmd = ["python3", gen_script, "--exp", str(exp_id), "--num_patients", str(num_patients)]
    
    output_text += f"> Running: {' '.join(gen_cmd)}\n"
    yield output_text, gr.update()
    
    try:
        process = subprocess.Popen(gen_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in iter(process.stdout.readline, ''):
            output_text += line
            yield output_text, gr.update()
        process.stdout.close()
        process.wait()
        if process.returncode != 0:
            output_text += f"\nError: Data generation failed with exit code {process.returncode}\n"
            yield output_text, gr.update()
            return
    except Exception as e:
        output_text += f"\nException during data generation: {str(e)}\n"
        yield output_text, gr.update()
        return

    # 2. Train Latent ODE
    output_text += f"\n\n> Data Generation Complete. Starting Training...\n"
    yield output_text, gr.update()
    
    train_cmd = [
        "python3", "run_models.py", 
        "--dataset", "PK_Tacro", 
        "--latent-ode",
        "--experiment", str(exp_id),
        "--niters", str(niters),
        "-n", "50", "-s", "40", "-l", "10", 
        "--noise-weight", "0.01", "--max-t", "5."
    ]
    if use_film and gen_type != "Custom Generic PK":
        train_cmd.append("--use_film")
        
    output_text += f"> Running: {' '.join(train_cmd)}\n"
    yield output_text, gr.update()
    
    try:
        process = subprocess.Popen(train_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in iter(process.stdout.readline, ''):
            output_text += line
            yield output_text, gr.update()
        process.stdout.close()
        process.wait()
        if process.returncode != 0:
            output_text += f"\nError: Training failed with exit code {process.returncode}\n"
            yield output_text, gr.update()
            return
    except Exception as e:
        output_text += f"\nException during training: {str(e)}\n"
        yield output_text, gr.update()
        return

    # 3. Predict and Plot Latent Space
    output_text += f"\n\n> Training Complete. Generating Predictions and Latent Space Plots...\n"
    yield output_text, gr.update()
    
    test_cmd = ["python3", "test_model.py", "--load", str(exp_id), "--dataset", "PK_Tacro", "--latent-ode", "-n", "50", "-s", "40", "-l", "10", "--noise-weight", "0.01", "--max-t", "5."]
    if use_film and gen_type != "Custom Generic PK":
        test_cmd.append("--use_film")
    
    output_text += f"> Running: {' '.join(test_cmd)}\n"
    yield output_text, gr.update()
    
    try:
        process = subprocess.Popen(test_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in iter(process.stdout.readline, ''):
            output_text += line
            yield output_text, gr.update()
        process.stdout.close()
        process.wait()
        if process.returncode != 0:
            output_text += f"\nError: Evaluation failed with exit code {process.returncode}\n"
            yield output_text, gr.update()
            return
    except Exception as e:
        output_text += f"\nException during evaluation: {str(e)}\n"
        yield output_text, gr.update()
        return
        
    output_text += f"\n\n--- Pipeline Complete for {exp_id} ---\n"
    new_choices = get_available_experiments()
    yield output_text, gr.update(choices=new_choices, value=str(exp_id))

# ==========================================
# Gradio UI Layout
# ==========================================
with gr.Blocks() as demo:
    gr.Markdown("# 💊 PharmaNODE Interactive Dashboard")
    gr.Markdown("Visualize pre-trained Latent ODE models, compare predictions against true PK trajectories, explore latent spaces, and train new models.")
    
    with gr.Tabs():
        with gr.Tab("Inference & Visualization"):
            available_exps = get_available_experiments()
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Select Model & Patient")
                    exp_dropdown = gr.Dropdown(choices=available_exps, 
                                               value=available_exps[0] if available_exps else None, 
                                               label="Select Pre-trained Model (Experiment ID)")
                    
                    init_patients = get_patients_for_experiment(available_exps[0]) if available_exps else []
                    patient_dropdown = gr.Dropdown(choices=init_patients,
                                                   value=init_patients[0] if init_patients else None,
                                                   label="Select Patient ID")
                                                   
                    plot_horizon = gr.Radio(choices=["12h", "24h"], value="24h", label="Plot Horizon")
                    
                    gr.Markdown("### 2. Model Configuration")
                    config_viewer = gr.Code(language="json", label="drug_config.json")
                    
                with gr.Column(scale=2):
                    gr.Markdown("### 3. Patient PK Profile")
                    pk_plot = gr.Plot(label="Time vs. Concentration")
                    
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 4. Latent Space Visualization")
                    gr.Markdown("PCA projection of the `z0` latent means for the test cohort. If this is empty, you need to run `plot_latent_space.py` for this experiment.")
                    latent_plot = gr.Image(type="filepath", label="Latent Space (PCA)")

            # Callbacks
            exp_dropdown.change(
                fn=update_patient_dropdown,
                inputs=[exp_dropdown],
                outputs=[patient_dropdown]
            ).then(
                fn=update_dashboard,
                inputs=[exp_dropdown, patient_dropdown, plot_horizon],
                outputs=[pk_plot, latent_plot, config_viewer]
            )
            
            patient_dropdown.change(
                fn=update_dashboard,
                inputs=[exp_dropdown, patient_dropdown, plot_horizon],
                outputs=[pk_plot, latent_plot, config_viewer]
            )
            
            plot_horizon.change(
                fn=update_dashboard,
                inputs=[exp_dropdown, patient_dropdown, plot_horizon],
                outputs=[pk_plot, latent_plot, config_viewer]
            )

            # Initial load trigger
            demo.load(
                fn=update_dashboard,
                inputs=[exp_dropdown, patient_dropdown, plot_horizon],
                outputs=[pk_plot, latent_plot, config_viewer]
            )

        with gr.Tab("Train New Model"):
            gr.Markdown("### Train a Latent ODE from Scratch")
            gr.Markdown("This interface automates data generation and Latent ODE training. **Warning: Depending on the iterations, this may take several minutes.**")
            
            with gr.Row():
                with gr.Column(scale=1):
                    train_exp_id = gr.Textbox(label="Experiment ID (e.g., test_run_1)", placeholder="Enter a unique ID...")
                    gen_type = gr.Dropdown(choices=["Standard Tacrolimus (Legacy)", "Custom Generic PK"], value="Standard Tacrolimus (Legacy)", label="Data Generation Type")
                    
                    # Custom PK fields (hidden by default)
                    custom_compartments = gr.Dropdown(choices=[1, 2, 3], value=2, label="Number of Compartments", visible=False)
                    custom_absorption = gr.Dropdown(choices=["oral", "iv"], value="oral", label="Absorption Type", visible=False)
                    custom_distribution = gr.Dropdown(choices=["log-normal", "normal"], value="log-normal", label="PK Parameter Distribution", visible=False)
                    custom_pop_cl = gr.Number(value=10.5, label="Population Median Clearance (L/h)", visible=False)
                    custom_pop_vc = gr.Number(value=45.0, label="Population Median Central Vol (L)", visible=False)
                    custom_pop_q = gr.Number(value=15.0, label="Inter-compartmental Clearance", visible=False)
                    custom_pop_vp = gr.Number(value=60.0, label="Peripheral Volume", visible=False)
                    custom_res_prop = gr.Number(value=0.1, label="Proportional Residual Error (e.g. 0.1=10%)", visible=False)
                    custom_res_add = gr.Number(value=0.01, label="Additive Residual Error (mg/L)", visible=False)
                    custom_covariates = gr.Textbox(value="WT:50-100", label="Covariates to Simulate with Ranges (e.g., WT:50-100, AGE:20-80)", visible=False)
                    
                    train_patients = gr.Number(value=200, label="Number of Patients to Generate")
                    train_iters = gr.Number(value=100, label="Training Iterations")
                    train_film = gr.Checkbox(value=False, label="Use FiLM Pipeline (Paired Visits)")
                    train_btn = gr.Button("Start Training Pipeline", variant="primary")
                    
                with gr.Column(scale=2):
                    train_output = gr.Textbox(label="Terminal Output", lines=25, max_lines=25, autoscroll=True)
            
            gen_type.change(
                fn=toggle_custom_inputs,
                inputs=[gen_type],
                outputs=[custom_compartments, custom_absorption, custom_distribution, custom_pop_cl, custom_pop_vc, custom_pop_q, custom_pop_vp, custom_res_prop, custom_res_add, custom_covariates]
            )
            
            train_btn.click(
                fn=train_model_ui,
                inputs=[train_exp_id, gen_type, custom_compartments, custom_absorption, custom_distribution, custom_pop_cl, custom_pop_vc, custom_pop_q, custom_pop_vp, custom_res_prop, custom_res_add, custom_covariates, train_patients, train_iters, train_film],
                outputs=[train_output, exp_dropdown]
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, theme=gr.themes.Soft())
