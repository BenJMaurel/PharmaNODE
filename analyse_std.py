import re
import numpy as np
from scipy import stats
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_model_results(results_text):
    """
    Parses the given results text to extract error and RMSE values,
    then calculates and prints the mean and standard deviation for each metric.

    Args:
        results_text (str): A string containing the model training results
                            for different seeds.
    """
    # Initialize lists to store extracted values
    errors = []
    errors_be = []
    rmses = []
    rmses_be = []

    # Regular expressions to find the metrics and their values
    error_pattern = re.compile(r'-\s*error\s*=\s*([-\d.]+)')
    error_be_pattern = re.compile(r'-\s*error_be\s*=\s*([-\d.]+)')
    rmse_pattern = re.compile(r'-\s*rmse\s*=\s*([-\d.]+)')
    rmse_be_pattern = re.compile(r'-\s*rmse_be\s*=\s*([-\d.]+)')

    # Process each line in the input text
    for line in results_text.splitlines():
        # Search for error values
        match = error_pattern.search(line)
        if match:
            try:
                errors.append(float(match.group(1)))
            except ValueError:
                print(f"Warning: Could not parse error value from line: {line}")
            continue

        match = error_be_pattern.search(line)
        if match:
            try:
                errors_be.append(float(match.group(1)))
            except ValueError:
                print(f"Warning: Could not parse error_be value from line: {line}")
            continue

        # Search for RMSE values
        match = rmse_pattern.search(line)
        if match:
            try:
                rmses.append(float(match.group(1)))
            except ValueError:
                print(f"Warning: Could not parse rmse value from line: {line}")
            continue

        match = rmse_be_pattern.search(line)
        if match:
            try:
                rmses_be.append(float(match.group(1)))
            except ValueError:
                print(f"Warning: Could not parse rmse_be value from line: {line}")
            continue

    # Calculate and print statistics
    metrics = {
        "Error": errors,
        "Error BE": errors_be,
        "RMSE": rmses,
        "RMSE BE": rmses_be
    }

    print("\n--- Analysis Results ---")
    print("-" * 24)

    for name, values in metrics.items():
        if values: # Ensure there are values to calculate statistics
            mean_val = np.abs(np.mean(values))
            std_val = np.std(values)
            print(f"Metric: {name}")
            print(f"  Mean: {mean_val:.4f}")
            print(f"  Standard Deviation: {std_val:.4f}")
            print("-" * 24)
        else:
            print(f"No data found for {name}.")
            print("-" * 24)
    return metrics

def perform_t_tests(metrics):
    """
    Performs paired t-tests between 'Error' and 'Error BE', and 'RMSE' and 'RMSE BE'.

    Args:
        metrics (dict): Dictionary containing lists of metric values.
    """
    print("\n--- Paired T-Tests ---")
    print("-" * 24)

    # Paired t-test for Error vs. Error BE
    errors = metrics.get('Error', [])
    errors_be = metrics.get('Error BE', [])
    if len(errors) > 1 and len(errors) == len(errors_be): # Need at least 2 samples
        t_statistic_er, p_value_er = stats.ttest_rel(errors, errors_be)
        print(f"Paired t-test for Error vs. Error BE:")
        print(f"  T-statistic: {t_statistic_er:.4f}")
        print(f"  P-value: {p_value_er:.4f}")
        if p_value_er <= 0.05:
            print("  The difference is statistically significant (p <= 0.05).")
        else:
            print("  The difference is NOT statistically significant (p > 0.05).")
        print("-" * 24)
    else:
        print("Not enough data for Paired t-test for Error vs. Error BE or unequal lengths.")
        print("-" * 24)


    # Paired t-test for RMSE vs. RMSE BE
    rmses = metrics.get('RMSE', [])
    rmses_be = metrics.get('RMSE BE', [])
    if len(rmses) > 1 and len(rmses) == len(rmses_be):
        t_statistic_rmse, p_value_rmse = stats.ttest_rel(rmses, rmses_be)
        print(f"Paired t-test for RMSE vs. RMSE BE:")
        print(f"  T-statistic: {t_statistic_rmse:.4f}")
        print(f"  P-value: {p_value_rmse:.4f}")
        if p_value_rmse <= 0.05:
            print("  The difference is statistically significant (p <= 0.05).")
        else:
            print("  The difference is NOT statistically significant (p > 0.05).")
        print("-" * 24)
    else:
        print("Not enough data for Paired t-test for RMSE vs. RMSE BE or unequal lengths.")
        print("-" * 24)

def create_boxplots(metrics, output_filename="model_performance_boxplots_all.png"):
    """
    Generates and saves a figure with two box plots, each with its own
    independent y-axis. A horizontal line at y=0 is added to each.

    Args:
        metrics (dict): Dictionary containing lists of metric values.
        output_filename (str): The name of the file to save the plot.
    """
    print("\n--- Generating Box Plots ---")
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # ✨ REMOVED sharey=True from this line
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # fig.suptitle('Distribution of Model Performance Metrics Across Runs', fontsize=16)

    # --- Prepare data and plot for Error metrics ---
    error_data = []
    if metrics.get('Error') and metrics.get('Error BE'):
        error_data.extend([{'Metric': 'Error', 'Value': v} for v in metrics['Error']])
        error_data.extend([{'Metric': 'Error BE', 'Value': v} for v in metrics['Error BE']])
        error_df = pd.DataFrame(error_data)

        sns.boxplot(ax=axes[0], x='Metric', y='Value', data=error_df, palette="pastel")
        axes[0].axhline(0, color='gray', linestyle='--', linewidth=1.5)
        axes[0].set_title('Error vs. Error BE')
        axes[0].set_xlabel('')
        axes[0].set_ylabel('Value')
    else:
        axes[0].text(0.5, 0.5, 'Not enough data for Error plot', ha='center', va='center')
        axes[0].set_title('Error vs. Error BE')


    # --- Prepare data and plot for RMSE metrics ---
    rmse_data = []
    if metrics.get('RMSE') and metrics.get('RMSE BE'):
        rmse_data.extend([{'Metric': 'RMSE', 'Value': v} for v in metrics['RMSE']])
        rmse_data.extend([{'Metric': 'RMSE BE', 'Value': v} for v in metrics['RMSE BE']])
        rmse_df = pd.DataFrame(rmse_data)
        
        sns.boxplot(ax=axes[1], x='Metric', y='Value', data=rmse_df, palette="pastel")
        axes[1].axhline(0, color='gray', linestyle='--', linewidth=1.5)
        axes[1].set_title('RMSE vs. RMSE BE')
        axes[1].set_xlabel('')
        # ✨ RE-ADDED the y-label for the second, independent axis
        axes[1].set_ylabel('Value') 
    else:
        axes[1].text(0.5, 0.5, 'Not enough data for RMSE plot', ha='center', va='center')
        axes[1].set_title('RMSE vs. RMSE BE')


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    try:
        plt.savefig(output_filename, dpi=300)
        print(f"Box plots saved successfully as '{output_filename}'")
    except Exception as e:
        print(f"Error saving the plot: {e}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <path_to_results_file>")
        sys.exit(1)

    results_file_path = sys.argv[1]

    try:
        with open(results_file_path, 'r') as f:
            full_results_text = f.read()
    except FileNotFoundError:
        print(f"Error: Results file not found at {results_file_path}")
        sys.exit(1)

    # Run the full analysis
    extracted_metrics = analyze_model_results(full_results_text)
    perform_t_tests(extracted_metrics)
    create_boxplots(extracted_metrics)