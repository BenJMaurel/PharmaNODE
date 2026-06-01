import pandas as pd
import matplotlib.pyplot as plt

def plot_concentration_data(file_path1, file_path2):
    """
    Reads two CSV files, plots TIME vs. CONC from the first file,
    and TIME vs. DV from the second file on the same graph.

    Args:
        file_path1 (str): The file path for the first CSV file (containing CONC).
        file_path2 (str): The file path for the second CSV file (containing DV).
    """
    try:
        # Read the data from the CSV files into pandas DataFrames
        df1 = pd.read_csv(file_path1)
        df2 = pd.read_csv(file_path2)

        # --- Data Validation ---
        # Check if required columns exist in the first dataframe
        if 'TIME' not in df1.columns or 'DV' not in df1.columns:
            print(f"Error: '{file_path1}' must contain 'TIME' and 'CONC' columns.")
            return

        # Check if required columns exist in the second dataframe
        if 'TIME' not in df2.columns or 'DV' not in df2.columns:
            print(f"Error: '{file_path2}' must contain 'TIME' and 'DV' columns.")
            return
            
        # Drop rows where the plotting values are missing to avoid errors
        
        df1['DV'] = pd.to_numeric(df1['DV'], errors='coerce')
        df2['DV'] = pd.to_numeric(df2['DV'], errors='coerce')
        df1 = df1.dropna(subset=['TIME', 'DV'])
        df2 = df2.dropna(subset=['TIME', 'DV'])
        df1 = df1.rename(columns={'AMOUNT': 'AMT'})

        # Drop rows where essential values are missing
        df1 = df1.dropna(subset=['TIME', 'DV', 'ST', 'AMT'])
        df2 = df2.dropna(subset=['TIME', 'DV', 'ST', 'AMT'])

        # --- Filtering Data ---
        df1_filtered = df1[(df1['ST'] == 1)]
        df2_filtered = df2[(df2['ST'] == 1)]
        import pdb; pdb.set_trace()
        if df1_filtered.empty and df2_filtered.empty:
            print("No data left to plot after applying filters (ST=1, AMT between 3 and 5).")
            return
        df3 = df1_filtered
        df4 = df2_filtered
        
        # --- Plotting ---
        # --- Plotting ---
        plt.style.use('seaborn-v0_8-whitegrid') # Use a nice style for the plot
        fig, ax = plt.subplots(figsize=(10, 6)) # Create a figure and an axes object

        # Plot data from the first file
        ax.plot(df3['TIME'], df3['DV'], marker='o', linestyle='-', label='CONC from file 1', color='b', alpha = 0.5)

        # Plot data from the second file
        ax.plot(df4['TIME'], df4['DV'], marker='s', linestyle='--', label='DV from file 2', color='r', alpha = 0.5)

        # --- Formatting the Plot ---
        ax.set_title('Concentration vs. Time', fontsize=16)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Value (DV / CONC)', fontsize=12)
        ax.legend() # Display the legend
        ax.grid(True) # Ensure the grid is visible

        # Display the plot
        plt.show()

    except FileNotFoundError as e:
        print(f"Error: The file was not found - {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    # --- Instructions for Use ---
    # 1. Make sure you have pandas and matplotlib installed:
    #    pip install pandas matplotlib
    #
    # 2. Save your CSV files in the same directory as this script.
    # 3. Replace 'your_first_file.csv' and 'your_second_file.csv' with the actual filenames.
    
    # Example usage with placeholder file names
    # Please replace these with the actual paths to your CSV files.
    file1 = 'exp_all_runs/simulatedData.csv'
    file2 = 'exp_run_all/42391/virtual_cohort_train.csv'
    
    # Create dummy files for demonstration if they don't exist
    try:
        pd.read_csv(file1)
        pd.read_csv(file2)
    except FileNotFoundError:
        print("Creating dummy CSV files for demonstration purposes...")
        dummy_df1 = pd.DataFrame({
            'TIME': [0, 1, 2, 4, 8, 12, 24],
            'CONC': [0, 10.5, 8.2, 5.1, 2.3, 1.1, 0.2]
        })
        dummy_df2 = pd.DataFrame({
            'TIME': [0, 1.5, 2.5, 4.5, 8.5, 12.5, 24.5],
            'DV': [0, 12.1, 9.8, 6.2, 3.0, 1.5, 0.4]
        })
        dummy_df1.to_csv(file1, index=False)
        dummy_df2.to_csv(file2, index=False)
        print(f"'{file1}' and '{file2}' created.")

    plot_concentration_data(file1, file2)
