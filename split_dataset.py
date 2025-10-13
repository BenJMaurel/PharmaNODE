import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import os

def split_dataset(file_path, output_dir='splits', test_size=0.2, random_seed=None):
    """
    Splits a dataset into a single training and testing set.

    Args:
        file_path (str): The path to the input CSV file.
        output_dir (str): The directory where the split files will be saved.
        test_size (float): The proportion of the dataset to include in the test split.
        random_seed (int): The seed for the random number generator.
    """
    # Load the dataset
    df = pd.read_csv(file_path)

    # Get unique patient IDs
    patient_ids = df['ID'].unique()

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Split patient IDs into training and testing sets
    train_ids, test_ids = train_test_split(patient_ids, test_size=test_size, random_state=random_seed)

    train_df = df[df['ID'].isin(train_ids)]
    test_df = df[df['ID'].isin(test_ids)]

    train_file = os.path.join(output_dir, f'train_split.csv')
    test_file = os.path.join(output_dir, f'test_split.csv')

    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f'Split with random seed {random_seed}:')
    print(f'  Train set size: {len(train_df)} rows, {len(train_ids)} patients')
    print(f'  Test set size: {len(test_df)} rows, {len(test_ids)} patients')
    print(f'  Train file: {train_file}')
    print(f'  Test file: {test_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split a dataset into a single training and testing set.')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--output_dir', type=str, default='splits', help='Directory to save the split files.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of the dataset to include in the test split.')
    parser.add_argument('--random_seed', type=int, default=None, help='Seed for the random number generator.')
    args = parser.parse_args()

    split_dataset(args.file_path, args.output_dir, args.test_size, args.random_seed)