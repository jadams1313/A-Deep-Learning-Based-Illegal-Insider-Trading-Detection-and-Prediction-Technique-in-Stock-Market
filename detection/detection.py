#!/usr/bin/env python3
"""
Author: Sheikh Rabiul Islam (Converted to Python)
Date: 11/20/2017 (Converted: 2025)
Purpose: Discrete signal (time series) similarity measures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate
import os
import math

class TimeSeriesSimilarity:
    def __init__(self):
        # Global variables equivalent
        self.window_size = 50
        self.overlap = 10
        self.actual_data = None
        self.predicted_data_window_based = None
        self.predicted_data_day_based = None
        self.predicted_data_historical_based = None
        self.pattern_desc_arr = None
        self.pattern_mat = None
        self.result = []
        
        # Configuration
        self.num_method = 3
        self.num_pattern = 11
    
    def handle_pattern_data(self):
        with open('input/pattern.csv', 'r') as f:
            lines = f.readlines()

        data = []
        max_cols = 0

        for line in lines:
            row = [x.strip() for x in line.strip().split(',') if x.strip()]
            max_cols = max(max_cols, len(row))

        for line in lines:
            row = [x.strip() for x in line.strip().split(',')]
            row_floats = []
            for item in row:
                if item.strip() == '':
                    row_floats.append(0.0)
                else:
                    try:
                        row_floats.append(float(item))
                    except ValueError:
                        row_floats.append(0.0)
            
            # Pad with zeros to reach max_cols
            while len(row_floats) < max_cols:
                row_floats.append(0.0)
            
            data.append(row_floats)

        # Create DataFrame
        df = pd.DataFrame(data)
        return df.values
        
    def load_data(self):
        """Load data from CSV files"""
        try:
            self.actual_data = pd.read_csv('input/bp_full_window_act.csv', header=None).values.flatten()
            self.predicted_data_window_based = pd.read_csv('input/bp_full_window_pred.csv', header=None).values.flatten()
            self.predicted_data_day_based = pd.read_csv('input/bp_full_point_pred.csv', header=None).values.flatten()
            self.predicted_data_historical_based = pd.read_csv('input/bp_full_sequence_pred.csv', header=None).values.flatten()
            self.pattern_mat = self.handle_pattern_data()
            self.pattern_desc_arr = pd.read_csv('input/pattern_sizes.csv', header=None).values.flatten()
            
            print("Data loaded successfully")
            print(f"Actual data shape: {self.actual_data.shape}")
            print(f"Pattern matrix shape: {self.pattern_mat.shape}")
            
        except FileNotFoundError as e:
            print(f"Error loading files: {e}")
            print("Please ensure all input CSV files are in the 'input' directory")
    
    def normalized_cross_correlation(self, method, window, pattern, day):
        """
        Compute normalized cross correlation between signals
        
        Args:
            method (int): Prediction method (1, 2, or 3)
            window (int): Window number
            pattern (int): Pattern number (0 for no pattern comparison)
            day (int): Day offset for sliding window
        """
        signal1 = []
        signal2 = []
        predicted_data = []
        
        # Select prediction method
        if method == 1:
            predicted_data = self.predicted_data_window_based
            method_name = "window based"
        elif method == 2:
            predicted_data = self.predicted_data_day_based
            method_name = "day based"
        elif method == 3:
            predicted_data = self.predicted_data_historical_based
            method_name = "whole historical based"
        
        graph_title1 = f'True vs predicted ({method_name}).(w={window},p={pattern},d={day})'
        graph_title2 = f'Day lag vs normalized cross correlation.(w={window},p={pattern},d={day})'
        
        if pattern == 0:  # Actual vs predicted checking; no comparison with anomalous pattern
            signal1_start = (window - 1) * self.window_size
            signal1_end = signal1_start + self.window_size
            
            if len(predicted_data) < signal1_end:
                signal1_end = len(predicted_data)
            
            signal2_start = signal1_start
            signal2_end = signal1_end
            
            signal1 = predicted_data[signal1_start:signal1_end]
            signal2 = self.actual_data[signal2_start:signal2_end]
            
        elif pattern > 0:  # Comparison involves anomalous pattern
            signal1_start = (window - 1) * self.window_size + day - 1  # -1 for 0-based indexing
            signal1_end = signal1_start + int(self.pattern_desc_arr[pattern - 1])  # -1 for 0-based indexing
            
            if len(predicted_data) < signal1_end:
                signal1_end = len(predicted_data)
            
            signal1 = predicted_data[signal1_start:signal1_end]
            pattern_length = int(self.pattern_desc_arr[pattern - 1])
            signal2 = self.pattern_mat[pattern - 1, :pattern_length]  # -1 for 0-based indexing
        
        # Check if signals have same length
        if len(signal1) == len(signal2) and len(signal1) > 0:
            # Compute normalized cross-correlation
            # Using scipy's correlate with 'full' mode and manual normalization
            correlation = correlate(signal1, signal2, mode='full')
            
            # Normalize the correlation
            norm_factor = np.sqrt(np.sum(signal1**2) * np.sum(signal2**2))
            if norm_factor > 0:
                cor_sequence = correlation / norm_factor
            else:
                return
            
            # Create lag array
            lag = np.arange(-len(signal2) + 1, len(signal1))
            
            max_cor = np.max(cor_sequence)
            
            if max_cor > 0.80:
                self.result.append([method, window, pattern, day])
                
                # Plot signals comparison
                x1 = np.linspace(1, len(signal1), len(signal1))
                plt.figure(figsize=(10, 6))
                plt.plot(x1, signal2, label='actual', linewidth=2)
                plt.plot(x1, signal1, label='predicted', linewidth=2)
                plt.legend()
                plt.xlabel('day')
                plt.ylabel('volume')
                plt.title(graph_title1)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Plot cross-correlation
                plt.figure(figsize=(10, 6))
                plt.plot(lag, cor_sequence, linewidth=2)
                
                max_index = np.argmax(cor_sequence)
                max_lag = lag[max_index]
                
                xlabel2 = f'Maximum ncr = {max_cor:.4f} at day lag = {max_lag}'
                plt.xlabel(f'day lag\n{xlabel2}')
                plt.ylabel('normalized cross correlation')
                plt.title(graph_title2)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                plt.show()
    
    def run_analysis(self):
        """Main analysis function"""
        
        # Load data
        self.load_data()
        
        if self.actual_data is None:
            print("Cannot proceed without data. Please check file paths.")
            return
        
        # Calculate methods matrix (equivalent to MATLAB methods array)
        methods = [
            [len(self.predicted_data_window_based), 
             math.ceil(len(self.predicted_data_window_based) / self.window_size)],
            [len(self.predicted_data_day_based), 
             math.ceil(len(self.predicted_data_day_based) / self.window_size)],
            [len(self.predicted_data_historical_based), 
             math.ceil(len(self.predicted_data_historical_based) / self.window_size)]
        ]
        
        print("Starting analysis...")
        print("Methods configuration:")
        for i, method_info in enumerate(methods, 1):
            print(f"Method {i}: Length={method_info[0]}, Windows={method_info[1]}")
        
        # Main analysis loops
        for method in range(1, 4):  # methods 1, 2, 3
            print(f"\nProcessing method {method}...")
            num_window = methods[method - 1][1]  # -1 for 0-based indexing
            
            for window in range(1, num_window + 1):
                # Process window without pattern (pattern = 0)
                self.normalized_cross_correlation(method, window, 0, 0)
                
                # Process each pattern
                for pattern in range(1, self.num_pattern + 1):
                    self.overlap = 10
                    if window == num_window:
                        self.overlap = 0
                    
                    # Calculate range for sliding window
                    pattern_size = int(self.pattern_desc_arr[pattern - 1])  # -1 for 0-based indexing
                    max_day = self.window_size - (pattern_size - self.overlap)
                    
                    for day in range(1, max_day + 1):
                        self.normalized_cross_correlation(method, window, pattern, day)
        
        # Save results
        if self.result:
            result_array = np.array(self.result)
            np.savetxt('output/bp_output.csv', result_array, delimiter=',', fmt='%d')
            print(f"\nAnalysis complete. Found {len(self.result)} correlations > 0.80")
            print("Results saved to 'output/bp_output.csv'")
            
            # Print results
            print("\nResults (Method, Window, Pattern, Day):")
            print("=" * 40)
            for row in self.result:
                print(f"{row[0]}, {row[1]}, {row[2]}, {row[3]}")
        else:
            print("\nNo correlations > 0.80 found.")

def main():
    """Main function to run the analysis"""
    print("Time Series Similarity Analysis")
    print("=" * 50)
    
    analyzer = TimeSeriesSimilarity()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()