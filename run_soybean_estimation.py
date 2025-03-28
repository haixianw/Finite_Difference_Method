#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Soybean Meal Futures SDE Parameter Estimation Program
"""

from csv_data_estimation import run_parameter_estimation
import os
from datetime import datetime

if __name__ == "__main__":
    print("=== Soybean Meal Futures SDE Parameter Estimation ===")
    print("CSV file: soybean_meal_futures.csv")
    print("Using closing price data for parameter estimation\n")
    print("Estimating both drift (μ) and volatility parameters\n")
    
    # Create output directory if it doesn't exist
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"soybean_meal_futures_results_{timestamp}.pdf")
    
    # Initial parameters [mu, alpha, beta, theta_bar, sigma_S, sigma_L, rho1, rho2, rho3, lambda_, phi]
    # Run parameter estimation
    estimated_params, L_estimated, param_stats = run_parameter_estimation(
        csv_file_path="soybean_meal_future_window5.csv",
        price_column="收盘价",
        date_column="交易时间",
        initial_params=[0.02, 2, 0.5, 0.22, 0.15, 0.1, 0.2, 0.5, 0.3, 5.0, 0.5],
        save_results=True,
        output_file=output_file
    )
    #新的参数：专门用于window1的参数 [0.02, 2, 0.5, 0.2, 0.15, 0.1, 0.2, 0.5, 0.3, 5.0, 0.5]
    #新的参数：专门用于window2的参数 [0.02, 2, 0.4, 0.2, 0.12, 0.1, 0.2, 0.5, 0.3, 5.0, 0.5]
    #新的参数：专门用于window3的参数 [0.02, 2, 0.305, 0.12, 0.144, 0.03, 0.2, 0.5, 0.3, 5.0, 0.5]
    #新的参数：专门用于window4的参数 [0.02, 2, 0.455, 0.12, 0.185, 0.05, 0.2, 0.5, 0.3, 5.0, 0.5]
    #新的参数：专门用于window5的参数 [0.02, 2, 0.5, 0.22, 0.15, 0.1, 0.2, 0.5, 0.3, 5.0, 0.5]
   
    print("\n=== Parameter Estimation Complete ===")
    print(f"Results saved to: {output_file}")
    print("Please check the PDF file for detailed results") 