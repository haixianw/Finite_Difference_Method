import os
import sys
import numpy as np
from datetime import datetime
from gbm_data_estimation import run_gbm_estimation

def main():
    """
    Main program to run Geometric Brownian Motion parameter estimation
    """
    print("=== Geometric Brownian Motion (GBM) Parameter Estimation ===")
    print("Model: dS_t = μ·S_t·dt + σ·S_t·dW_t")
    print("Estimating both drift (μ) and volatility (σ) parameters")
    
    # Set risk-free rate as initial guess for mu (optional)
    risk_free_rate = -0.002  # 2% annualized risk-free rate as initial guess
    print(f"Using risk-free rate r = {risk_free_rate:.2%} as initial guess for drift parameter μ")
    
    # Set output directory
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate timestamped output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"soybean_meal_futures_gbm_results_{timestamp}.pdf")
    
    # Set initial parameter values
    initial_params = [risk_free_rate, 0.1]  # [mu, sigma]
    
    # Run parameter estimation
    estimated_params, log_likelihood, param_stats = run_gbm_estimation(
        csv_file_path="25_soybean_meal_future_window1.csv",  # Use same CSV file as original code
        price_column="收盘价",  # Closing price column
        date_column="交易时间",  # Trading date column
        risk_free_rate=risk_free_rate,  # Used as initial guess only
        initial_params=initial_params,
        save_results=True,
        output_file=output_file
    )
    
    print("\n=== Parameter Estimation Complete ===")
    if estimated_params is not None:
        estimated_mu, estimated_sigma = estimated_params
        print(f"Estimated drift parameter μ = {estimated_mu:.4f}")
        print(f"Estimated volatility parameter σ = {estimated_sigma:.4f}")
        print(f"Maximum log-likelihood value = {log_likelihood:.4f}")
    print(f"Results saved to: {output_file}")
    print("Please check the PDF file for detailed results")

if __name__ == "__main__":
    main() 