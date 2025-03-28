import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from scipy.stats import norm
from gbm_parameter_estimation import GBMParameterEstimator

def load_futures_data(csv_file_path, price_column='收盘价', date_column='交易时间'):
    """
    Load futures data from CSV file
    
    Parameters:
    - csv_file_path: Path to CSV file
    - price_column: Price column name
    - date_column: Date column name
    
    Returns:
    - prices: Price array
    - dates: Date array
    - dt: Time step
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_file_path)
        print(f"Successfully read CSV file with {len(df)} rows")
        print(f"CSV file columns: {df.columns.tolist()}")
        
        # Check if necessary columns exist
        if price_column not in df.columns:
            print(f"Error: Price column '{price_column}' not found in CSV")
            return None, None, None
            
        # Extract price data
        prices = df[price_column].values
        
        # Convert price data to float
        try:
            # Handle thousands separators (if prices are strings)
            if prices.dtype == np.object_:
                processed_prices = []
                for p in prices:
                    if isinstance(p, str) and ',' in p:
                        processed_prices.append(float(p.replace(',', '')))
                    else:
                        processed_prices.append(float(p))
                prices = np.array(processed_prices)
                print("Processed string price data")
            else:
                # If already numeric type, convert directly to float
                prices = prices.astype(float)
        except Exception as e:
            print(f"Warning when processing price data: {e}")
            # Try to convert directly to float
            prices = df[price_column].astype(float).values
            
        # Check for missing values
        missing_count = np.isnan(prices).sum()
        if missing_count > 0:
            print(f"Warning: {missing_count} missing values in price data, will use forward fill")
            prices = pd.Series(prices).fillna(method='ffill').values
            
        # Extract date data (if exists)
        dates = None
        if date_column in df.columns:
            dates = df[date_column].values
            
        # Calculate time step
        if dates is not None:
            try:
                # Convert to datetime format
                date_series = pd.to_datetime(dates)
                # Calculate average date interval (in days)
                date_diffs = np.diff(date_series)
                avg_days = np.mean([d.days + d.seconds/86400 for d in date_diffs])
                
                # Convert date interval to years
                if avg_days < 2:  # If daily data, use trading day count
                    dt = 1/252  # Assume 252 trading days per year
                else:
                    dt = avg_days / 365  # Convert to years
            except Exception as e:
                print(f"Warning: Could not calculate time step from dates: {e}")
                dt = 1/252  # Default to trading day time step
        else:
            dt = 1/252  # Default to trading day time step
            
        print(f"Data loading complete: {len(prices)} price points, time step dt={dt:.6f} years")
        print(f"Price range: Min={min(prices):.2f}, Max={max(prices):.2f}, Mean={np.mean(prices):.2f}")
        
        if dates is not None:
            try:
                date_range = f"{pd.to_datetime(dates[0])} to {pd.to_datetime(dates[-1])}"
            except:
                date_range = "Unknown date range"
            print(f"Date range: {date_range}")
            
        return prices, dates, dt
        
    except Exception as e:
        print(f"Error loading data from {csv_file_path}: {e}")
        return None, None, None

def analyze_returns(prices, log_returns=True, return_fig=False):
    """
    Analyze return distribution
    
    Parameters:
    - prices: Price series
    - log_returns: Whether to use log returns
    - return_fig: Whether to return figure object
    
    Returns:
    - fig: Figure object if return_fig=True
    """
    # Calculate returns
    if log_returns:
        returns = np.diff(np.log(prices))
        title = "Log Returns Distribution"
    else:
        returns = np.diff(prices) / prices[:-1]
        title = "Simple Returns Distribution"
        
    # Basic statistics
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    min_return = np.min(returns)
    max_return = np.max(returns)
    skew = np.mean(((returns - mean_return) / std_return) ** 3)
    kurt = np.mean(((returns - mean_return) / std_return) ** 4) - 3
    
    print(f"\n{title} Analysis:")
    print(f"Mean: {mean_return:.6f}")
    print(f"Standard Deviation: {std_return:.6f}")
    print(f"Minimum: {min_return:.6f}")
    print(f"Maximum: {max_return:.6f}")
    print(f"Skewness: {skew:.6f}")
    print(f"Kurtosis: {kurt:.6f}")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Returns time series
    ax1.plot(returns, 'b-')
    ax1.set_title(f"{title} Time Series")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Return")
    ax1.grid(True)
    
    # Returns distribution histogram
    ax2.hist(returns, bins=50, density=True, alpha=0.7)
    
    # Add normal distribution fit curve
    x = np.linspace(min_return, max_return, 100)
    y = norm.pdf(x, mean_return, std_return)
    ax2.plot(x, y, 'r-', linewidth=2)
    
    ax2.set_title(f"{title} Distribution with Normal Fit")
    ax2.set_xlabel("Return")
    ax2.set_ylabel("Probability Density")
    ax2.grid(True)
    
    # Add statistics text box
    stats_text = f"Mean: {mean_return:.6f}\nStd Dev: {std_return:.6f}\nSkewness: {skew:.6f}\nKurtosis: {kurt:.6f}"
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if return_fig:
        return fig
    else:
        plt.show()
        return None

def calculate_parameter_significance(estimated_params, log_likelihood, n_observations):
    """
    Calculate parameter significance and confidence intervals
    
    Parameters:
    - estimated_params: Estimated parameters [mu, sigma]
    - log_likelihood: Maximum log-likelihood value
    - n_observations: Number of observations
    
    Returns:
    - param_stats: DataFrame with parameter estimates and statistics
    """
    # Unpack parameters
    estimated_mu, estimated_sigma = estimated_params
    
    # For GBM, the Fisher information matrix has entries:
    # I_mu,mu = n/σ² * dt 
    # I_sigma,sigma = 2n/σ²
    # I_mu,sigma = I_sigma,mu = 0 (approximation, parameters are asymptotically orthogonal)
    
    # Standard errors
    se_mu = estimated_sigma / np.sqrt(n_observations * 1/252)  # Using approx dt=1/252 for daily data
    se_sigma = estimated_sigma / np.sqrt(2 * n_observations)
    
    # Calculate t-statistics
    t_stat_mu = estimated_mu / se_mu
    t_stat_sigma = estimated_sigma / se_sigma
    
    # Calculate 95% confidence intervals
    z_95 = norm.ppf(0.975)  # 97.5% quantile, corresponding to 95% confidence interval
    mu_lower = estimated_mu - z_95 * se_mu
    mu_upper = estimated_mu + z_95 * se_mu
    sigma_lower = estimated_sigma - z_95 * se_sigma
    sigma_upper = estimated_sigma + z_95 * se_sigma
    
    # Calculate AIC and BIC
    # For 2-parameter model
    aic = -2 * log_likelihood + 2 * 2  # 2 parameters: mu and sigma
    bic = -2 * log_likelihood + np.log(n_observations) * 2
    
    # Create results DataFrame
    result = {
        'Parameter': ['mu', 'sigma', 'log-likelihood', 'AIC', 'BIC'],
        'Estimate': [
            f"{estimated_mu:.4f}\n(t={t_stat_mu:.2f})",
            f"{estimated_sigma:.4f}\n(t={t_stat_sigma:.2f})",
            f"{log_likelihood:.4f}",
            f"{aic:.4f}",
            f"{bic:.4f}"
        ]
    }
    
    param_stats = pd.DataFrame(result)
    
    # Print confidence intervals
    print(f"\n95% Confidence Interval for μ: [{mu_lower:.4f}, {mu_upper:.4f}]")
    print(f"95% Confidence Interval for σ: [{sigma_lower:.4f}, {sigma_upper:.4f}]")
    
    return param_stats

def save_results_to_file(prices, dates, estimated_params, log_likelihood, param_stats, output_file, dt):
    """
    Save estimation results to PDF and LaTeX files
    
    Parameters:
    - prices: Observed price series
    - dates: Date series
    - estimated_params: Estimated parameters [mu, sigma]
    - log_likelihood: Maximum log-likelihood value
    - param_stats: Parameter statistics DataFrame
    - output_file: Output file path
    - dt: Time step
    """
    import os
    from matplotlib.backends.backend_pdf import PdfPages
    
    # Unpack parameters
    estimated_mu, estimated_sigma = estimated_params
    
    # Generate PDF and LaTeX outputs
    pdf_file = output_file if output_file.endswith('.pdf') else f"{output_file}.pdf"
    latex_file = os.path.splitext(output_file)[0] + '.tex'
    
    # Create GBM estimator for simulation
    estimator = GBMParameterEstimator(prices, dt)
    
    # Create PDF output
    with PdfPages(pdf_file) as pdf:
        # First page: Parameter estimation results
        fig = plt.figure(figsize=(8.5, 11))
        ax = plt.axes([0, 0, 1, 1], frame_on=False)
        ax.set_axis_off()
        
        # Add title
        plt.figtext(0.5, 0.95, "Geometric Brownian Motion Parameter Estimation Results", 
                   fontsize=16, ha='center', weight='bold')
        
        # Add data summary
        summary_text = (
            "Data Summary:\n"
            f"- Number of observations: {len(prices)}\n"
            f"- Time period: {min(pd.to_datetime(dates)) if dates is not None else 'N/A'} to "
            f"{max(pd.to_datetime(dates)) if dates is not None else 'N/A'}\n"
            f"- Time step (dt): {dt:.6f} years\n\n"
            "Model Description:\n"
            "- Geometric Brownian Motion (GBM): dS_t = μ·S_t·dt + σ·S_t·dW_t\n"
            "- Both drift (μ) and volatility (σ) parameters are estimated\n"
            "- Log-normal distributed stock prices\n"
            "- Maximum Likelihood Estimation method\n"
        )
        
        plt.figtext(0.1, 0.85, summary_text, fontsize=10, va="top")
        
        # Parameter table title
        plt.figtext(0.1, 0.65, "Parameter Estimation Results:", fontsize=12, weight='bold')
        
        # Create parameter data
        param_names = param_stats['Parameter'].tolist()
        param_values = param_stats['Estimate'].tolist()
        
        # Create borderless table
        row_height = 0.03
        rows = len(param_names)
        
        # Create header
        plt.figtext(0.2, 0.62, "Parameter", fontsize=10, weight='bold')
        plt.figtext(0.6, 0.62, "Estimate", fontsize=10, weight='bold')
        
        # Draw horizontal line under header
        plt.axhline(y=0.61, xmin=0.15, xmax=0.85, color='black', alpha=0.3, linewidth=1)
        
        # Create parameter rows
        for i, (name, value) in enumerate(zip(param_names, param_values)):
            y_pos = 0.6 - (i * row_height)
            plt.figtext(0.2, y_pos, name, fontsize=10)
            plt.figtext(0.6, y_pos, value, fontsize=10)
        
        # Draw horizontal line at bottom of table
        plt.axhline(y=0.6 - (rows * row_height), xmin=0.15, xmax=0.85, color='black', alpha=0.3, linewidth=1)
        
        # Add confidence interval information
        se_mu = estimated_sigma / np.sqrt((len(prices) - 1) * dt)
        se_sigma = estimated_sigma / np.sqrt(2 * (len(prices) - 1))
        z_95 = norm.ppf(0.975)
        mu_lower = estimated_mu - z_95 * se_mu
        mu_upper = estimated_mu + z_95 * se_mu
        sigma_lower = estimated_sigma - z_95 * se_sigma
        sigma_upper = estimated_sigma + z_95 * se_sigma
        
        interval_text = (
            f"μ 95% Confidence Interval: [{mu_lower:.4f}, {mu_upper:.4f}]\n"
            f"σ 95% Confidence Interval: [{sigma_lower:.4f}, {sigma_upper:.4f}]\n"
            "Note: t-statistics indicate parameter significance relative to zero"
        )
        plt.figtext(0.1, 0.45, interval_text, fontsize=10)
        
        pdf.savefig()
        plt.close()
        
        # Second page: Prices and simulated paths
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot observed prices
        ax.plot(prices, 'b-', label='Observed Prices')
        
        # Plot several simulated paths
        for i in range(5):
            simulated_prices = estimator.simulate_gbm(estimated_mu, estimated_sigma)
            ax.plot(simulated_prices, 'r-', alpha=0.3, label='Simulated Path' if i == 0 else "")
        
        # If dates exist, set x-axis labels
        if dates is not None:
            try:
                time_indices = np.arange(len(prices))
                clean_dates = pd.to_datetime(pd.Series(dates)).dropna()
                
                # Set evenly distributed ticks
                num_ticks = min(10, len(clean_dates))
                tick_indices = np.linspace(0, len(prices)-1, num_ticks, dtype=int)
                ax.set_xticks(tick_indices)
                
                # Format date labels
                date_labels = [clean_dates[i] if i < len(clean_dates) else pd.NaT for i in tick_indices]
                date_labels = [d.strftime('%Y-%m-%d') if not pd.isna(d) else '' for d in date_labels]
                ax.set_xticklabels(date_labels, rotation=45)
            except Exception as e:
                print(f"Warning when plotting dates: {e}")
        
        ax.set_title("Observed Prices and GBM Simulated Paths")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)
        
        # Add parameter text box
        param_text = f"μ = {estimated_mu:.4f}\nσ = {estimated_sigma:.4f}"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, param_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        pdf.savefig()
        plt.close()
        
        # Third page: Returns analysis
        analyze_fig = analyze_returns(prices, log_returns=True, return_fig=True)
        pdf.savefig(analyze_fig)
        plt.close(analyze_fig)
    
    # Create LaTeX file
    with open(latex_file, 'w') as f:
        # LaTeX preamble
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage{booktabs}\n")
        f.write("\\usepackage{graphicx}\n")
        f.write("\\usepackage{float}\n")
        f.write("\\usepackage[margin=1in]{geometry}\n")
        f.write("\\begin{document}\n\n")
        
        # Title
        f.write("\\begin{center}\n")
        f.write("\\Large\\textbf{Geometric Brownian Motion Parameter Estimation Results}\n")
        f.write("\\end{center}\n\n")
        
        # Summary section
        f.write("\\section*{Data Summary}\n")
        f.write("\\begin{itemize}\n")
        f.write(f"\\item Number of observations: {len(prices)}\n")
        f.write(f"\\item Time period: {min(pd.to_datetime(dates)) if dates is not None else 'N/A'} to ")
        f.write(f"{max(pd.to_datetime(dates)) if dates is not None else 'N/A'}\n")
        f.write(f"\\item Time step (dt): {dt:.6f} years\n")
        f.write("\\end{itemize}\n\n")
        
        f.write("\\section*{Model Description}\n")
        f.write("\\begin{itemize}\n")
        f.write("\\item Geometric Brownian Motion (GBM): $dS_t = \\mu S_t dt + \\sigma S_t dW_t$\n")
        f.write("\\item Both drift ($\\mu$) and volatility ($\\sigma$) parameters are estimated\n")
        f.write("\\item Log-normal distributed stock prices\n")
        f.write("\\item Maximum Likelihood Estimation method\n")
        f.write("\\end{itemize}\n\n")
        
        # Parameter table
        f.write("\\section*{Parameter Estimation Results}\n")
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lc}\n")
        f.write("\\toprule\n")
        f.write("Parameter & Estimate \\\\\n")
        f.write("\\midrule\n")
        
        # Add parameter rows
        for i, (name, value) in enumerate(zip(param_names, param_values)):
            # Replace newlines with LaTeX newlines
            value_latex = value.replace('\n', ' $\\quad$ ')
            f.write(f"{name} & {value_latex} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Geometric Brownian Motion Parameter Estimation Results}\n")
        f.write("\\end{table}\n\n")
        
        # Add confidence interval information
        f.write("\\begin{small}\n")
        f.write(f"$\\mu$ 95\\% Confidence Interval: [{mu_lower:.4f}, {mu_upper:.4f}]\\\\\n")
        f.write(f"$\\sigma$ 95\\% Confidence Interval: [{sigma_lower:.4f}, {sigma_upper:.4f}]\\\\\n")
        f.write("Note: t-statistics indicate parameter significance relative to zero\n")
        f.write("\\end{small}\n\n")
        
        # End LaTeX document
        f.write("\\end{document}\n")
    
    print(f"Results saved to {pdf_file} and {latex_file}")

def run_gbm_estimation(csv_file_path, price_column='收盘价', date_column='交易时间', 
                       risk_free_rate=None, initial_params=None, 
                       plot_result=True, analyze_data=True,
                       save_results=True, output_file=None):
    """
    Main function to run Geometric Brownian Motion parameter estimation
    
    Parameters:
    - csv_file_path: Path to CSV file
    - price_column: Price column name
    - date_column: Date column name
    - risk_free_rate: Risk-free rate, used as initial guess for μ parameter if provided
    - initial_params: Initial parameters [mu, sigma] estimate
    - plot_result: Whether to plot results
    - analyze_data: Whether to analyze data distribution
    - save_results: Whether to save results to file
    - output_file: Output file path
    
    Returns:
    - estimated_params: Estimated parameters [mu, sigma]
    - log_likelihood: Maximum log-likelihood value
    - param_stats: Parameter statistics
    """
    # Load data
    prices, dates, dt = load_futures_data(csv_file_path, price_column, date_column)
    if prices is None:
        print("Data loading failed, cannot continue")
        return None, None, None
    
    # Visualize raw data
    if plot_result:
        plt.figure(figsize=(10, 6))
        plt.plot(prices, 'b-')
        plt.title("Raw Price Data")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.grid(True)
        plt.show()
    
    # Analyze returns
    if analyze_data:
        analyze_returns(prices)
    
    # Create GBM estimator
    estimator = GBMParameterEstimator(prices, dt, risk_free_rate)
    
    # Estimate parameters
    print("\nStarting parameter estimation...")
    estimated_params, log_likelihood = estimator.estimate_parameters(initial_params)
    
    # Calculate parameter significance
    print("\nCalculating parameter significance...")
    param_stats = calculate_parameter_significance(estimated_params, log_likelihood, len(prices) - 1)
    
    # Print results
    print("\nParameter Estimation Results:")
    print(param_stats.to_string())
    
    # Visualize results
    if plot_result:
        estimator.plot_results(estimated_params, title="Geometric Brownian Motion Parameter Estimation Results")
    
    # Save results to file
    if save_results:
        if output_file is None:
            # Generate default output filename
            base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "results"
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            output_file = os.path.join(output_dir, f"{base_name}_gbm_results_{timestamp}.pdf")
        
        # Save results
        save_results_to_file(
            prices, dates, estimated_params, log_likelihood, param_stats, output_file, dt
        )
    
    return estimated_params, log_likelihood, param_stats 