import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm
import os
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import io
from sde_parameter_estimation import SDEParameterEstimator
from copy import deepcopy
# Set matplotlib to use a locale that supports UTF-8
mpl.rcParams['font.family'] = 'DejaVu Sans'

def load_futures_data(csv_file_path, price_column='收盘价', date_column='交易时间'):
    """
    Load futures price data from CSV file
    
    Parameters:
    - csv_file_path: Path to CSV file
    - price_column: Price column name (default: '收盘价')
    - date_column: Date column name (default: '交易时间')
    
    Returns:
    - prices: Price array (numpy array)
    - dates: Date array
    - dt: Estimated time step (annualized)
    """
    # Read CSV file
    try:
        data = pd.read_csv(csv_file_path)
        print(f"Successfully read CSV file with {len(data)} rows")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None, None, None
    
    # Print all column names for debugging
    print(f"CSV file columns: {data.columns.tolist()}")
    
    # Check if required columns exist
    required_columns = [price_column]
    if date_column:
        required_columns.append(date_column)
    
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"Missing required columns in CSV file: {missing_columns}")
        print(f"Available columns: {data.columns.tolist()}")
        return None, None, None
    
    # Handle thousands separator in price data
    if data[price_column].dtype == object:
        try:
            # Remove thousands separator and convert to float
            data[price_column] = data[price_column].str.replace(',', '').astype(float)
            print("Processed thousands separators in price data")
        except Exception as e:
            print(f"Error processing price data: {e}")
    
    # If date column exists, convert to datetime format and sort by date
    if date_column in data.columns:
        try:
            data[date_column] = pd.to_datetime(data[date_column])
            data = data.sort_values(by=date_column)
            
            # Calculate time step (daily average)
            time_diffs = data[date_column].diff()[1:].dt.days
            avg_days = time_diffs.mean()
            dt = 1/252 if avg_days < 2 else avg_days/365  # Assume daily data if average diff < 2
            
            dates = data[date_column].values
        except Exception as e:
            print(f"Error processing date column: {e}")
            dt = 1/252  # Default to daily data
            dates = np.arange(len(data))
    else:
        dt = 1/252  # Default to daily data
        dates = np.arange(len(data))
    
    # Extract price data
    prices = data[price_column].values
    
    # Check and handle missing values
    if np.isnan(prices).any():
        print(f"Warning: {np.isnan(prices).sum()} missing values in price data, will use forward fill")
        prices = pd.Series(prices).fillna(method='ffill').values
    
    print(f"Data loading complete: {len(prices)} price points, time step dt={dt:.6f} years")
    
    # Display data range
    print(f"Price range: Min={np.min(prices):.2f}, Max={np.max(prices):.2f}, Mean={np.mean(prices):.2f}")
    print(f"Date range: {pd.to_datetime(dates[0]).date()} to {pd.to_datetime(dates[-1]).date()}")
    
    return prices, dates, dt

def visualize_raw_data(prices, dates=None, title="Raw Price Data", return_fig=False):
    """
    Visualize raw price data
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if dates is not None and len(dates) == len(prices):
        ax.plot(dates, prices, 'b-')
        # Set appropriate date format
        fig.autofmt_xdate()
    else:
        ax.plot(prices, 'b-')
    
    ax.set_title(title)
    ax.set_ylabel("Price")
    ax.grid(True)
    
    if return_fig:
        return fig
    else:
        plt.tight_layout()
        plt.show()

def analyze_returns(prices, log_returns=True, return_fig=False):
    """
    Analyze price returns distribution
    """
    if log_returns:
        # Calculate log returns
        returns = np.diff(np.log(prices))
        return_type = "Log Returns"
    else:
        # Calculate simple returns
        returns = np.diff(prices) / prices[:-1]
        return_type = "Simple Returns"
    
    # Plot returns distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Time series
    ax1.plot(returns, 'b-')
    ax1.set_title(f"{return_type} Time Series")
    ax1.set_xlabel("Time")
    ax1.set_ylabel(return_type)
    ax1.grid(True)
    
    # Histogram
    ax2.hist(returns, bins=50, density=True, alpha=0.7)
    ax2.set_title(f"{return_type} Distribution")
    ax2.set_xlabel(return_type)
    ax2.set_ylabel("Frequency Density")
    ax2.grid(True)
    
    # Calculate statistics
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    skew = np.mean(((returns - mean_return) / std_return) ** 3)
    kurtosis = np.mean(((returns - mean_return) / std_return) ** 4) - 3
    
    # Display statistics
    stats_text = f"Mean: {mean_return:.6f}\nStd Dev: {std_return:.6f}\nSkewness: {skew:.4f}\nKurtosis: {kurtosis:.4f}"
    ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes, verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    if return_fig:
        return fig, returns
    else:
        plt.show()
        return returns

def calculate_parameter_significance(estimated_params, prices, L_estimated, dt):
    """
    计算参数显著性统计量（使用正则化方法处理黑塞矩阵问题）
    
    参数:
    - estimated_params: 估计的参数 (mu, alpha, beta, theta_bar, sigma_S, sigma_L, rho1, rho2, rho3, lambda_, phi)
    - prices: 观测价格序列
    - L_estimated: 估计的流动性序列
    - dt: 时间步长
    
    返回:
    - param_stats: 包含参数估计和t统计量的DataFrame
    - log_likelihood: 对数似然值
    """
    # 创建一个估计器以计算对数似然
    estimator = SDEParameterEstimator(prices, dt)
    
    def nll_func(params):
        """计算负对数似然函数值"""
        try:
            _, log_lik = estimator.extended_kalman_filter(params, prices)
            return -log_lik
        except Exception as e:
            print(f"Error in likelihood calculation: {e}")
            return 1e10
    
    # 计算Hessian矩阵（对负对数似然函数的二阶导数）
    eps = 1e-4  # 增加有限差分步长以减少数值误差
    n_params = len(estimated_params)
    hessian = np.zeros((n_params, n_params))
    
    # 计算Hessian矩阵
    for i in range(n_params):
        for j in range(i, n_params):  # 只计算上三角，之后复制到下三角
            # 如果i==j，使用中心差分计算二阶导数
            if i == j:
                h = max(abs(estimated_params[i]) * eps, eps)
                params_plus = estimated_params.copy()
                params_center = estimated_params.copy()
                params_minus = estimated_params.copy()
                
                params_plus[i] += h
                params_minus[i] -= h
                
                f_plus = nll_func(params_plus)
                f_center = nll_func(params_center)
                f_minus = nll_func(params_minus)
                
                # 中心二阶差分公式
                hessian[i, j] = (f_plus - 2*f_center + f_minus) / (h * h)
            else:
                # 对混合偏导数使用交叉差分
                h_i = max(abs(estimated_params[i]) * eps, eps)
                h_j = max(abs(estimated_params[j]) * eps, eps)
                
                params_pp = estimated_params.copy()
                params_pm = estimated_params.copy()
                params_mp = estimated_params.copy()
                params_mm = estimated_params.copy()
                
                params_pp[i] += h_i
                params_pp[j] += h_j
                params_pm[i] += h_i
                params_pm[j] -= h_j
                params_mp[i] -= h_i
                params_mp[j] += h_j
                params_mm[i] -= h_i
                params_mm[j] -= h_j
                
                f_pp = nll_func(params_pp)
                f_pm = nll_func(params_pm)
                f_mp = nll_func(params_mp)
                f_mm = nll_func(params_mm)
                
                # 混合偏导数公式
                hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * h_i * h_j)
                hessian[j, i] = hessian[i, j]  # 保证对称性
    
    # 打印原始Hessian矩阵的条件数，用于诊断
    try:
        orig_cond = np.linalg.cond(hessian)
        print(f"Original Hessian matrix condition number: {orig_cond:.2e}")
    except:
        print("Could not compute original condition number (likely singular matrix)")
    
    # 正则化Hessian矩阵
    # 添加小的对角元素以确保正定性和数值稳定性
    regularization_factor = 1e-2
    diag_mean = np.mean(np.abs(np.diag(hessian)))
    reg_value = max(regularization_factor * diag_mean, 1e-5)
    
    # 应用Tikhonov正则化 (Ridge-like regularization)
    for i in range(n_params):
        hessian[i, i] += reg_value
    
    # 重新检查条件数
    try:
        reg_cond = np.linalg.cond(hessian)
        print(f"Regularized Hessian matrix condition number: {reg_cond:.2e}")
    except:
        print("Still could not compute condition number after regularization")
    
    # 计算协方差矩阵（Hessian矩阵的逆）
    try:
        # 先尝试使用Cholesky分解求逆，更稳定
        L = np.linalg.cholesky(hessian)
        y = np.linalg.inv(L)
        cov_matrix = y.T @ y
    except np.linalg.LinAlgError:
        print("警告: Cholesky分解失败，尝试直接求逆")
        try:
            # 如果Cholesky分解失败，尝试直接求逆
            cov_matrix = np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            print("警告: 直接求逆也失败，使用伪逆")
            # 如果直接求逆失败，使用伪逆
            cov_matrix = np.linalg.pinv(hessian)
    
    # 从协方差矩阵提取标准误差
    std_errors = np.sqrt(np.diag(cov_matrix))
    
    # 设置最小标准误差的阈值，防止除以接近零的值
    min_std_error = 1e-5 * np.max(np.abs(estimated_params))
    std_errors = np.maximum(std_errors, min_std_error)
    
    # 计算t统计量
    t_stats = estimated_params / std_errors
    
    # 对于极端值进行处理，避免无限大的t统计量
    t_stats = np.clip(t_stats, -100, 100)
    
    # 格式化参数估计和t统计量
    param_names = ['mu', 'alpha', 'beta', 'theta_bar', 
                  'sigma_S', 'sigma_L', 'rho1', 'rho2', 'rho3', 'lambda_', 'phi']
    
    results = []
    for i, (name, estimate, t_stat) in enumerate(zip(param_names, estimated_params, t_stats)):
        if abs(t_stat) > 99:
            formatted = f"{estimate:.4f}\n(>99)" if t_stat > 0 else f"{estimate:.4f}\n(<-99)"
        else:
            formatted = f"{estimate:.4f}\n({t_stat:.2f})"
        
        results.append({'Parameter': name, 'Estimate': formatted})
    
    # 添加对数似然结果
    _, log_likelihood = estimator.extended_kalman_filter(estimated_params, prices)
    results.append({'Parameter': 'Log-Likelihood', 'Estimate': f"{log_likelihood:.3f}"})
    
    # 返回结果
    param_stats_df = pd.DataFrame(results)
    
    # 打印原始标准误差和t统计量的详细信息（对于调试）
    print("\n详细参数估计结果:")
    for i, (name, est, se, t) in enumerate(zip(param_names, estimated_params, std_errors, t_stats)):
        print(f"{name}: 估计值={est:.6f}, 标准误差={se:.6f}, t统计量={t:.2f}")
    
    return param_stats_df, log_likelihood

def plot_estimated_vs_true(L_estimated, dates=None, title="Estimated Liquidity Process", return_fig=False):
    """
    绘制估计的流动性过程，不显示参数估计值
    """
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    
    if dates is not None:
        ax.plot(dates, L_estimated, 'b-', label='Estimated Liquidity')
    else:
        ax.plot(range(len(L_estimated)), L_estimated, 'b-', label='Estimated Liquidity')
    
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Liquidity')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    
    if return_fig:
        return fig
    plt.show()

def save_results_to_file(prices, dates, estimated_params, L_estimated, param_stats, output_file, dt):
    """
    Save estimation results, including parameters and plots to PDF and LaTeX files.
    
    Parameters:
        prices (array): Observed price series
        dates (array): Dates corresponding to the price observations
        estimated_params (array): Estimated SDE parameters
        L_estimated (array): Estimated liquidity process
        param_stats (DataFrame): Parameter statistics including t-statistics
        output_file (str): Output file path (without extension)
        dt (float): Time step size
    """
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import pandas as pd
    import numpy as np
    from datetime import datetime
    import os
    
    # Generate both PDF and LaTeX outputs
    pdf_file = output_file if output_file.endswith('.pdf') else f"{output_file}.pdf"
    latex_file = os.path.splitext(output_file)[0] + '.tex'
    
    # First, create the PDF output
    with PdfPages(pdf_file) as pdf:
        # First page: Summary and Parameter Estimates
        fig = plt.figure(figsize=(8.5, 11))
        # Remove the main axes to avoid the large frame
        ax = plt.axes([0, 0, 1, 1], frame_on=False)
        ax.set_axis_off()
        
        # Add title
        plt.figtext(0.5, 0.95, "Soybean Meal Futures SDE Parameter Estimation Results", 
                   fontsize=16, ha='center', weight='bold')
        
        # Add summary information
        summary_text = (
            "Data Summary:\n"
            f"- Number of observations: {len(prices)}\n"
            f"- Time period: {min(pd.to_datetime(dates))} to {max(pd.to_datetime(dates))}\n"
            f"- Time step (dt): {dt:.6f} years\n\n"
            "Model Specification:\n"
            "- Stochastic Differential Equation (SDE) with liquidity feedback\n"
            "- Extended Kalman Filter (EKF) for latent state estimation\n"
            "- Maximum Likelihood Estimation (MLE) for parameter estimation\n"
        )
        
        plt.figtext(0.1, 0.85, summary_text, fontsize=10, va="top")
        
        # Parameter table section title
        plt.figtext(0.1, 0.65, "Parameter Estimation Results:", fontsize=12, weight='bold')
        
        # Create parameter table data
        param_names = param_stats['Parameter'].tolist()
        param_values = param_stats['Estimate'].tolist()
        
        # Create a cleaner table without borders for a more professional look
        # We'll use a grid layout instead of matplotlib's table
        row_height = 0.025
        rows = len(param_names)
        table_height = rows * row_height
        
        # Create header row
        plt.figtext(0.2, 0.6, "Parameter", fontsize=10, weight='bold')
        plt.figtext(0.6, 0.6, "Soybean Meal Futures", fontsize=10, weight='bold')
        
        # Draw a horizontal line under the header
        plt.axhline(y=0.59, xmin=0.15, xmax=0.85, color='black', alpha=0.3, linewidth=1)
        
        # Create parameter rows
        for i, (name, value) in enumerate(zip(param_names, param_values)):
            y_pos = 0.58 - (i * row_height)
            plt.figtext(0.2, y_pos, name, fontsize=10)
            plt.figtext(0.6, y_pos, value, fontsize=10)
        
        # Draw a horizontal line at the bottom of the table
        plt.axhline(y=0.58 - (rows * row_height), xmin=0.15, xmax=0.85, color='black', alpha=0.3, linewidth=1)
        
        # Add note about significance
        note_text = (
            "Note: Values in parentheses are t-statistics.\n"
            "Statistical significance: |t| > 2.58 (1%), |t| > 1.96 (5%), |t| > 1.65 (10%)"
        )
        plt.figtext(0.1, 0.15, note_text, fontsize=8)
        
        # Add log-likelihood information
        if 'Log-Likelihood' in param_stats.index:
            log_lik = param_stats.loc['Log-Likelihood', 'Estimate']
            plt.figtext(0.1, 0.1, f"Log-Likelihood: {log_lik}", fontsize=10)
        
        pdf.savefig()
        plt.close()
        
        # Second page: Price plot
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Create a numerical time vector for plotting
        if dates is not None:
            # Convert dates to numbers for plotting
            try:
                time_indices = np.arange(len(prices))
                clean_dates = pd.to_datetime(pd.Series(dates)).dropna()
                
                # Plot observed prices
                ax1.plot(time_indices, prices, 'b-')
                
                # Format x-axis with dates at regular intervals
                num_ticks = min(10, len(clean_dates))
                tick_indices = np.linspace(0, len(prices)-1, num_ticks, dtype=int)
                ax1.set_xticks(tick_indices)
                
                # Format date labels
                date_labels = [clean_dates[i] if i < len(clean_dates) else pd.NaT for i in tick_indices]
                date_labels = [d.strftime('%Y-%m-%d') if not pd.isna(d) else '' for d in date_labels]
                ax1.set_xticklabels(date_labels, rotation=45)
            except Exception as e:
                print(f"Warning when plotting dates: {e}")
                ax1.plot(prices, 'b-')
        else:
            ax1.plot(prices, 'b-')
            
        ax1.set_title("Soybean Meal Futures Prices")
        ax1.set_ylabel("Price")
        ax1.grid(True)
        
        pdf.savefig()
        plt.close()
        
        # Third page: Estimated liquidity
        fig, ax2 = plt.subplots(figsize=(10, 6))
        
        # Plot estimated liquidity
        if dates is not None:
            try:
                time_indices = np.arange(len(L_estimated))
                ax2.plot(time_indices, L_estimated, 'r-')
                
                # Format x-axis with dates at regular intervals
                num_ticks = min(10, len(clean_dates))
                tick_indices = np.linspace(0, len(L_estimated)-1, num_ticks, dtype=int)
                ax2.set_xticks(tick_indices)
                
                # Format date labels
                date_labels = [clean_dates[i] if i < len(clean_dates) else pd.NaT for i in tick_indices]
                date_labels = [d.strftime('%Y-%m-%d') if not pd.isna(d) else '' for d in date_labels]
                ax2.set_xticklabels(date_labels, rotation=45)
            except Exception as e:
                print(f"Warning when plotting dates for liquidity: {e}")
                ax2.plot(L_estimated, 'r-')
        else:
            ax2.plot(L_estimated, 'r-')
        
        ax2.set_title("Estimated Latent Liquidity Process")
        ax2.set_ylabel("Liquidity")
        ax2.grid(True)
        
        pdf.savefig()
        plt.close()
    
    # Now create the LaTeX file
    with open(latex_file, 'w') as f:
        # Write LaTeX preamble
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage{booktabs}\n")
        f.write("\\usepackage{graphicx}\n")
        f.write("\\usepackage{float}\n")
        f.write("\\usepackage[margin=1in]{geometry}\n")
        f.write("\\begin{document}\n\n")
        
        # Title
        f.write("\\begin{center}\n")
        f.write("\\Large\\textbf{Soybean Meal Futures SDE Parameter Estimation Results}\n")
        f.write("\\end{center}\n\n")
        
        # Summary section
        f.write("\\section*{Data Summary}\n")
        f.write("\\begin{itemize}\n")
        f.write(f"\\item Number of observations: {len(prices)}\n")
        f.write(f"\\item Time period: {min(pd.to_datetime(dates))} to {max(pd.to_datetime(dates))}\n")
        f.write(f"\\item Time step (dt): {dt:.6f} years\n")
        f.write("\\end{itemize}\n\n")
        
        f.write("\\section*{Model Specification}\n")
        f.write("\\begin{itemize}\n")
        f.write("\\item Stochastic Differential Equation (SDE) with liquidity feedback\n")
        f.write("\\item Extended Kalman Filter (EKF) for latent state estimation\n")
        f.write("\\item Maximum Likelihood Estimation (MLE) for parameter estimation\n")
        f.write("\\end{itemize}\n\n")
        
        # Parameters table
        f.write("\\section*{Parameter Estimation Results}\n")
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lc}\n")
        f.write("\\toprule\n")
        f.write("Parameter & Soybean Meal Futures \\\\\n")
        f.write("\\midrule\n")
        
        # Add parameter rows
        for i, (name, value) in enumerate(zip(param_names, param_values)):
            f.write(f"{name} & {value} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Estimated parameters with t-statistics in parentheses}\n")
        f.write("\\end{table}\n\n")
        
        # Add note about significance
        f.write("\\begin{small}\n")
        f.write("Note: Values in parentheses are t-statistics.\\\\")
        f.write("Statistical significance: $|t| > 2.58$ (1\\%), $|t| > 1.96$ (5\\%), $|t| > 1.65$ (10\\%)\n")
        f.write("\\end{small}\n\n")
        
        # Add log-likelihood information
        if 'Log-Likelihood' in param_stats.index:
            log_lik = param_stats.loc['Log-Likelihood', 'Estimate']
            f.write(f"Log-Likelihood: {log_lik}\n\n")
        
        # End LaTeX document
        f.write("\\end{document}\n")
    
    print(f"Results saved to {pdf_file} and {latex_file}")

def run_parameter_estimation(csv_file_path, price_column='收盘价', date_column='交易时间', 
                            initial_params=None, plot_result=True, analyze_data=True,
                            save_results=True, output_file=None):
    """
    Load data from CSV file and run SDE parameter estimation
    
    Parameters:
    - csv_file_path: Path to CSV file
    - price_column: Price column name
    - date_column: Date column name
    - initial_params: Initial parameter estimates, if None use default values
    - plot_result: Whether to plot results
    - analyze_data: Whether to analyze data distribution
    - save_results: Whether to save results to file
    - output_file: Output file path, if None generates default name
    
    Returns:
    - estimated_params: Estimated parameters
    - L_estimated: Estimated liquidity sequence
    """
    # Load data
    prices, dates, dt = load_futures_data(csv_file_path, price_column, date_column)
    if prices is None:
        print("Data loading failed, cannot continue")
        return None, None
    
    # Visualize raw data
    if plot_result:
        visualize_raw_data(prices, dates, title="Soybean Meal Futures Prices")
    
    # Analyze returns (optional)
    if analyze_data:
        analyze_returns(prices)
    
    # If no initial parameters provided, use default values
    if initial_params is None:
        # Set initial parameters based on data characteristics
        # These are rough estimates, can be adjusted based on specific data
        mu = 0.02  # Initial drift estimate
        alpha = 0.5  # Mean-reversion speed
        beta = 0.2  # Liquidity impact coefficient
        theta_bar = 1.0  # Long-term average liquidity
        sigma_S = np.std(np.diff(np.log(prices))) * np.sqrt(252 if dt <= 1/200 else 1/dt)  # Annualized price volatility
        sigma_L = sigma_S * 0.5  # Liquidity volatility (rough estimate as half of price volatility)
        rho1 = 0.3  # Correlation coefficients
        rho2 = 0.2
        rho3 = 0.1
        
        # 添加lambda_和phi参数
        lambda_ = 5.0  # 非线性反馈强度参数
        phi = 0.5      # 控制非线性程度的幂参数
        
        initial_params = [mu, alpha, beta, theta_bar, sigma_S, sigma_L, rho1, rho2, rho3, lambda_, phi]
        
        print("\nAutomatically set initial parameters based on data:")
        print(f"mu (drift parameter): {mu:.4f}")
        print(f"alpha (mean-reversion speed): {alpha:.4f}")
        print(f"beta (liquidity impact): {beta:.4f}")
        print(f"theta_bar (long-term liquidity): {theta_bar:.4f}")
        print(f"sigma_S (price volatility): {sigma_S:.4f}")
        print(f"sigma_L (liquidity volatility): {sigma_L:.4f}")
        print(f"rho1, rho2, rho3 (correlations): {rho1:.2f}, {rho2:.2f}, {rho3:.2f}")
        print(f"lambda_ (非线性反馈强度): {lambda_:.2f}")
        print(f"phi (非线性程度的幂): {phi:.2f}")
    
    # Create estimator
    estimator = SDEParameterEstimator(prices, dt)
    
    # Estimate parameters
    print("\nStarting parameter estimation...")
    estimated_params, L_estimated = estimator.estimate_parameters(initial_params)
    
    # Calculate parameter significance
    print("Calculating parameter significance...")
    param_stats, log_likelihood = calculate_parameter_significance(estimated_params, prices, L_estimated, dt)
    
    # Print results
    print("\nParameter Estimation Results:")
    print(param_stats.to_string(index=False))
    
    # Visualize results
    if plot_result:
        estimator.plot_results(prices, L_estimated, estimated_params, 
                             title="Soybean Meal Futures SDE Parameter Estimation")
    
    # Save results to file
    if save_results:
        if output_file is None:
            # Generate default output filename
            base_name = os.path.splitext(csv_file_path)[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{base_name}_results_{timestamp}.pdf"
            
        save_results_to_file(prices, dates, estimated_params, L_estimated, param_stats, output_file, dt)
    
    return estimated_params, L_estimated, param_stats

if __name__ == "__main__":
    # Soybean Meal Futures CSV file path
    csv_file_path = "soybean_meal_futures.csv"
    
    # Column names based on the provided file
    price_column = '收盘价'  # Price column name
    date_column = '交易时间'  # Date column name
    
    # Run parameter estimation with output file saved to current directory
    estimated_params, L_estimated, param_stats = run_parameter_estimation(
        csv_file_path, 
        price_column=price_column, 
        date_column=date_column,
        save_results=True
    ) 