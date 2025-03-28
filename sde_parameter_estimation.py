import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

class SDEParameterEstimator:

    
    def __init__(self, stock_prices, dt):
        """
        初始化估计器
        
        参数:
        - stock_prices: 股票价格时间序列数据 (numpy array)
        - dt: 时间步长
        """
        self.stock_prices = stock_prices
        self.dt = dt
        self.n = len(stock_prices)

    def extended_kalman_filter(self, params, S_observed):

        # 解包参数
        mu, alpha, beta, theta_bar, sigma_S, sigma_L, rho1, rho2, rho3,lambda_,phi = params
        kappa=0.00005
        n = len(S_observed)
        dt = self.dt
        
        # 初始化
        L_estimated = np.zeros(n)
        L_estimated[0] = theta_bar  # 使用长期均值作为初始估计
        
        # 初始协方差矩阵 - 基于Ornstein-Uhlenbeck过程的稳态方差
        P_prev = sigma_L**2 / (2*alpha)
        
        # 初始化对数似然
        log_likelihood = 0.0
        
        for t in range(1, n):
            # 上一个观测值和状态
            S_prev = S_observed[t-1]
            L_prev = L_estimated[t-1]
            
            # 预测步骤
            # 预测L
            L_pred = L_prev + alpha * (theta_bar + kappa*lambda_*(L_prev**phi) - L_prev) * dt
            
            # 计算状态方程的雅可比矩阵 F = ∂f/∂L
            F = 1 + dt * (alpha * kappa * lambda_ * phi * L_prev**(phi-1) - alpha)
            
            # 预测协方差，考虑过程噪声和相关性
            # 计算过程噪声协方差Q，考虑相关系数rho3对流动性过程的影响
            Q = sigma_L**2 * dt  # 基础过程噪声方差
            
            # 修正Q以考虑布朗运动相关性
            # rho3影响S和L布朗运动之间的相关性
            Q_adjusted = Q * (1 + rho3**2)  # 相关性会增加有效噪声方差
            
            P_pred = F**2 * P_prev + Q_adjusted
            
            # 预测观测值
            dS_expected = mu * S_prev * dt  # 使用估计的mu作为漂移项
            
            # 观测方程的雅可比矩阵 H = ∂h/∂L
            # beta控制流动性对价格波动的影响
            H = beta * S_prev * np.sqrt(dt)
            
            # 计算创新(innovation)
            dS_observed = S_observed[t] - S_prev
            innovation = dS_observed - dS_expected
            
            # 计算观测噪声方差R，考虑相关系数对观测方程的影响
            R_base = (sigma_S * S_prev * np.sqrt(dt))**2
            
            # 相关系数rho1, rho2, rho3共同影响观测噪声
            # rho1: 流动性冲击与价格直接冲击的相关性
            # rho2: 流动性冲击与流动性过程的相关性
            corr_factor = 1 + 2*rho1*beta/sigma_S + rho2*rho3
            R = R_base * max(0.01, corr_factor)  # 确保R为正
            
            # 计算创新协方差
            innovation_variance = H**2 * P_pred + R
            
            # 卡尔曼增益
            K = P_pred * H / innovation_variance
            
            # 更新步骤
            L_estimated[t] = L_pred + K * innovation
            P_new = (1 - K * H) * P_pred
            
            # 更新对数似然
            log_likelihood -= 0.5 * (np.log(2*np.pi*innovation_variance) + 
                                    innovation**2/innovation_variance)
            
            # 为下一个时间步更新P
            P_prev = P_new
        
        return L_estimated, log_likelihood
    
    def negative_log_likelihood(self, params):
        """
        计算负对数似然函数，用于参数估计
        
        参数:
        - params: (mu, alpha, beta, theta_bar, sigma_S, sigma_L, rho1, rho2, rho3)
        
        返回:
        - 负对数似然值
        """
        # 检查参数约束
        mu, alpha, beta, theta_bar, sigma_S, sigma_L, rho1, rho2, rho3,lambda_,phi = params
        
        # 验证参数是否在有效范围内
        if (alpha <= 0 or sigma_S <= 0 or sigma_L <= 0 or theta_bar <= 0 or
            abs(rho1) >= 1 or abs(rho2) >= 1 or abs(rho3) >= 1):
            return 1e10  # 返回一个大的惩罚值
        
        try:
            # 运行扩展卡尔曼滤波并计算对数似然
            _, log_likelihood = self.extended_kalman_filter(params, self.stock_prices)
            return -log_likelihood  # 返回负对数似然用于最小化
        except Exception as e:
            print(f"Error in likelihood calculation: {e}")
            return 1e10  # 出错时返回大的惩罚值
    
    def estimate_parameters(self, initial_params, bounds=None):
        """
        估计SDE模型的参数
        
        参数:
        - initial_params: 初始参数猜测 (mu, alpha, beta, theta_bar, sigma_S, sigma_L, rho1, rho2, rho3, lambda_, phi)
        - bounds: 参数边界的列表，如不提供则使用默认边界
        
        返回:
        - estimated_params: 估计的参数
        - L_estimated: 估计的流动性序列
        """
        if bounds is None:
            bounds = [
                (-0.5, 0.5),      # mu
                (1e-6, 10),       # alpha
                (-10, 10),        # beta
                (1e-6, 10),       # theta_bar
                (1e-6, 2),        # sigma_S
                (1e-6, 2),        # sigma_L
                (-0.99, 0.99),    # rho1
                (-0.99, 0.99),    # rho2
                (-0.99, 0.99),    # rho3
                (0.1, 10),        # lambda_
                (0.1, 2)          # phi
            ]
        
        # 使用L-BFGS-B算法优化参数（支持边界约束）
        result = minimize(
            self.negative_log_likelihood,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'disp': True}
        )
        
        # 检查优化结果
        if result.success:
            estimated_params = result.x
            print(f"Optimization successful: {result.message}")
        else:
            print(f"Optimization failed: {result.message}")
            estimated_params = initial_params
        
        # 使用估计的参数计算潜在流动性过程
        L_estimated, _ = self.extended_kalman_filter(estimated_params, self.stock_prices)
        
        return estimated_params, L_estimated
    
    def plot_results(self, S_observed, L_estimated, params, title="SDE Parameter Estimation Results"):
        """
        绘制结果
        """
        mu, alpha, beta, theta_bar, sigma_S, sigma_L, rho1, rho2, rho3,lambda_,phi = params
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        time = np.arange(len(S_observed)) * self.dt
        
        # 绘制股票价格
        ax1.plot(time, S_observed, 'b-', label='Observed Stock Price')
        ax1.set_ylabel('Stock Price (S)')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制估计的流动性
        ax2.plot(time, L_estimated, 'r-', label='Estimated Liquidity (L)')
        ax2.set_ylabel('Liquidity (L)')
        ax2.set_xlabel('Time')
        ax2.legend()
        ax2.grid(True)
        
        fig.suptitle(title)
        
        # 显示估计的参数
        param_text = (
            f"Estimated Parameters:\n"
            f"μ = {mu:.4f}, α = {alpha:.4f}, β = {beta:.4f}, θ̄ = {theta_bar:.4f}\n"
            f"σ_S = {sigma_S:.4f}, σ_L = {sigma_L:.4f}, λ = {lambda_:.4f}, φ = {phi:.4f}\n"
            f"ρ1 = {rho1:.4f}, ρ2 = {rho2:.4f}, ρ3 = {rho3:.4f}"
        )
        fig.text(0.5, 0.01, param_text, ha='center', fontsize=10)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.97])
        plt.show()

