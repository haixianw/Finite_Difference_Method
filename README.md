# Finite Difference Method for American Options Pricing

This repository contains MATLAB code implementing the Alternating Direction Implicit (ADI) method for pricing American options with both exogenous and endogenous transaction costs.

## Features

- Implements ADI scheme for solving 2D PDE pricing problem
- Handles both holder and writer perspectives
- Includes transaction costs modeling
- Computes optimal exercise boundaries
- Supports variable parameter settings for:
  - Asset price (S)
  - Liquidity (L) 
  - Interest rate (r)
  - Volatilities (sigmaS, sigmaL)
  - Correlation coefficients (rho1, rho2, rho3)
  - Transaction costs (k_TC)

## Key Files

- `main.m` - Main script for running the pricing calculation
- `ADI_Dong_holder_final.m` - ADI implementation for option holder
- `ADI_Dong_writer_final.m` - ADI implementation for option writer
- `Tridiagonal_LU.m` - LU decomposition solver for tridiagonal systems
- Various matrix-vector conversion utilities

## Usage

Set parameters in `main.m` and run:

```matlab
[Sf, put] = ADI_Dong_holder_final(S0, L0, tau, K, k_TC, deltat, beta, rho1, rho2, rho3, sigmaS, sigmaL, alpha, theta, r, N_S, N_L, N_T, const)
```


## Parameter Estimation
These programs are designed to estimate model parameters from market data, including:
- Asset price volatility (σS)
- Liquidity volatility (σL)
- Correlation coefficients (ρ1, ρ2, ρ3)
- Other structural parameters

The estimation utilizes historical price and liquidity data to calibrate the model for practical applications.

## Reference

Based on the paper: "Pricing American options with exogenous and endogenous transaction costs"
