function x = Tri_LU_variable(alpha,beta,gamma,f)
% This code is used to solve Tridiagonal LU decomposition with variable
% diagonal numbers in the matrix
% gamma is the lower and beta is the upper
% for example (... gamma,alpha,beta ...)
% Coded on 13/11/2014

%% Gaussian elimination

n = length(alpha);
x = zeros(1,n);

for k=1:n-1
    
    m = gamma(k)/alpha(k);
    alpha(k+1) = alpha(k+1)-m*beta(k);
    f(k+1) = f(k+1)-m*f(k);
    
end

%% Solve the variables x

x(n) = f(n)/alpha(n);

for k=n-1:-1:1
    
    x(k) = (f(k)-beta(k)*x(k+1))/alpha(k);
    
end

end