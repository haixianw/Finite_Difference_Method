function x = Tridiagonal_LU(Alpha,Beta,Gamma,f)
% This is also included linear equations solving.
% alpha is the element of the diagnol of coefficient matrix A
% beta is the element above diagnol and gamma is under diagnol
% coded on 16/10/2014

n = length(f);
x = zeros(1,n);

%% Generate coefficient matrix A

%Alpha = alpha*ones(1,n);
%Beta = beta*ones(1,n-1);
%Gamma = gamma*ones(1,n-1);

A = diag(Alpha)+diag(Beta,1)+diag(Gamma,-1);
%% Gaussian elimination

for k=1:n-1
    
    m = Gamma(k)/Alpha(k);
    Alpha(k+1) = Alpha(k+1)-m*Beta(k);
    f(k+1) = f(k+1)-m*f(k);
    
end

%% Solve the variables x

x(n) = f(n)/Alpha(n);

for k=n-1:-1:1
    
    x(k) = (f(k)-Beta(k)*x(k+1))/Alpha(k);
    
end

end