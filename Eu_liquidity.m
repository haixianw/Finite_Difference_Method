function put=Eu_liquidity(S0,L0,K,tau,r,alpha,beta,theta,sigmaS,sigmaL,rho1,rho2,rho3,N_S,N_L,N_T)
% Solving our paper: American option with liquidity risk+transaction costs
% by the explicit finite difference method
% Coded on 16/11/23

%% Parameters
% S0=8;
% L0=0.3;
% K=10;
% r=0.02;
% beta=0.5;
% sigma_S=0.3;
% alpha=0.2;
% theta=0.3;
% sigma_L=0.9;
% rho1=0.2;
% rho2=0.8;
% rho3=-0.5;
% kappa=0.008;
% tau=1;
% ddt=1/12;
% eta=0.5;
tic
% deltat=1/6;
% k_TC=0.001;

% Stock price
Smax=4*K;
Smin=0;
% Liquidity risk
Lmax=20*L0;
Lmin=0;

tmin=0;

% building uniform grids
% N_S = 100;
% N_L = 100;
% N_T = 80000;

S = linspace(Smin, Smax, N_S);
L = linspace(Lmin, Lmax, N_L);
t = linspace(tmin, tau, N_T);

% space and time interval
dS = (Smax -Smin) / (N_S-1);
dL = (Lmax - Lmin) / (N_L-1);
dt = (tau-tmin) / (N_T-1);

U=zeros(N_S,N_L);
%result=zeros(N_S,N_L,N_T);
%% main body
% initial condition
for i=1:N_S  
        U(i,:) = max(K-S(i),0);
end
% result(:,:,1)=U(:,:);

for n=1:N_T-1
    V=U;%Update the temporary grid u(s,t)
    % boundary condition in S direction
    for j=1:N_L
           U(1,j)=K;
            U(N_S,j)=0; % S=Smax 
    end
    % Dirichlet boundary condition in L direction    
    for i=2:N_S-1
           U(i,N_L)=U(i,N_L-1); % L=Lmax
    end   
    % Neumann boundary condition for L=0 
%     for i=2:N_S-1
%         % Coefficient for V(i,1)
%         A=1-(i-1)^2*sigmaS^2*dt+sigmaL^2*dt/(dL^2)-alpha*theta*dt/dL-r*dt;
%         % Coefficient for V(i+1,1)
%         B=(i-1)*dt/2*((i-1)*sigmaS^2-rho2*sigmaS*sigmaL/dL+r);
%         % Coefficient for V(i-1,1)
%         C=(i-1)*dt/2*((i-1)*sigmaS^2+rho2*sigmaS*sigmaL/dL-r);
%         % Coefficient for V(i,2)
%         D=dt/dL*(alpha*theta-5*sigmaL^2/(2*dL));
%         E=sigmaL^2*dt/(2*dL^2);
%         F=rho2*sigmaS*sigmaL*(i-1)*dt/(2*dL);
%         % Transaction costs
%         psi1=sigmaS*(i-1)/dS*(V(i+1,1)-2*V(i,1)+V(i-1,1));
%         psi2=sigmaL/(2*dS*dL)*(V(i+1,2)-V(i+1,1)-V(i-1,2)+V(i-1,1));
%         TC=sqrt(2/(pi*deltat))*k_TC*(i-1)*dS*dt...
%             *sqrt(psi1^2+psi2^2+2*rho2*psi1*psi2);
%         %Neumann condition for v=vmin
%         U(i,1)=A*V(i,1)+B*V(i+1,1)+C*V(i-1,1)+D*V(i,2)+E*(4*V(i,3)-V(i,4))...
%             +F*(V(i+1,2)-V(i-1,2))-TC;
%     end

    for i=2:N_S-1
        % Coefficient for V(i,1)
        A=1-r*dt;

        B=(i-1)^2*sigmaS^2*dt/2;
        C=sigmaL^2*dt/(2*dL^2);
        D=rho2*sigmaS*sigmaL*(i-1)*dt/(2*dL);
        E=r*(i-1)*dt/2;
        F=alpha*theta*dt/dL;
        
        %Neumann condition for v=vmin
        U(i,1)=A*V(i,1)+B*(V(i+1,1)-2*V(i,1)+V(i-1,1))...
            +C*(2*V(i,1)-5*V(i,2)+4*V(i,3)-V(i,4))...
            +D*(V(i+1,2)-V(i+1,1)-V(i-1,2)+V(i-1,1))+E*(V(i+1,1)-V(i-1,1))...
            +F*(V(i,2)-V(i,1));
    end
    
    V=U;%Update the temporary grid u(s,t)
    for i = 2 : N_S-1
        for j = 2 : N_L-1
            A=1-(i-1)^2*dt*(beta^2*(j-1)^2*dL^2+sigmaS^2+2*rho1*sigmaS*beta*(j-1)*dL)...
                -sigmaL^2*dt/(dL^2)-r*dt;
            B0=(i-1)*(beta^2*(j-1)^2*dL^2+sigmaS^2+2*rho1*sigmaS*beta*(j-1)*dL);
            B1=(i-1)*dt/2*(B0+r);
            B2=(i-1)*dt/2*(B0-r);
            C=dt/(2*dL)*(sigmaL^2/dL+alpha*(theta-(j-1)*dL));
            D=dt/(2*dL)*(sigmaL^2/dL-alpha*(theta-(j-1)*dL));
            E=(rho3*beta*(j-1)*dL+rho2*sigmaS)*(i-1)*dt*sigmaL/(4*dL);
            U(i,j)=A*V(i,j)+B1*V(i+1,j)+B2*V(i-1,j)+C*V(i,j+1)+D*V(i,j-1)...
                +E*(V(i+1,j+1)-V(i+1,j-1)-V(i-1,j+1)+V(i-1,j-1));
        end
    end

% n   
end

put = interp2(L, S, U, L0, S0);

toc
end