function Put=Fully_Explicit(S0,L0,tau,K,kappa,deltat,beta,rho1,rho2,rho3,sigmaS,sigmaL,alpha,theta,r,N_S,N_L,N_T)
% Use the explicit method solve the problem
% Using Neumann boundary condition for Lmin
% N_S=N_L
%14/10/2023
tic
%% 参数如下：
% S0=8
% L0=0.3
% K=10
% r=0.02
% beta=0.5
% sigma_S=0.3
% alpha=0.2
% theta=0.3
% sigma_L=0.9
% rho1=0.2
% rho2=0.8
% rho3=-0.5
%eta=0.5
%kappa=0.008
%tau=1
%deltat=1/12
Smax=4*K;
Smin=0;
Lmax=4*L0;
Lmin=0;
tmin=0;
dS = (Smax-Smin)/(N_S-1);
dL = (Lmax-Lmin)/(N_L-1);
dt = (tau-tmin)/(N_T-1);
S = linspace(Smin, Smax, N_S);
L = linspace(Lmin, Lmax, N_L);
V=zeros(N_S,N_L);
%result=zeros(N_S,N_v,N_T);
%% 函数主体
for i=1:N_S
    for j=1:N_L
        V(i,j) = max(K-S(i),0);%初始函数值
    end
end
for t=1:N_T-1
     u=V;%Update the temporary grid u(s,t)
    %boundary conditions
    for j=1:N_L-1
        V(1,j)=K*exp(-r*t*dt); %S=0
        V(N_S,j)=0; %S=Smax
    end
        for i=1:N_S
        V(i,N_L)=K; %L=Lmax
        end
    for i=2:N_S-1
        %Dirichlet condition for L=Lmin
        V(i,1)=0;
    end
     for i=2:N_S-1
         for j=2:N_L-1
             %设置手续费中的Phi
          Phi=beta*(i-1)*(j-1)*dL*(u(i+1,j)-2*u(i,j)+u(i-1,j))/dS;
          %设置手续费中的psi1
          Psi1=sigmaS*(i-1)*(u(i+1,j)-2*u(i,j)+u(i-1,j))/dS ;
           %设置手续费中的psi2
          Psi2=sigmaL*(u(i+1,j+1)-u(i+1,j-1)-u(i-1,j+1)+u(i-1,j-1))/(4*dS*dL);
          %时间层为n时V（i,j)
          A=1-dt*(i-1)^2*(beta^2*dL^2*(j-1)^2+sigmaS^2+2*rho1*sigmaS*beta*dL*(j-1))-dt*sigmaL^2/(dL^2)-dt*r;
          %时间层为n时V(i+1,j)前面的系数
          B=dt*0.5*(i-1)*((i-1)*(beta^2*dL^2*(j-1)^2+sigmaS^2+2*rho1*sigmaS*beta*dL*(j-1))+r);
          %时间层为n时V（i-1，j)前面的系数
          C=dt*0.5*(i-1)*((i-1)*(beta^2*dL^2*(j-1)^2+sigmaS^2+2*rho1*sigmaS*beta*dL*(j-1))-r);
          %时间层为n时V(i,j+1)前面的系数
          D=0.5*dt/dL*(sigmaL^2/dL+alpha*(theta-(j-1)*dL));
          %时间层为n时V(i,j-1)前面的系数
          E=0.5*dt/dL*(sigmaL^2/dL-alpha*(theta-(j-1)*dL));
          %时间层为n时V（i+1,j+1)-V(i+1,j-1)-V(i-1,j+1)+V(i-1,j-1)前面的系数
          F=dt*(rho3*sigmaL*beta*dL*(j-1)+rho2*sigmaS*sigmaL)*(i-1)/(4*dL);
          %时间层为n时的手续费
          G=dt*sqrt(2/(pi*deltat))*kappa*(i-1)*dS*(Phi^2+...
            Psi1^2+Psi2^2+...
            2*rho1*Psi1*Phi+2*rho2*Psi1*Psi2+...
            2*rho3*Phi*Psi2)^0.5;
          V(i,j)=A*u(i,j)+B*u(i+1,j)+C*u(i-1,j)+D*u(i,j+1)+E*u(i,j-1)+F*(u(i+1,j+1)-u(i+1,j-1)-u(i-1,j+1)+u(i-1,j-1))-G;
         end
     end
%        % For American options check if early exercise
%     for s = 1 : N_S
%             V(s,:) = max(V(s,:), K-S(s));
%     end
% end
Put=interp2(L,S,V,L0,S0);
toc
end