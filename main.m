S0=3667;
L0=0.2;
K=3200;
tau=0.098630136986;
r=0.015;
alpha=1.9924;
beta=0;
theta=0.2498;
sigmaS=0.1953;
sigmaL=0.1685;
rho1=0.1697;
rho2=0.5035;
rho3=0.3045;
deltat=1/12;
k_TC=0.00005;
const=5;



% EOC for the tau direction
N_S=100;
N_L=100;
N_T=[1000]';

for i = 1 : size(N_T)
% print out N_T and 10 option prices, with 5 holding price + 5 writing price
N_T(i)
% ADI scheme 
Sf,put=ADI_Dong_holder_final(S0,L0,tau,K,k_TC,deltat,beta,rho1,rho2,rho3,sigmaS,sigmaL,alpha,theta,r,N_S,N_L,N_T(i),const);
end



