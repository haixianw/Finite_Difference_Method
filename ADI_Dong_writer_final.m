function Put=ADI_Dong_writer_final(Sf,S0,L0,tau,K,k_TC,deltat,beta,rho1,rho2,rho3,sigmaS,sigmaL,alpha,theta,r,N_S,N_L,N_T,const)
% Use the ADI method solve the problem
% Using Neumann boundary condition for Lmin
% N_S=N_L
% 28/11/2023
% Coded by Dong Yan

%%   ����������ʾ
tic
Smax = 2*K;
Smin = 0;
Lmax = 5;
Lmin = 0;
tmin = 0;
dS = (Smax-Smin)/(N_S-1);
dL = (Lmax-Lmin)/(N_L-1);
dt = (tau-tmin)/(N_T-1);
S = linspace(Smin, Smax, N_S);
L = linspace(Lmin, Lmax, N_L);
t = linspace(tmin, tau, N_T);
u = zeros(N_S,N_L);
% parameter eta controls the type of weighing being implemented
% eta=0 produces the fully explicit scheme;
% eta=1/2 produces the Crank-Nicolson scheme;
% eta=1 produces the fully implicit scheme.
 eta = 1/2;
% eta=0;
% eta=1;
% Coefficient for convenience
q = eta * r * dt / 2;

% Initialization for theta_new
theta_new = zeros(N_L,1);
% Modification for theta
% theta_new=theta+k_TC*f(L)*constant
for j = 1 : N_L
% Choice 1: f(L)=ln(L+1)
% theta_new(j)=theta+k_TC*log((j-1)*dL+1)*const;
% Choice 2: f(L)=sqrt(L)
 theta_new(j)=theta+k_TC*sqrt((j-1)*dL)*const;  
% Choice 3: f(L)=L^(1/3)
% theta_new(j)=theta+k_TC*((j-1)*dL)^(1/3)*const;  
end
%result=zeros(N_S,N_v,N_T);

%% ��������
% ������ֵ����
for i = 1 : N_S
    for j = 1 : N_L
        u(i,j) = max(K-S(i),0);
    end
end
bndS0 = u(1,:);
BndS0 = zeros(1,N_L);

for n = 1 : N_T-1
    U = u; 
    % boundary conditions
    % Boundary condition for S=0: time dependent
        for j = 1 : N_L
            %European
        %u(1,j) = K*exp(-r*n*dt);
        %American
        u(1,j)=K;
        end
        BndS0(:) = u(1,:);
        InterBndS0 = (1 + q) * BndS0 - q * bndS0;
        bndS0 = BndS0;

    % Boundary condition for S=S_max
        u(N_S,:) = 0; 
   % Boundary condition 1 for L=0
%        u(2 : N_S-1,1) = BSPrice(2 : N_S-1,n+1);
%     % Boundary condition 2 for L=0
%          u(2 : N-1,1) = max(K-S(2:N-1), 0);
%    %  Boundary condition 3 for L=0
%          u(2 : N-1,1) = max(K*exp(-r*n*dt)-S(2:N-1), 0);
    % Boundary condition 4 for L=0 
    for i=2:N_S-1
        % Coefficient for V(i,1)
        a=1-(i-1)^2*sigmaS^2*dt+sigmaL^2*dt/(dL^2)-alpha*theta_new(1)*dt/dL-r*dt;
        % Coefficient for V(i+1,1)
        b=(i-1)*dt/2*((i-1)*sigmaS^2-rho2*sigmaS*sigmaL/dL+r);
        % Coefficient for V(i-1,1)
        c=(i-1)*dt/2*((i-1)*sigmaS^2+rho2*sigmaS*sigmaL/dL-r);
        % Coefficient for V(i,2)
        d=dt/dL*(alpha*theta_new(1)-5*sigmaL^2/(2*dL));
        e=sigmaL^2*dt/(2*dL^2);
        f=rho2*sigmaS*sigmaL*(i-1)*dt/(2*dL);
        % Transaction costs
        psi1=sigmaS*(i-1)/dS*(U(i+1,1)-2*U(i,1)+U(i-1,1));
        psi2=sigmaL/(2*dS*dL)*(U(i+1,2)-U(i+1,1)-U(i-1,2)+U(i-1,1));
        TC=sqrt(2/(pi*deltat))*k_TC*(i-1)*dS*dt...
            *sqrt(psi1^2+psi2^2+2*rho2*psi1*psi2);
        %Neumann condition for v=vmin
        u(i,1)=a*U(i,1)+b*U(i+1,1)+c*U(i-1,1)+d*U(i,2)+e*(4*U(i,3)-U(i,4))...
            +f*(U(i+1,2)-U(i-1,2))+TC;
    end
    %% step 1
    A = zeros(1,(N_S-2)*(N_L-2));
    B = zeros(1,(N_S-2)*(N_L-2)-1);
    C = zeros(1,(N_S-2)*(N_L-2)-1);
    RHS = zeros((N_S-2)*(N_L-2),1);
    %ʱ���Ϊn+0.5ʱu(i,j)ǰ���ϵ��
    for j = 2 : N_L-1
        for i = 2 : N_S-1
     A0 = (i-1)^2*(beta^2*(j-1)^2*dL^2+sigmaS^2+2*rho1*sigmaS*beta*(j-1)*dL);
     A((i-1+(j-2)*(N_S-2))) = 1+eta*dt*(A0+r/2);
        end
    end
    %ʱ���Ϊn+0.5ʱV(i+1,j)ǰ���ϵ��    
    for j = 2 : N_L-2
        for i = 2 : N_S-1
      A0 = (i-1)^2*(beta^2*(j-1)^2*dL^2+sigmaS^2+2*rho1*sigmaS*beta*(j-1)*dL);
      B((i-1+(j-2)*(N_S-2))) = -dt*eta*(A0+r*(i-1))/2;
        end
    end
    %ʱ���Ϊn+0.5ʱV(i-1,j)ǰ���ϵ��
    for j = 3 : N_L-1
      for i = 2 : N_S-1
      A0 = (i-1)^2*(beta^2*(j-1)^2*dL^2+sigmaS^2+2*rho1*sigmaS*beta*(j-1)*dL);
      C(i-2+(j-2)*(N_S-2)) = -dt*eta*(A0-r*(i-1))/2;
     end
    end
    % diag(B,1) for j=N_L-1
    for i = 2 : N_S-2
        A0 = (i-1)^2*(beta^2*(N_L-2)^2*dL^2+sigmaS^2+2*rho1*sigmaS*beta*(N_L-2)*dL);
        B(i-1+(N_L-3)*(N_S-2)) = -dt*eta*(A0+r*(i-1))/2;
    end
    % diag(C,-1) for j=2
    for i = 3 : N_S-1
        A0 = (i-1)^2*(beta^2*dL^2+sigmaS^2+2*rho1*sigmaS*beta*dL);
        C(i-2) = -dt*eta*(A0-r*(i-1))/2;
    end
    % Settings for RHS  
    for j = 2 : N_L-1
        for i = 2 : N_S-1
          %�����������е�Phi
          Phi = beta*(i-1)*(j-1)*dL*(U(i+1,j)-2*U(i,j)+U(i-1,j))/dS;
          %�����������е�psi1
          Psi1 = sigmaS*(i-1)*(U(i+1,j)-2*U(i,j)+U(i-1,j))/dS ;
          %�����������е�psi2
          Psi2 = sigmaL*(U(i+1,j+1)-U(i+1,j-1)-U(i-1,j+1)+U(i-1,j-1))/(4*dS*dL); 
          % coefficient for convenience
          A0 = (i-1)^2*(beta^2*(j-1)^2*dL^2+sigmaS^2+2*rho1*sigmaS*beta*dL*(j-1));
          %ʱ���ΪnʱV��i,j)
          a = 1+dt*(eta-1)*(A0+r/2)-dt*(sigmaL^2/(dL^2)+r/2);
          %ʱ���ΪnʱV(i+1,j)ǰ���ϵ��
          b = dt*(1-eta)*(A0+(i-1)*r)/2;
          %ʱ���ΪnʱV��i-1��j)ǰ���ϵ��
          c = dt*(1-eta)*(A0-(i-1)*r)/2;
          %ʱ���ΪnʱV(i,j+1)ǰ���ϵ��
          d = dt/dL*(sigmaL^2/dL+alpha*(theta_new(j)-(j-1)*dL))/2;
          %ʱ���ΪnʱV(i,j-1)ǰ���ϵ��
          e = dt/dL*(sigmaL^2/dL-alpha*(theta_new(j)-(j-1)*dL))/2;
          %ʱ���ΪnʱV��i+1,j+1)-u(i+1,j-1)-u(i-1,j+1)+u(i-1,j-1)ǰ���ϵ��
          f = dt*(rho3*sigmaL*beta*dL*(j-1)+rho2*sigmaS*sigmaL)*(i-1)/(4*dL);
          %ʱ���Ϊnʱ��������
          TC = dt*sqrt(2/(pi*deltat))*k_TC*(i-1)*dS...
              *sqrt(Phi^2+Psi1^2+Psi2^2+2*rho1*Psi1*Phi+2*rho2*Psi1*Psi2+2*rho3*Phi*Psi2);
          %ÿһ��ѭ���õ���pΪRHS��ÿһ��
          RHS(i-1+(j-2)*(N_S-2)) = a*U(i,j)+b*U(i+1,j)+c*U(i-1,j)+d*U(i,j+1)...
              +e*U(i,j-1)+f*(U(i+1,j+1)-U(i+1,j-1)-U(i-1,j+1)+U(i-1,j-1))+TC;
        end
    end
    %RHS�ĵ�һ�����c(2,2)*UU(1,2);UUΪn+1/2ʱ���ʱ�ĺ���
    C0 = beta^2*dL^2+sigmaS^2+2*rho1*sigmaS*beta*dL;
    RHS(1) = RHS(1)+dt*eta*(C0-r)*InterBndS0(2)/2;
    Y_vec = Tridiagonal_LU(A,B,C,RHS);
    Y = vector_to_matrix_column_final(Y_vec,N_S-2,N_L-2);%���õ�������ת��Ϊ��������step2�ļ���

%% step 2
    A = zeros(1,(N_S-2)*(N_L-2));
    B = zeros(1,(N_S-2)*(N_L-2)-1);
    C = zeros(1,(N_S-2)*(N_L-2)-1);
    RRHS = zeros((N_S-2)*(N_L-2),1);
    %ʱ���Ϊn+1ʱV(i,j)ǰ���ϵ��
    for i = 2 : N_S-1
        for j = 2 : N_L-1
      A(j-1+(i-2)*(N_L-2)) = 1+eta*dt*(sigmaL^2/(dL^2)+r/2);
        end
    end
    %���һ��A��N-1,N-1)Ҫ�ټ�ȥһ��B(N-1,N-1)
    A((N_S-2)*(N_L-2)) = A((N_S-2)*(N_L-2))-eta*dt*(sigmaL^2/dL+alpha*(theta_new(N_L-1)-(N_L-2)*dL))/(2*dL);
    %ʱ���Ϊn+1ʱV(i+1,j)ǰ���ϵ��
    for i = 2 : N_S-2
        for j = 2 : N_L-1
      B(j-1+(i-2)*(N_L-2)) = -eta*dt*(sigmaL^2/dL+alpha*(theta_new(j)-(j-1)*dL))/(2*dL);
        end
    end
    %ʱ���Ϊn+1ʱV(i-1,j)ǰ���ϵ��
    for i = 3 : N_S-1
        for j = 2 : N_L-1
      C(j-2+(i-2)*(N_L-2)) = -eta*dt*(sigmaL^2/dL-alpha*(theta_new(j)-(j-1)*dL))/(2*dL);
        end
    end
    % diag(B,1) for i=N_S-1
    for j = 2 : N_L-2
        B((j-1+(N_S-3)*(N_L-2))) = -eta*dt*(sigmaL^2/dL+alpha*(theta_new(j)-(j-1)*dL))/(2*dL);
    end
    % diag(C,-1) for i=2
    for j = 3 : N_L-1
        C(j-2) = -eta*dt*(sigmaL^2/dL-alpha*(theta_new(j)-(j-1)*dL))/(2*dL);
    end
    % RRHS
    for i = 2 : N_S-1
        for j = 2 : N_L-1
          %ʱ���ΪnʱV��i,j)
          a = dt*eta*(sigmaL^2/(dL^2)+r/2);
          %ʱ���ΪnʱV(i,j+1)ǰ���ϵ��
          b = -dt*eta*(sigmaL^2/dL+alpha*(theta_new(j)-(j-1)*dL))/(2*dL);
          %ʱ���ΪnʱV��i,j-1)ǰ���ϵ��
          c = -dt*eta*(sigmaL^2/dL-alpha*(theta_new(j)-(j-1)*dL))/(2*dL);
           %ÿһ��ѭ���õ���qΪRRHS��ÿһ��
          RRHS(j-1+(i-2)*(N_L-2)) = a*U(i,j)+b*U(i,j+1)+c*U(i,j-1)+Y(i-1,j-1);
        end
    end
   
    %RRHS�ĵ�һ�����c(2,2)*u(2,1);VΪn+1ʱ���ʱ�ĺ���
    RRHS(1) = RRHS(1)+eta*dt*(sigmaL^2/dL-alpha*(theta_new(2)-dL))/(2*dL)*u(2,1);
    Y_vec = Tridiagonal_LU(A,B,C,RRHS);
    Y = vector_to_matrix_final(Y_vec,N_S-2,N_L-2);%���õ�������ת��Ϊ�����������յļ���
    u(2:N_S-1,2:N_L-1) = Y;
    % Boundary condition for L=L_max
        u(2:N_S-1,N_L) = u(2:N_S-1,N_L-1); 

    % For the writer
    for s = 2 : N_S-1
        for d= 2 : N_L-1
             if S(s) <= Sf(d,n+1) 
             u(s,d) = max(K-S(s),0);  
            end
        end 
    end
end
Put=interp2(L,S,u,L0,S0);

toc
end