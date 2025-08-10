function [ put] = liquidity_MC( r,sigmaS,alpha,beta,sigmaL,theta,rho1,rho2,rho3,L0,S0,tau,K)


m=500000;
n=101;
s=S0*ones(m,1);
l=L0*ones(m,1);
dt=tau/(n-1);  
a=(rho3-rho1*rho2)/sqrt(1-rho1^2);
for i=1:m
    w1=randn(1,n-1);
    B=randn(1,n-1);
    w2=rho1*w1+sqrt(1-rho1^2)*B;
    w3=rho2*w1+a*B+sqrt(1-rho2^2-a^2)*randn(1,n-1);
    for j=2:n
        s(i)=s(i)+r*s(i)*dt+sigmaS*s(i)*w1(j-1)*sqrt(dt)+beta*l(i)*s(i)*w2(j-1)*sqrt(dt);
        l(i)=l(i)+alpha*(theta-l(i))*dt+sigmaL*w3(j-1)*sqrt(dt);
    end
end
put=exp(-r*tau)*mean(max(K-s(:),0));

end