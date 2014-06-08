clc; clear all; close all;

%% parametri modello

T=1.;
K=200;
S01=80;
S02=120;
r=0.1;
sigma1=0.1256;
sigma2=0.2;
rho=-0.2;

p1=0.20761;
lambda1=0.330966;
lambda_piu_1=9.65997;
lambda_meno_1=3.13868;

p2=0.20761;
lambda2=0.330966;
lambda_piu_2=9.65997;
lambda_meno_2=3.13868;

b1=-(sigma1^2/2+lambda1*(p1/(lambda_piu_1-1)-(1-p1)/(lambda_meno_1+1)));
b2=-(sigma2^2/2+lambda2*(p2/(lambda_piu_2-1)-(1-p2)/(lambda_meno_2+1)));

%% parametri simulazione

Nsim=50000;
Nstep=252;

dt=T/(Nstep+1);
t=linspace(0,T,Nstep+1);

%% simulazione

rnd1=randn(Nsim, Nstep+1);
rnd2=randn(Nsim, Nstep+1);

X1=zeros(Nsim, Nstep+1);
X2=zeros(Nsim, Nstep+1);

S1=zeros(Nsim, Nstep+1);
S2=zeros(Nsim, Nstep+1);

NT1=icdf('Poisson', rand(Nsim,1), lambda1*T);
NT2=icdf('Poisson', rand(Nsim,1), lambda2*T);

for i=1:Nsim
    
    istante_salto1=sort(rand(NT1(i),1));
    indexes1=zeros(NT1(i),1);
    for j=1:NT1(i)
        [~, indice]=min(abs(istante_salto1(j)-t(2:end)));
        indexes1(j)=indice(1);
    end
    istante_salto1=t(indexes1+1);
    
    istante_salto2=sort(rand(NT2(i),1));
    indexes2=zeros(NT2(i),1);
    for j=1:NT2(i)
        [~, indice]=min(abs(istante_salto2(j)-t(2:end)));
        indexes2(j)=indice(1);
    end
    istante_salto2=t(indexes2+1);
        
    u_intens_1=rand(NT1(i),1);
    u_intens_2=rand(NT2(i),1);
    
    for j=1:Nstep
        X1(i,j+1)=X1(i,j)+b1*dt+sqrt(dt)*sigma1*(rho*rnd1(i,j)+sqrt(1-rho^2)*rnd2(i,j));
        X2(i,j+1)=X2(i,j)+b2*dt+sqrt(dt)*sigma2*(rho*rnd2(i,j)+sqrt(1-rho^2)*rnd1(i,j));
        
        for k=1:NT1(i)
            if istante_salto1(k)==t(j+1)
                if rand<p1
                    X1(i,j+1)=X1(i,j+1)+icdf('exp', u_intens_1(k), 1/lambda_piu_1);
                else
                    X1(i,j+1)=X1(i,j+1)-icdf('exp', u_intens_1(k), 1/lambda_meno_1);
                end
            end
        end
        
        for k=1:NT2(i)
            if istante_salto2(k)==t(j+1)
                if rand<p2
                    X2(i,j+1)=X2(i,j+1)+icdf('exp', u_intens_2(k), 1/lambda_piu_2);
                else
                    X2(i,j+1)=X2(i,j+1)-icdf('exp', u_intens_2(k), 1/lambda_meno_2);
                end
            end
        end
    end
end

S1=S01*exp(ones(Nsim,1)*r*t+X1);
S2=S02*exp(ones(Nsim,1)*r*t+X2);

figure;
plot(t,S1);
figure;
plot(t,S2);

[prezzo, ~, IC]=normfit(exp(-r*T)*max(S1(:,end)+S2(:,end)-K,0))
