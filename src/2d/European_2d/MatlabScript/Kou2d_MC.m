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

%% parametri simulazione

Nsim=10000;
Nstep=100;

dt=T/(Nstep+1);
t=linspace(0,T,Nstep+1);

%% simulazione

rnd1=randn(Nsim, Nstep+1);
rnd2=randn(Nsim, Nstep+1);

S1=zeros(Nsim, Nstep+1);
S2=zeros(Nsim, Nstep+1);

S1(:,1)=S01;
S2(:,1)=S02;

NT1=icdf('Poisson', rand(Nsim,1), 1/lambda1);
NT2=icdf('Poisson', rand(Nsim,1), 1/lambda2);

for i=1:Nsim
    
    istante_salto1=sort(rand(NT1(i),1));
    indexes1=zeros(NT1(i),1);
    for j=1:NT1(i)
        [~, indice]=min(abs(istante_salto1(j)-t(2:end)));
        indexes1(j)=indice(1);
    end
    istante_salto1=t(indexes1+1);
    
    istante_salto2=sort(rand(NT2(i),1));
    indexes1=zeros(NT2(i),1);
    for j=1:NT2(i)
        [~, indice]=min(abs(istante_salto2(j)-t(2:end)));
        indexes2(j)=indice(1);
    end
    istante_salto2=t(indexes2+1);
        
    for j=1:Nstep
        S1(i,j+1)=S1(i,j)*exp((r-sigma1^2/2)*dt+sqrt(dt)*sigma1*(rho*rnd1(i,j)+sqrt(1-rho^2)*rnd2(i,j)));
        S2(i,j+1)=S2(i,j)*exp((r-sigma2^2/2)*dt+sqrt(dt)*sigma2*(rho*rnd2(i,j)+sqrt(1-rho^2)*rnd1(i,j)));
        
        for k=1:NT1(i)
            if istante_salto1(k)==t(j+1)
                if rand<p1
                    S1(i,j+1)=S1(i,j+1)*exp(icdf('exp', rand, 1/lambda_piu_1));
                else
                    S1(i,j+1)=S1(i,j+1)*exp(-icdf('exp', rand, 1/lambda_meno_1));
                end
            end
        end
        
        for k=1:NT2(i)
            if istante_salto2(k)==t(j+1)
                if rand<p2
                    S2(i,j+1)=S2(i,j+1)*exp(icdf('exp', rand, 1/lambda_piu_2));
                else
                    S2(i,j+1)=S2(i,j+1)*exp(-icdf('exp', rand, 1/lambda_meno_2));
                end
            end
        end
        
    end
end

% figure;
% plot(t,S1);
% figure;
% plot(t,S2);

[prezzo, ~, IC]=normfit(exp(-r*T)*max(S1(:,end)+S2(:,end)-K,0))
