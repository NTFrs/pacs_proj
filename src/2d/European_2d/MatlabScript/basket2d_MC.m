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

%% parametri simulazione

Nsim=10000;
Nstep=1;

dt=T/Nstep;
t=linspace(0,T,Nstep);

%% simulazione

rnd1=randn(Nsim, Nstep+1);
rnd2=randn(Nsim, Nstep+1);

S1=zeros(Nsim, Nstep+1);
S2=zeros(Nsim, Nstep+1);

S1(:,1)=S01;
S2(:,1)=S02;

for i=1:Nsim
        
    for j=1:Nstep
        S1(i,j+1)=S1(i,j)*exp((r-sigma1^2/2)*dt+sqrt(dt)*sigma1*(rho*rnd1(i,j)+sqrt(1-rho^2)*rnd2(i,j)));
        S2(i,j+1)=S2(i,j)*exp((r-sigma2^2/2)*dt+sqrt(dt)*sigma2*(rho*rnd2(i,j)+sqrt(1-rho^2)*rnd1(i,j)));
    end
end

% figure;
% plot(t,S1);
% figure;
% plot(t,S2);

[prezzo, ~, IC]=normfit(exp(-r*T)*max(S1(:,end)+S2(:,end)-K,0))
