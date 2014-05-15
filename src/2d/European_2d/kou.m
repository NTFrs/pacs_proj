clc; clear all; close all;

%% Parametri
r=0.01;
K=200;
T=1;

S01=80;
sigma1=0.1256;

lambda1=0.330966;
p1=0.20761;
lambda_meno1=3.13868;
lambda_piu1=9.65997;

S02=120;
sigma2=0.2;

lambda2=0.330966;
p2=0.20761;
lambda_meno2=3.13868;
lambda_piu2=9.65997;

rho=-0.2;

Nsim=1000;
Nstep=100;

t=linspace(0,T,Nstep+1);

%% Simulo S1

NT=icdf('Poisson',rand(Nsim,1),lambda1*T);

X=zeros(Nsim,Nstep+1);
temp_1=randn(Nstep,Nsim);                           % Simulo W1_t
temp_2=rho*temp_1+sqrt(1-rho^2)*randn(Nstep,Nsim);  % Simulo W2_t
dt=t(2)-t(1);

indice=zeros(length(NT));

for k=1:Nsim
    % genero gli istanti di salto
    istante_salto=sort(rand(NT(k),1)*T);
    % sposto gli istanti di salto nel punto più vicino sulla griglia
    for i=1:NT(k)
        [minimo,index]=min(abs(istante_salto(i)-t(2:end)));
        indice(i)=index(1); % salvo gli indici degli istanti vicini
    end
    istante_salto_new=t(indice+1)'; % metto i nuovi indici in istante_salto_new
    
    % numeri casuali che userò nel ciclo successivo
    temp1=rand(NT(k),1);
    temp2=rand(NT(k),1);
    
    for i=1:Nstep
        %Simulazione parte continua
        X(k,i+1)=X(k,i)+r*dt+sigma1*sqrt(dt)*temp_1(i,k);
        %Aggiunta del salto
        for j=1:NT(k)
            if istante_salto_new(j)==t(i+1)
                if temp1(j)>p1
                    X(k,i+1)=X(k,i+1)-icdf('Exp',temp2(j), 1/lambda_meno1);
                else
                    X(k,i+1)=X(k,i+1)+icdf('Exp',temp2(j), 1/lambda_piu1);
                end
            end
        end
    end
end

S1=S01*exp(X);


figure;
plot(t,S1);
title('S1');

%% Simulo S2

NT=icdf('Poisson',rand(Nsim,1),lambda2*T);

indice=zeros(length(NT));

for k=1:Nsim
    % genero gli istanti di salto
    istante_salto=sort(rand(NT(k),1)*T);
    % sposto gli istanti di salto nel punto più vicino sulla griglia
    for i=1:NT(k)
        [minimo,index]=min(abs(istante_salto(i)-t(2:end)));
        indice(i)=index(1); % salvo gli indici degli istanti vicini
    end
    istante_salto_new=t(indice+1)'; % metto i nuovi indici in istante_salto_new
    
    % numeri casuali che userò nel ciclo successivo
    temp1=rand(NT(k),1);
    temp2=rand(NT(k),1);
    
    for i=1:Nstep
        %Simulazione parte continua
        X(k,i+1)=X(k,i)+r*dt+sigma2*sqrt(dt)*temp_2(i,k);
        %Aggiunta del salto
        for j=1:NT(k)
            if istante_salto_new(j)==t(i+1)
                if temp1(j)>p1
                    X(k,i+1)=X(k,i+1)-icdf('Exp',temp2(j), 1/lambda_meno2);
                else
                    X(k,i+1)=X(k,i+1)+icdf('Exp',temp2(j), 1/lambda_piu2);
                end
            end
        end
    end
end

S2=S02*exp(X);


figure;
plot(t,S2);
title('S2');

Prezzo=exp(-r*T)*mean(max(S1(:,2)+S2(:,2)-K,0))