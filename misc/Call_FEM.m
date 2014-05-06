% Risolvere con elementi finiti la pde di B&S -> per opzioni europea e
% opzioni barriera di tipo out (dove ho la barriera metteo BC=0)

% Usiamo il log price per avere la matrice con coeff costanti
% x=logS, dV/dt + (r-sigma^2/2)*dV/dx + sigma^2/2 * d2V/dx2 = rV

% function Prezzo=Call_FEM_Implicito(spot,K,r,sigma,T,L,N)

% L = #intervalli temporali
% N = #intervalli nel prezzo

%% (1) Crea la griglia

Smin=spot*exp((r-sigma^2/2)*T-6*sigma*sqrt(T));
Smax=spot*exp((r-sigma^2/2)*T+6*sigma*sqrt(T));

xmin=log(Smin);
xmax=log(Smax);

x=linspace(xmin,xmax,N+1);
dx=x(2)-x(1);
nodi=x(2:end-1)';

t=linspace(0,T,L+1);
dt=t(2)-t(1)

%% (2) Payoff (condizione finale)

V=max(exp(nodi)-K,0)

%% (3) Calcolo coefficienti della matrice

diff=sigma^2/2;
trasp=r-sigma^2/2;
reaz=-r;

%% (4) Matrici (derivataderivate, funzionefunzione, derivatafunzione)=(DD,FF,DF)
% Sono matrici N-1*N-1

tridiag=ones(N-1,1)*[-1 2 -1]/dx;
                %prima colonna sulla sottodiag, 2a sulla diagonale, 3a
                %colonna sulla sovrediagonale
DD=spdiags(tridiag,[-1 0 1],N-1,N-1)

tridiag=ones(N-1,1)*[1/6 2/3 1/6]*dx;
FF=spdiags(tridiag,[-1 0 1],N-1,N-1)

bidiag=ones(N-1,1)*[-0.5 0.5];
DF=spdiags(bidiag,[-1 1],N-1,N-1)

% Sono già trasposte, DD e FF sono simmetriche, DF invece è già trasposta,
% così che poi lavoriamo su questa senza dover trasporre più avanti

M1=FF/dt+(diff*DD-trasp*DF-reaz*FF)
M2=FF/dt

BC1_N=M1(1,2);
BC2_N=M2(1,2);

% per la put
% BC1_0=M1(2,1); cioè la sottodiagonale
% BC2_0=M2(2,1);

%% (5) Risoluzione

BC=sparse(N-1,1);
for it=L-1:-1:0
    % BC(1)=0 perché è una call
                   % sarebbe exp(xmax), ma allora metto direttamente Smax
    BC(end)=-BC1_N*(Smax-K*exp(-r*(T-it*dt)))+BC2_N*(Smax-K*exp(-r*(T-(it+1)*dt)));
    BC1_N
    it
    dt
    BC2_N
    rhs=M2*V
    rhs=rhs+BC
    V=M1\rhs
end

%% (6) Prezzo e Greche

Prezzo=interp1(nodi,V,log(spot),'spline');
V=[0;V;Smax-K*exp(-r*T)]
S=exp(x);

close all;

Smax-K

figure;
plot(S,V);
title('prezzo');

% con le greche ci sono problemi, perché non posso passare a exp(dx)
% perché la griglia non sarebbe più equidistanziata in S

title('stima derivata prima');

errore=abs(Prezzo-blsprice(spot,K,r,T,sigma))










