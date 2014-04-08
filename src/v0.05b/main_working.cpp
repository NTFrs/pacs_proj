#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/IterativeLinearSolvers>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>

using namespace std;
using namespace Eigen;

typedef Eigen::SparseMatrix<double, ColMajor> SpMatrix;
typedef Eigen::SparseVector<double> SpVector;

double pide(double S0, double K, double H, double T, double r, double sigma, double p, double lambda,
            double lambda_piu, double lambda_meno, int N, int M);

double k(double y, double p, double lambda, double lambda_piu, double lambda_meno);

void integrale_Levy(double &alpha, double &lambda2, double &Bmin, double &Bmax, double p, double lambda,
                    double lambda_piu, double lambda_meno, double xmin, double xmax, int NN);

void buildMatrix(SpMatrix &M1, SpMatrix &M2, SpMatrix &FF, double sigma, double r, double lambda, double alpha, int N, double dx, double dt);

double payoff(double x, double K, double S0);

void integrale2_Levy(VectorXd &J, double Bmin, double Bmax, VectorXd const &x, VectorXd const &u, int NN, double K, double S0,
                     double p, double lambda, double lambda_piu, double lambda_meno);

void f_u(VectorXd &val, double * x_array, double * u_array, VectorXd const &y, double K, double S0, int n);

int main(){
        
        //Dati
        double T=1;                 // Scadenza
        double K=90;                // Strike price
        double H=110;               // Barriera up
        double S0=95;               // Spot price
        double r=0.0367;            // Tasso risk free
        
        // Parametri della parte continua
        double sigma=0.120381;      // Volatilità
        
        // Parametri della parte salto
        double p=0.20761;           // Parametro 1 Kou
        double lambda=0.330966;     // Parametro 2 Kou
        double lambda_piu=9.65997;  // Parametro 3 Kou
        double lambda_meno=3.13868; // Parametro 4 Kou
        
        // Discretizzazione
        int N=10; // Spazio
        int M=10; // Tempo
        
        // Griglie
        double dt=T/M;
        double Smin=0.5*S0*exp((r-sigma*sigma/2)*T-6*sigma*sqrt(T));
        
        // Troncamento di x=log(S/S0)
        double xmin=log(Smin/S0);
        double xmax=log(H/S0);
        //x=linspace(xmin,xmax,N+1);
        double dx=(xmax-xmin)/N;
        
        VectorXd x(N+1);
        for (int i=0; i<N+1; ++i) {
                x(i)=xmin+i*dx;
        }
        
        double alpha, lambda2, Bmin, Bmax;
        
        integrale_Levy(alpha, lambda2, Bmin, Bmax, p, lambda, lambda_piu, lambda_meno, xmin, xmax, 2*N);
        
        cout<<"lambda "<<lambda<<" lambda2 "<<lambda2<<"\n";
        
        SpMatrix M1(N-1,N-1);
        SpMatrix M2(N-1,N-1);
        SpMatrix FF(N-1,N-1);
        
        buildMatrix(M1, M2, FF, sigma, r, lambda2, alpha, N, dx, dt);
        
        VectorXd u(x.size()-2);
        
        for (int i=0; i<u.size(); ++i) {
                u(i)=payoff(x(i+1), K, S0);
        }
        
        double BC1_0=M1.coeffRef(1,0); 
        double BC1_N=M1.coeffRef(0,1);
        double BC2_0=M2.coeffRef(1,0); 
        double BC2_N=M2.coeffRef(0,1);
        
        for (int j=M-1; j>=0; --j) {
                cout<<j<<"\n";
                
                // rhs
                VectorXd J(N-1);
                VectorXd z(u.size()+2);
                z(0)=(K-S0*exp(xmin))*exp(r*(T-(j+1)*dt));
                for (int i=1; i<u.size(); ++i) {
                        z(i)=u(i-1);
                }
                z(z.size()-1)=0;
                integrale2_Levy(J, Bmin, Bmax, x, z, N, K, S0, p, lambda, lambda_piu, lambda_meno);
                VectorXd rhs(N-1);
                rhs=FF*J;
                rhs(0)+=-BC1_0*(K-S0*exp(xmin))*exp(r*(T-(j-2)*dt))+BC2_0*(K-S0*exp(xmin))*exp(r*(T-(j-1)*dt));
                
                // Solver
                SparseLU<SpMatrix> solver;
                M1.makeCompressed();
                solver.analyzePattern(M1);
                solver.factorize(M1);
                u=solver.solve(M2*u+rhs);
        }
        
        VectorXd sol(u.size()+2);
        sol(0)=(K-S0*exp(xmin))*exp(r*T);
        for (int i=0; i<u.size(); ++i) {
                sol(i+1)=u(i);
        }
        sol(sol.size()-1)=0;

        VectorXd S(x.size());
        VectorXd Put(sol.size());
        for (int i=0; i<x.size(); ++i) {
                S(i)=S0*exp(x(i));
                Put(i)=sol(i)*exp(-r*T);
        }
        
        double x_array[S.size()];
        double u_array[Put.size()];
        
        for (int i=0; i<x.size(); ++i) {
                x_array[i]=S(i);
                u_array[i]=Put(i);
        }
        
        gsl_interp_accel *my_accel_ptr = gsl_interp_accel_alloc ();
        gsl_spline *my_spline_ptr = gsl_spline_alloc (gsl_interp_cspline, x.size());
        gsl_spline_init (my_spline_ptr, x_array, u_array, x.size());
        
        double Prezzo=gsl_spline_eval(my_spline_ptr, S0 , my_accel_ptr);
        
        gsl_spline_free(my_spline_ptr);
        gsl_interp_accel_free(my_accel_ptr);
        
        cout<<"Prezzo="<<setprecision(8)<<Prezzo<<"\n";
                
        return 0;
}

double k(double y, double p, double lambda, double lambda_piu, double lambda_meno){
        if (y>0)
                return p*lambda*lambda_piu*exp(-lambda_piu*y);
        else
                return (1-p)*lambda*lambda_meno*exp(lambda_meno*y);
}

void integrale_Levy(double &alpha, double &lambda2, double &Bmin, double &Bmax, double p, double lambda,
                    double lambda_piu, double lambda_meno, double xmin, double xmax, int NN){
        
        // Calcolo di Bmin e Bmax
        double step=0.5;
        double tol=10e-9;
        
        Bmin=xmin;
        Bmax=xmax;
        
        while(k(Bmin, p, lambda, lambda_piu, lambda_meno)>tol)
                Bmin-=step;
        
        while(k(Bmax, p, lambda, lambda_piu, lambda_meno)>tol)
                Bmax+=step;
        
        // Calcolo di alpha e lambda con formula dei trapezi
        double dB=(Bmax-Bmin)/(NN-1);
        VectorXd y(NN); // Nodi di quadratura
        VectorXd w(NN); // Pesi di quadratura
        
        for (int i=0; i<NN; ++i) {
                y(i)=Bmin+i*dB;
                w(i)=dB;
        }
        w(0)=dB/2;
        w(NN-1)=dB/2;
        
        alpha=0;
        lambda2=0;

        for (int i=0; i<NN; ++i) {
                alpha+=w(i)*(exp(y(i))-1)*k(y(i), p, lambda, lambda_piu, lambda_meno);
                lambda2+=w(i)*k(y(i), p, lambda, lambda_piu, lambda_meno);
        }
        
        return;
        
}

void buildMatrix(SpMatrix &M1, SpMatrix &M2, SpMatrix &FF, double sigma, double r, double lambda, double alpha, int N, double dx, double dt){
        
        double diff=sigma*sigma/2;
        double trasp=r-sigma*sigma/2-alpha;
        double reaz=-lambda;
       
        
        SpMatrix DD(N-1,N-1);
        SpMatrix DF(N-1,N-1);
        
        // Assemblaggio
        for (int i=0; i<N-1; ++i) {
                DD.insert(i,i)=2/dx;
                FF.insert(i,i)=2*dx/3;
        }
        for (int i=0; i<N-2; ++i) {
                DD.insert(i+1,i)=-1/dx;
                FF.insert(i+1,i)=dx/6;
                DF.insert(i+1,i)=-0.5;
        }
        for (int i=0; i<N-2; ++i) {
                DD.insert(i,i+1)=-1/dx;
                FF.insert(i,i+1)=dx/6;
                DF.insert(i,i+1)=0.5;
        }
        
        cout<<dt<<"\t"<<diff<<"\t"<<trasp<<"\t"<<reaz<<"\n";
        
        M1=FF/dt+0.5*(diff*DD-trasp*DF-reaz*FF);
        M2=FF/dt-0.5*(diff*DD-trasp*DF-reaz*FF);
        
        return;
}
         
double payoff(double x, double K, double S0){
        return max(K-S0*exp(x), 0.);
}


void integrale2_Levy(VectorXd &J, double Bmin, double Bmax, VectorXd const &x, VectorXd const &u, int NN, double K, double S0,
                     double p, double lambda, double lambda_piu, double lambda_meno){
        
        double dB=(Bmax-Bmin)/(NN-1);
        VectorXd y(NN); // Nodi di quadratura
        VectorXd w(NN); // Pesi di quadratura
        
        for (int i=0; i<NN; ++i) {
                y(i)=Bmin+i*dB;
                w(i)=dB;
        }
        w(0)=dB/2;
        w(NN-1)=dB/2;
        
        for (int i=0; i<J.size(); ++i) {
                J(i)=0;
        }
        
        double x_array[x.size()];
        double u_array[u.size()];
        
        for (int i=0; i<x.size(); ++i) {
                x_array[i]=x(i);
                u_array[i]=u(i);
        }
        
        VectorXd ones=VectorXd::Constant(y.size(),1);
        
        for (int i=0; i<J.size(); ++i) {
                VectorXd val;
                f_u(val,x_array,u_array,y+x(i+1)*ones,K,S0,x.size());
                for (int j=0; j<y.size(); ++j) {
                        J(i)+=w(j)*k(y(j), p, lambda, lambda_piu, lambda_meno)*val(j);
                }
        }
}

void f_u(VectorXd &val, double * x_array, double * u_array, VectorXd const &y, double K, double S0, int n){
        
        gsl_interp_accel *my_accel_ptr = gsl_interp_accel_alloc ();
        gsl_spline *my_spline_ptr = gsl_spline_alloc (gsl_interp_cspline, n);
        gsl_spline_init (my_spline_ptr, x_array, u_array, n);
        
        val=VectorXd(y.size());
        int j=0;
        int k=0;
        while (y(j)<x_array[0]) {
                val(k)=payoff(y(k),K,S0);
                ++k;
                ++j;
        }
        while (j<y.size() && y(j)<x_array[n-1]) {
                val(k)=gsl_spline_eval(my_spline_ptr, y(k) , my_accel_ptr);
                ++j;
                ++k;
        }
        for (int i=k; i<y.size(); ++i) {
                val(i)=0;
        }
        
        gsl_spline_free(my_spline_ptr);
        gsl_interp_accel_free(my_accel_ptr);
        
        return;
}


