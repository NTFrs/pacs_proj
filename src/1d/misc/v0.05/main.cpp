// These include files are already known to you. They declare the classes
// which handle triangulations and enumeration of degrees of freedom:
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
// And this is the file in which the functions are declared that create grids:
#include <deal.II/grid/grid_generator.h>

// The next three files contain classes which are needed for loops over all
// cells and to get the information from the cell objects. The first two have
// been used before to get geometric information from cells; the last one is
// new and provides information about the degrees of freedom local to a cell:
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>

// In this file contains the description of the Lagrange interpolation finite
// element:
#include <deal.II/fe/fe_q.h>

// And this file is needed for the creation of sparsity patterns of sparse
// matrices, as shown in previous examples:
#include <deal.II/dofs/dof_tools.h>

// The next two file are needed for assembling the matrix using quadrature on
// each cell. The classes declared in them will be explained below:
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

// The following three include files we need for the treatment of boundary
// values:
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

// We're now almost to the end. The second to last group of include files is
// for the linear algebra which we employ to solve the system of equations
// arising from the finite element discretization of the Laplace equation. We
// will use vectors and full matrices for assembling the system of equations
// locally on each cell, and transfer the results into a sparse matrix. We
// will then use a Conjugate Gradient solver to solve the problem, for which
// we need a preconditioner (in this program, we use the identity
// preconditioner which does nothing, but we need to include the file anyway):
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>

#include <deal.II/lac/sparse_direct.h>

// Finally, this is for output to a file and to the console:
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
#include <vector>

#include <deal.II/grid/grid_out.h>

#include <cmath>

#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>

using namespace std;
using namespace dealii;

//#define __VERBOSE__
#define __MATLAB__
#define __MARAZZINA__
#define __PIDE__

const int dim=1;
const double dt_=0.5;
const int refinement=3;

class Parametri{
public:
        //Dati
        double T;               // Scadenza
        double K;               // Strike price
        double S0;              // Spot price
        double r;               // Tasso risk free
        
        // Parametri della parte continua
        double sigma;           // Volatilità
        
        // Parametri della parte salto
        double p;           // Parametro 1 Kou
        double lambda;     // Parametro 2 Kou
        double lambda_piu;  // Parametro 3 Kou
        double lambda_meno; // Parametro 4 Kou
        
        Parametri()=default;
        Parametri(const Parametri &)=default;
};

class PayOff : public Function<dim>
{
public:
        PayOff (double K_, double S0_) : Function<dim>(), K(K_), S0(S0_) {};
        
        virtual double value (const Point<dim>   &p,
                              const unsigned int  component = 0) const;
private:
        double K;
        double S0;
};


double PayOff::value (const Point<dim>  &p,
                      const unsigned int component) const
{
        Assert (component == 0, ExcInternalError());
        return max(S0*exp(p(0))-K,0.);
}

class Boundary_Left_Side : public Function<dim>
{
public:
        Boundary_Left_Side() : Function< dim>() {};
        
        virtual double value (const Point<dim> &p, const unsigned int component =0) const;
        
};

double Boundary_Left_Side::value(const Point<dim> &p, const unsigned int component) const
{
        Assert (component == 0, ExcInternalError());
        return 0;
        
}

class Boundary_Right_Side: public Function<dim>
{
public:
        Boundary_Right_Side(double S0, double K, double r, double T, double t) :
        Function< dim>(), _S0(S0), _K(K), _r(r), _T(T), _t(t) {};
        virtual double value (const Point<dim> &p, const unsigned int component =0) const;
private:
        double _S0;
        double _K;
        double _r;
        double _T;
        double _t;
};

double Boundary_Right_Side::value(const Point<dim> &p, const unsigned int component) const
{
        return _S0*exp(p[0])-_K*exp(-_r*(_T-_t));
        
}

class Opzione{
private:
        Parametri par;
        void setup_system ();
        void solve () ;
        void output_results () const {};
	double GetPrice() const {};
        
        Triangulation<dim>   triangulation;
        FE_Q<dim>            fe;
        DoFHandler<dim>      dof_handler;
        
        SparsityPattern      sparsity_pattern;
        SparseMatrix<double> system_matrix;
        SparseMatrix<double> mass_matrix;
        SparseMatrix<double> laplace_matrix;
        SparseMatrix<double> df_matrix;
        
        Vector<double>       solution;
        Vector<double>       system_rhs;
        
        double               dt;
        
        double Smin, Smax, xmin, xmax;
        
        // Jump!
        double alpha, Bmin, Bmax;
        
        double k(double y);
        void integrale_Levy(int NN);
        void integrale2_Levy(Vector<double> &J, Vector<double> const &x, Vector<double> const &u, int NN);
        void f_u(Vector<double> &val, double * x_array, double * u_array, Vector<double> const &y, int n);
        double payoff(double x, double K, double S0);
public:
        Opzione(Parametri const &par_):
        par(par_),
        fe (1),
        dof_handler (triangulation),
        dt (dt_)
        {};
        double run(){
                setup_system();
                solve();
		double Price;
		Price=GetPrice();
		return Price;
        };
};

double Opzione::payoff(double x, double K, double S0){
        return max(S0*exp(x)-K, 0.);
}

double Opzione::k(double y){
        if (y>0)
                return par.p*par.lambda*par.lambda_piu*exp(-par.lambda_piu*y);
        else
                return (1-par.p)*par.lambda*par.lambda_meno*exp(par.lambda_meno*y);
}

void Opzione::integrale_Levy(int NN){
        
        //cout<<"NN "<<NN<<"\n";
                
        // Calcolo di Bmin e Bmax
        double step=0.5;
        double tol=10e-9;
        
        Bmin=xmin;
        Bmax=xmax;
        
        while(k(Bmin)>tol)
                Bmin-=step;
        
        while(k(Bmax)>tol)
                Bmax+=step;
        
        // Calcolo di alpha e lambda con formula dei trapezi
        double dB=(Bmax-Bmin)/(NN-1);
        Vector<double> y; // Nodi di quadratura
        Vector<double> w; // Pesi di quadratura
        y.reinit(NN);
        w.reinit(NN);
        
        for (int i=0; i<NN; ++i) {
                y(i)=Bmin+i*dB;
                w(i)=dB;
        }
        w(0)=dB/2;
        w(NN-1)=dB/2;
        
        alpha=0;
        
        for (int i=0; i<NN; ++i) {
                alpha+=w(i)*(exp(y(i))-1)*k(y(i));
        }
        
        return;
        
}

void Opzione::integrale2_Levy(Vector<double> &J, Vector<double> const &x, Vector<double> const &u, int NN) {
        
        //cout<<Bmax<<" "<<Bmin<<"NN "<<NN<<"\n";
        --NN;
        double dB=(Bmax-Bmin)/(NN-1);
        Vector<double> y(NN); // Nodi di quadratura
        Vector<double> w(NN); // Pesi di quadratura
        y.reinit(NN);
        w.reinit(NN);
        //cout<<"1\n";
        for (int i=0; i<NN; ++i) {
                y(i)=Bmin+i*dB;
                w(i)=dB;
        }
        w(0)=dB/2;
        w(NN-1)=dB/2;
        
        //cout<<"y ";
        for (int i=0; i<NN; ++i) {
                //cout<<y(i)<<"\t";
        }
        //cout<<"\n";
        
        //cout<<"w ";
        for (int i=0; i<NN; ++i) {
                //cout<<w(i)<<"\t";
        }
        //cout<<"\n";
        
        //cout<<"2\n";
        for (int i=0; i<J.size(); ++i) {
                J(i)=0;
        }
        //cout<<"3\n";
        double x_array[x.size()];
        double u_array[u.size()];
        
        for (int i=0; i<x.size(); ++i) {
                x_array[i]=x(i);
                u_array[i]=u(i);
        }
        //cout<<"4\n";
        //cout<<"J.size "<<J.size()<<" x.size "<<x.size()<<"\n";
        for (int i=1; i<J.size()-1; ++i) {
                //cout<<"*** "<<i<<" ***\n";
                Vector<double> val;
                Vector<double> z(y);
                z.add(x(i));/*
                if (i<x.size()-1) {
                        z.add(x(i));
                }
                else
                        z.add(x(i-1));*/
                //cout<<"z ";
                for (int j=0; j<z.size(); ++j) {
                        //cout<<z(j)<<"\t";
                }
                //cout<<"\n";
                //cout<<"5\n";
                //cout<<"z.size "<<z.size()<<"\n";
                //f_u(val,x_array,u_array,y+x(i+1)*ones,K,S0,x.size());
                f_u(val,x_array,u_array,z,x.size());
                cout<<"val ";
                for (int j=0; j<val.size(); ++j) {
                        cout<<val(j)<<" ";
                }
                cout<<"\n";
                //cout<<"6\nciclo:\n";
                for (int j=0; j<y.size(); ++j) {
                        J(i)+=w(j)*k(y(j))*val(j);
                }
                //cout<<"\n";
        }
        //cout<<"7\n";
        //cout<<"J ";
        for (int i=0; i<J.size(); ++i){
                //cout<<J(i)<<"\t";
        }
        //cout<<"\n";
}

void Opzione::f_u(Vector<double> &val, double * x_array, double * u_array, Vector<double> const &y, int n){
        
        gsl_interp_accel *my_accel_ptr = gsl_interp_accel_alloc ();
        gsl_spline *my_spline_ptr = gsl_spline_alloc (gsl_interp_cspline, n);
        gsl_spline_init (my_spline_ptr, x_array, u_array, n);
        
        //cout<<"n "<<n<<" y.size "<<y.size()<<"\n";
        
        val=Vector<double>(y.size());
        int j=0;
        int k=0;
        while (y(j)<x_array[0]) {
                val(k)=0;
                ++k;
                ++j;
        }
        //cout<<"j "<<j<<"\n";
        while (j<y.size() && y(j)<x_array[n-1]) {
                val(k)=gsl_spline_eval(my_spline_ptr, y(k) , my_accel_ptr);
                ++j;
                ++k;
        }
        //cout<<"j "<<j<<"\n";
        for (int i=k; i<y.size(); ++i) {
                val(i)=payoff(y(k),par.K,par.S0);
        }
        
        gsl_spline_free(my_spline_ptr);
        gsl_interp_accel_free(my_accel_ptr);
        
        return;
}

void Opzione::setup_system ()
{

#ifdef __PIDE__
        Smin=0.5*par.S0*exp((par.r-par.sigma*par.sigma/2)*par.T-6*par.sigma*sqrt(par.T));
        Smax=1.5*par.S0*exp((par.r-par.sigma*par.sigma/2)*par.T+6*par.sigma*sqrt(par.T));
#else
        Smin=par.S0*exp((par.r-par.sigma*par.sigma/2)*par.T-6*par.sigma*sqrt(par.T));
        Smax=par.S0*exp((par.r-par.sigma*par.sigma/2)*par.T+6*par.sigma*sqrt(par.T));
#endif
        
        xmin=log(Smin/par.S0);
        xmax=log(Smax/par.S0);
        
        double diff=par.sigma*par.sigma/2;
        double trasp=par.r-par.sigma*par.sigma/2;
        double reaz=-par.r;
        
        GridGenerator::hyper_cube (triangulation, xmin, xmax);
        triangulation.refine_global (refinement);
        
        std::cout << "Number of active cells: "
        << triangulation.n_active_cells()
        << std::endl;
        
        dof_handler.distribute_dofs (fe);
        
        std::cout << "Number of degrees of freedom: "
        << dof_handler.n_dofs()
        << std::endl
        << std::endl;
        
        sparsity_pattern.reinit (dof_handler.n_dofs(),
                                 dof_handler.n_dofs(),
                                 dof_handler.max_couplings_between_dofs());
        DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern);
        sparsity_pattern.compress();
        
        mass_matrix.reinit (sparsity_pattern);
        laplace_matrix.reinit (sparsity_pattern);
        df_matrix.reinit(sparsity_pattern);
        
        system_matrix.reinit(sparsity_pattern);
        
        MatrixCreator::create_mass_matrix (dof_handler, QGauss<dim>(2),
                                           mass_matrix);
        MatrixCreator::create_laplace_matrix (dof_handler, QGauss<dim>(2),
                                              laplace_matrix);
        // build df_matrix
        QGauss<dim> quadrature_formula(1);
        FEValues<dim> fe_values (fe, quadrature_formula,
                                 update_values | update_gradients | update_JxW_values);
        
        const unsigned int   dofs_per_cell = fe.dofs_per_cell;
        //const unsigned int   n_q_points    = quadrature_formula.size();
        
        FullMatrix<double>   cell_df (dofs_per_cell, dofs_per_cell);
        
        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
        
        DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
        for (; cell!=endc; ++cell)
        {
                fe_values.reinit (cell);
                cell_df = 0;
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                        for (unsigned int j=0; j<dofs_per_cell; ++j)
                                        cell_df(i,j)+=fe_values.shape_value(i,0)*
                                                fe_values.shape_grad(j,0)[0] * fe_values.JxW (0);
                
                cell->get_dof_indices (local_dof_indices);
                
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                        for (unsigned int j=0; j<dofs_per_cell; ++j)
                        {
                                df_matrix.add ( local_dof_indices[i],
                                                local_dof_indices[j],
                                                cell_df(i,j));
                        }
        }
        // build system_matrix //M1=FF/dt+(diff*DD-trasp*DF-reaz*FF)
        system_matrix.add(1./dt-reaz,mass_matrix);
        system_matrix.add(diff,laplace_matrix);
        system_matrix.add(-trasp,df_matrix);
        
        // print matrix
#ifdef __VERBOSE__
        cout<<"mass_matrix:\n";
        mass_matrix.print(cout);
        cout<<"laplace_matrix:\n";
        laplace_matrix.print(cout);
        cout<<"df_matrix:\n";
        df_matrix.print(cout);
        cout<<"system_matrix:\n";
        system_matrix.print(cout);
#endif
        
        solution.reinit(dof_handler.n_dofs());
        system_rhs.reinit(dof_handler.n_dofs());
        
        integrale_Levy(dof_handler.n_dofs());
        
        //cout<<"alpha "<<alpha<<" Bmin "<<Bmin<<" Bmax "<<Bmax<<"\n";
}

void Opzione::solve () {
        
#ifdef __PIDE__
        Vector<double> x;
        x.reinit(dof_handler.n_dofs());
        cout<<"grid=[ ";
        for (int i=0; i<x.size(); ++i) {
                x(i)=xmin+i*(xmax-xmin)/(x.size()-1);
                cout<<x(i)<<" ";
        }
        cout<<"]\n";
#endif
        cout<<"x.size "<<x.size()<<"\n";
        VectorTools::interpolate (dof_handler, PayOff(par.S0, par.K),solution);
        cout<<"solution:\n";
        solution.print(cout);
        cout<<"\n";
#ifdef __VERBOSE__
	cout<<"solution:\n";
        solution.print(cout);
        cout<<"\n";
#endif
        mass_matrix/=dt;
#ifdef __MARAZZINA__
        double temp2=mass_matrix(0,1);
        double temp1=system_matrix(0,1);
        mass_matrix.set(0,1,0);
        mass_matrix.set(1,0,0);
        mass_matrix.set(mass_matrix.n()-1,mass_matrix.n()-2,0);
        mass_matrix.set(mass_matrix.n()-2,mass_matrix.n()-1,0);
#endif
        
#ifdef __VERBOSE__
        cout<<"mass_matrix/dt:\n";
        mass_matrix.print(cout);
#endif
        double t=par.T-dt;
        double time_step=par.T/dt-1;
        
        for ( ; t>=0; t-=dt, --time_step) {
                cout<<"*** "<<time_step<<" ***\n";
#ifdef __PIDE__
                Vector<double> J;
                J.reinit(solution.size());
                //cout<<J.size()<<" "<<x.size()<<" "<<solution.size()<<"\n";
                integrale2_Levy(J, x, solution, 2*solution.size());
                mass_matrix.vmult (system_rhs, J);
#else
                mass_matrix.vmult (system_rhs, solution);
#endif
                
#ifdef __VERBOSE__
                cout<<"***t "<<t<<" time_step "<<time_step<<"\n";
                cout<<"mass_matrix:\n";
                mass_matrix.print(cout);
                cout<<"\nrhs before:\n";
                system_rhs.print(cout);
#ifdef __PIDE__
                cout<<"J:\n";
                J.print(cout);
#endif
#endif
                double temp3=system_rhs(dof_handler.n_dofs()-2);
                // BC
                {
                        std::map<types::global_dof_index,double> boundary_values;
                        VectorTools::interpolate_boundary_values (dof_handler,
                                                                  0,
                                                                  Boundary_Left_Side(),
                                                                  boundary_values);
                        
                        
                        VectorTools::interpolate_boundary_values (dof_handler,
                                                                  1,
                                                                  Boundary_Right_Side(par.S0, par.K, par.r, par.T, t),
                                                                  boundary_values);
                        
                        MatrixTools::apply_boundary_values (boundary_values,
                                                            system_matrix,
                                                            solution,
                                                            system_rhs);
                }
#ifdef __VERBOSE__
                cout<<"system_matrix mid:\n";
                system_matrix.print(cout);
                cout<<"rhs mid:\n";
                system_rhs.print(cout);
#endif
#ifdef __MARAZZINA__
                // -BC1_N*(Smax-K*exp(-r*(T-it*dt)))+BC2_N*(Smax-K*exp(-r*(T-(it+1)*dt)))
                system_rhs[dof_handler.n_dofs()-2]=temp3-temp1*
                (Smax-par.K*exp(-par.r*(par.T-time_step*dt)))+
                temp2*(Smax-par.K*exp(-par.r*(par.T-(time_step+1)*dt)));
#endif
                
#ifdef __VERBOSE__
                cout<<"system_matrix after:\n";
                system_matrix.print(cout);
                cout<<"rhs after:\n";
                system_rhs.print(cout);
#endif
                // Risolvo il sistema
                SparseDirectUMFPACK solver;
                solver.initialize(sparsity_pattern);
                solver.factorize(system_matrix);
                solver.solve(system_rhs);
                
                solution=system_rhs;
#ifdef __VERBOSE__
                cout<<"solution:\n";
                solution.print(cout);
                cout<<"\n";
#endif

        }
#ifndef __MATLAB__
        cout<<"solution:\n";
        solution.print(cout);
        cout<<"\n";
#else
        cout<<"x=[ ";
        for (int i=0; i<dof_handler.n_dofs()-1; ++i) {
                cout<<solution[i]<<"; ";
        }
        cout<<solution[dof_handler.n_dofs()-1]<<" ]";
#endif
}


int main(){
        
        Parametri par;
        par.T=1.;
        par.K=90;
        par.S0=95;
        par.r=0.0367;
        par.sigma=0.120381;
        
        // Parametri della parte salto
        par.p=0.20761;           // Parametro 1 Kou
        par.lambda=0.330966;     // Parametro 2 Kou
        par.lambda_piu=9.65997;  // Parametro 3 Kou
        par.lambda_meno=3.13868; // Parametro 4 Kou
        
        Opzione x(par);
        x.run();
        
        return 0;
}