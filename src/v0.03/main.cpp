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

using namespace std;
using namespace dealii;

//#define __VERBOSE__
//#define __MATLAB__

const int dim=1;
const double dt_=0.001;
const int refinement=5;

class Parametri{
public:
        //Dati
        double T;               // Scadenza
        double K;               // Strike price
        double S0;              // Spot price
        double r;               // Tasso risk free
        
        // Parametri della parte continua
        double sigma;           // Volatilità
        
        Parametri()=default;
        Parametri(const Parametri &)=default;
};

class PayOff : public Function<dim>
{
public:
        PayOff (double K_) : Function<dim>(), K(K_) {};
        
        virtual double value (const Point<dim>   &p,
                              const unsigned int  component = 0) const;
private:
        double K;
};


double PayOff::value (const Point<dim>  &p,
                      const unsigned int component) const
{
        Assert (component == 0, ExcInternalError());
        return max(exp(p(0))-K,0.);
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
        Boundary_Right_Side(double K) : Function< dim>(), _K(K) {};
        
        virtual double value (const Point<dim> &p, const unsigned int component =0) const;
private:
        double _K;
};

double Boundary_Right_Side::value(const Point<dim> &p, const unsigned int component) const
{
        Assert (component == 0, ExcInternalError());
        return exp(p[0])-_K;
        
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

void Opzione::setup_system ()
{
        
        Smin=par.S0*exp((par.r-par.sigma*par.sigma/2)*par.T-6*par.sigma*sqrt(par.T));
        Smax=par.S0*exp((par.r-par.sigma*par.sigma/2)*par.T+6*par.sigma*sqrt(par.T));
        
        xmin=log(Smin);
        xmax=log(Smax);
        
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
}

void Opzione::solve () {
        
        VectorTools::interpolate (dof_handler, PayOff(par.K),solution);
#ifdef __VERBOSE__
	cout<<"solution:\n";
        solution.print(cout);
        cout<<"\n";
#endif
        mass_matrix/=dt;
        
#ifdef __VERBOSE__
        cout<<"mass_matrix/dt:\n";
        mass_matrix.print(cout);
#endif
        for (double t=par.T; t>0; t-=dt) {
                mass_matrix.vmult (system_rhs, solution);
#ifdef __VERBOSE__
                cout<<"rhs:\n";
                system_rhs.print(cout);
#endif
                // BC
                {
                        std::map<types::global_dof_index,double> boundary_values;
                        VectorTools::interpolate_boundary_values (dof_handler,
                                                                  0,
                                                                  Boundary_Left_Side(),
                                                                  boundary_values);
                        
                        
                        VectorTools::interpolate_boundary_values (dof_handler,
                                                                  1,
                                                                  Boundary_Right_Side(par.K),
                                                                  boundary_values);
                        
                        MatrixTools::apply_boundary_values (boundary_values,
                                                            system_matrix,
                                                            solution,
                                                            system_rhs);
                }
#ifdef __VERBOSE__
                cout<<"rhs:\n";
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
        cout<<"x=[";
        for (int i=0; i<dof_handler.n_dofs(); ++i) {
                cout<<solution[i]<<"; ";
        }
        cout<<"]";
#endif
}


int main(){
        
        Parametri par;
        par.T=1.;
        par.K=100;
        par.S0=100;
        par.r=0.03;
        par.sigma=0.2;
        
        Opzione x(par);
        x.run();
        
        return 0;
}