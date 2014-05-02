#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>

#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>

#include <vector>

#include <deal.II/grid/grid_out.h>

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/precondition.h>


#include <cmath>
#include <algorithm>

// #define __VERBOSE__


using namespace std;
using namespace dealii;

// #define dim 1

class Parametri{
public:
	//Dati
	double T;                                                // Scadenza
	double K;                                                // Strike price
	double S0;                                               // Spot price
	double r;                                                // Tasso risk free
        
	// Parametri della parte continua
	double sigma;                                            // Volatilità
        
	Parametri()=default;
	Parametri(const Parametri &)=default;
};

template<int dim>
class PayOff : public Function<dim>
{
public:
	PayOff (double K_) : Function<dim>(), K(K_) {};
        
	virtual double value (const Point<dim>   &p,
                              const unsigned int  component = 0) const;
private:
	double K;
};

template<int dim>
double PayOff<dim>::value (const Point<dim>  &p,
                           const unsigned int component) const
{
	Assert (component == 0, ExcInternalError());
	return max(exp(p[0])-K,0.);
}

template<int dim>
class Boundary_Left_Side : public Function<dim>
{
public:
	Boundary_Left_Side() : Function< dim>() {};
        
	virtual double value (const Point<dim> &p, const unsigned int component =0) const;
        
};

template<int dim>
double Boundary_Left_Side<dim>::value(const Point<dim> &p, const unsigned int component) const
{
	Assert (component == 0, ExcInternalError());
	return 0;
        
}

template<int dim>
class Boundary_Right_Side: public Function<dim>
{
public:
	Boundary_Right_Side(double K, double T,  double r) : Function< dim>(), _K(K), _T(T), _r(r) {};
        
	virtual double value (const Point<dim> &p, const unsigned int component =0) const;
private:
	double _K;
	double _T;
	double _r;
};

template<int dim>
double Boundary_Right_Side<dim>::value(const Point<dim> &p, const unsigned int component) const
{
	Assert (component == 0, ExcInternalError());
	/*
         cout<< "time in function: "<< this->get_time()<< endl;
         cout<< "T= "<< _T<< endl;
         double discount(exp(-_r*(_T-this->get_time())));
         cout<< "Discount is : "<< discount<< endl;
         */
	return exp(p[0])-_K*exp(-_r*(_T-this->get_time()));
        
}

template<int dim>
class Opzione{
private:
	Parametri par;
	void make_grid();
	void setup_system ();
	void assemble_system ();
	void solve ();
	double get_price();
	void output_results () const {};
        
	Triangulation<dim>   triangulation;
	FE_Q<dim>            fe;
	DoFHandler<dim>      dof_handler;
        
	SparsityPattern      sparsity_pattern;
	SparseMatrix<double> system_matrix;
	SparseMatrix<double> system_M2;
	SparseMatrix<double> dd_matrix;
	SparseMatrix<double> fd_matrix;
	SparseMatrix<double> ff_matrix;
        
	Vector<double>       solution;
	Vector<double>       system_rhs;
        
	
	unsigned int refs, Nsteps;
	double time_step;
	double Smin, Smax, xmin, xmax;
        
public:
	Opzione(Parametri const &par_, int Nsteps_,  int refinement):
	par(par_),
	fe (1),
	dof_handler (triangulation),
	refs(refinement), 
	Nsteps(Nsteps_), 
	time_step (par.T/double(Nsteps_))
	{};
        
	double run(){
                make_grid();
                setup_system();
                assemble_system();
                solve();
                return get_price();
                
        };
};


template<int dim>
void Opzione<dim>::make_grid() {
	//simple mesh generation
	
	Smin=par.S0*exp((par.r-par.sigma*par.sigma/2)*par.T
                        -par.sigma*sqrt(par.T)*6);
	Smax=par.S0*exp((par.r-par.sigma*par.sigma/2)*par.T
                        +par.sigma*sqrt(par.T)*6);
        
	cout<< "Smin= "<< Smin<< "\t e Smax= "<< Smax<< endl;
	xmin=log(Smin);xmax=log(Smax);
        
	GridGenerator::hyper_cube(triangulation,xmin,xmax);
	triangulation.refine_global(refs);
        
	std::cout << "   Number of active cells: "
	<< triangulation.n_active_cells()
	<< std::endl
	<< "   Total number of cells: "
	<< triangulation.n_cells()
	<< std::endl;
}

template<int dim>
void Opzione<dim>::setup_system() {
	
	dof_handler.distribute_dofs(fe);
        
	std::cout << "   Number of degrees of freedom: "
	<< dof_handler.n_dofs()
	<< std::endl;
        
	CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern (dof_handler, c_sparsity);
        
	sparsity_pattern.copy_from(c_sparsity);
	
	dd_matrix.reinit(sparsity_pattern);
	fd_matrix.reinit(sparsity_pattern);
	ff_matrix.reinit(sparsity_pattern);
	system_matrix.reinit(sparsity_pattern);
	system_M2.reinit(sparsity_pattern);
	
	typename Triangulation<dim>::cell_iterator
	cell = triangulation.begin (),
	endc = triangulation.end();
	for (; cell!=endc; ++cell)
		for (unsigned int face=0;
                     face<GeometryInfo<dim>::faces_per_cell;++face)
			if (cell->face(face)->at_boundary())
                                if (std::fabs(cell->face(face)->center()(0) - (xmax)) < 1e-8)
                                        cell->face(face)->set_boundary_indicator (1);
	
	cout<< "Controlling Boundary indicators\n";
	vector<types::boundary_id> info;
	info=triangulation.get_boundary_indicators();
	cout<< "Number of Boundaries: " << info.size()<< endl;
	cout<< "wich are"<< endl;
	for (unsigned int i=0; i<info.size();++i)
                cout<< info[i] << endl;
        
	solution.reinit(dof_handler.n_dofs());
	system_rhs.reinit(dof_handler.n_dofs());
}

template<int dim>
void Opzione<dim>::assemble_system() {
	
	QGauss<dim> quadrature_formula(2);
	FEValues<dim> fe_values (fe, quadrature_formula, update_values   | update_gradients |
                                 update_JxW_values);
        
	const unsigned int   dofs_per_cell = fe.dofs_per_cell;
	const unsigned int   n_q_points    = quadrature_formula.size();
        
	cout<< "Assembling System\n";
	cout<< "Degrees of freedom per cell: "<< dofs_per_cell<< endl;
	cout<< "Quadrature points per cell: "<< n_q_points<< endl;
        
	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        
	FullMatrix<double> cell_dd(dofs_per_cell);
	FullMatrix<double> cell_fd(dofs_per_cell);
	FullMatrix<double> cell_ff(dofs_per_cell);
        
	typename DoFHandler<dim>::active_cell_iterator
	cell=dof_handler.begin_active(),
	endc=dof_handler.end();
	Tensor< 1 , dim, double > ones;
	// 	Tensor< 1 , dim, double > increasing;
        
	for (unsigned i=0;i<dim;++i)
                ones[i]=1;
        
	for (; cell !=endc;++cell) {
                fe_values.reinit(cell);
                cell_dd=0;
                cell_fd=0;
                cell_ff=0;
                for (unsigned q_point=0;q_point<n_q_points;++q_point)
                        for (unsigned i=0;i<dofs_per_cell;++i)
                                for (unsigned j=0; j<dofs_per_cell;++j) {
                                        
                                        // 	  cout<<fe_values.JxW(q_point)<< " è il JxW quà\n";
                                        // 	  cout<< fe_values.shape_grad(i, q_point)<< " e "<< fe_values.shape_grad(j, q_point)<< " gradiente\n";
                                        // 	  cout<< fe_values.shape_value(i, q_point)<< " e " << fe_values.shape_value(j, q_point)<< " funzione\n";
                                        
                                        //a lot of time lost on this: it's important to summ all q_points (obvious,  but we forgot to use +=)
                                        cell_dd(i, j)+=fe_values.shape_grad(i, q_point)*fe_values.shape_grad(j, q_point)*fe_values.JxW(q_point);
                                        cell_fd(i, j)+=fe_values.shape_value(i, q_point)*(ones*fe_values.shape_grad(j,q_point))*fe_values.JxW(q_point);
                                        cell_ff(i, j)+=fe_values.shape_value(i, q_point)*fe_values.shape_value(j, q_point)*fe_values.JxW(q_point);
                                        
                                }
                
                cell->get_dof_indices (local_dof_indices);
                
                for (unsigned int i=0; i<dofs_per_cell;++i)
                        for (unsigned int j=0; j< dofs_per_cell; ++j) {
                                
                                dd_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_dd(i, j));
                                fd_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_fd(i, j));
                                ff_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_ff(i, j));
                                
                        }
                
        }
        
        system_M2.add(1, ff_matrix);
        system_matrix.add(1, ff_matrix);
        system_matrix.add(par.sigma*par.sigma*time_step/2, dd_matrix);
        system_matrix.add(-time_step*(par.r-par.sigma*par.sigma/2), fd_matrix);
        system_matrix.add(par.r*time_step, ff_matrix);
	
}

template<int dim>
void Opzione<dim>::solve() {
	
	cout<< "xmin e xmx\n";
	cout<<xmin<<"\t"<<xmax<<"\n";
	
	VectorTools::interpolate (dof_handler, PayOff<dim>(par.K),solution);
	cout<<"solution:\n";
	solution.print(cout);
	cout<<"\n";
	unsigned int Step=Nsteps;
	
	Boundary_Right_Side<dim> right_bound(par.K, par.T, par.r);
	cout<< "time step is"<< time_step<< endl;
	for (double time=par.T-time_step;time >=0;time-=time_step, --Step) {
                cout<< "Step "<< Step<<"\t at time \t"<< time<< endl;
                system_M2.vmult(system_rhs, solution);
                right_bound.set_time(time);
                
                {
                        
                        std::map<types::global_dof_index,double> boundary_values;
                        VectorTools::interpolate_boundary_values (dof_handler,
                                                                  0,
                                                                  Boundary_Left_Side<dim>(),
                                                                  boundary_values);
                        
                        
                        VectorTools::interpolate_boundary_values (dof_handler,
                                                                  1,
                                                                  right_bound,
                                                                  boundary_values);
                        
                        MatrixTools::apply_boundary_values (boundary_values,
                                                            system_matrix,
                                                            solution,
                                                            system_rhs, false);
                        
                }
                
                SparseDirectUMFPACK solver;
                solver.initialize(sparsity_pattern);
                solver.factorize(system_matrix);
                solver.solve(system_rhs);
                
                solution=system_rhs;
                
#ifdef __VERBOSE__
                cout<<"solution:\n";
                solution.print(cout);
                cout<<"\n";
# endif
                
                
                
	}
	
        
	cout<<"solution:\n";
	solution.print(cout);
	cout<<"\n";
}

template<int dim>
double Opzione<dim>::get_price() {
	
	return 0;
}


int main() {
	Parametri par;
	par.T=1.;
	par.K=100;
	par.S0=100;
	par.r=0.03;
	par.sigma=0.2;
	
	Opzione<1> Call(par, 100, 10);
	Call.run();
	
	return 0;
}