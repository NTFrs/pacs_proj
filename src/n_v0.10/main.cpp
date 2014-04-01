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

#include <cmath>
#include <algorithm>

using namespace std;
using namespace dealii;

#define dim 1

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
	return max(exp(p[0])-K,0.);
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
	return ZeroFunction<dim>();

}

class Boundary_Right_Side: public Function<dim>
{
public:
	Boundary_Right_Side(double K) : Function< dim>(), _K(K) {};

	virtual double value (const Point<dim> &p, const unsigned int component =0) const;
private:
	double _K;
	double T;
};

double Boundary_Right_Side::value(const Point<dim> &p, const unsigned int component) const
{
	Assert (component == 0, ExcInternalError());
	return exp(p[0])-_K*exp(-(T-this->get_time()));

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
	SparseMatrix<double>  dd_matrix;
	SparseMatrix<double> fd_matrix;
	SparseMatrix<double> ff_matrix;

	Vector<double>       solution;
	Vector<double>       system_rhs;

	double time_step;
	unsigned int refs;
	double Smin, Smax, xmin, xmax;

public:
	Opzione(Parametri const &par_, int Nsteps,  int refinement):
	par(par_),
	fe (1),
	dof_handler (triangulation),
	time_step (par.T/(Nsteps+1))
	refs(refinement)
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
					-par.sigma*sqrt(6)*par.T);
	Smin=par.S0*exp((par.r-par.sigma*par.sigma/2)*par.T
					+par.sigma*sqrt(6)*par.T);
					
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
  
  