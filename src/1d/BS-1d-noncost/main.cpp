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

#include <deal.II/numerics/fe_field_function.h>

#include <fstream>
#include <iostream>

#include <vector>
#include <ctime>
#include <algorithm>

#include <climits>

#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/precondition.h>
#include <boost/graph/graph_concepts.hpp>

#include <cmath>
#include <algorithm>

#include <omp.h>

# include <cstdlib>
# include <cmath>
# include <iostream>
# include <fstream>
# include <iomanip>
# include <ctime>
# include <string>

using namespace std;
using namespace dealii;

#define __MATLAB__

const double toll=1e-8;
const double eps=std::numeric_limits<double>::epsilon();

class Parametri{
public:
	//Dati
	double T;                                                  // Scadenza
	double K;                                                  // Strike price
	double S0;                                                 // Spot price
	double r;                                                  // Tasso risk free
        
	// Parametri della parte continua
	double sigma;                                              // Volatilità
        
	// Parametri della parte salto
	double p;                                                  // Parametro 1 Kou
	double lambda;                                             // Parametro 2 Kou
	double lambda_piu;                                         // Parametro 3 Kou
	double lambda_meno;                                        // Parametro 4 Kou
        
	Parametri()=default;
	Parametri(const Parametri &)=default;
};




template<int dim>
class PayOff : public Function<dim>
{
public:
	PayOff (double K_) : Function<dim>(), K(K_){};
        
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
	return max(p(0)-K,0.);
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
	return p[0]-_K*exp(-_r*(_T-this->get_time()));
        
}


template<int dim>
class Opzione{
private:
	Parametri par;
	void make_grid();
	void setup_system ();
	void assemble_system ();
	void solve ();
	void output_results () const {};
	
	Triangulation<dim>              triangulation;
	FE_Q<dim>                       fe;
	DoFHandler<dim>                 dof_handler;
	
	SparsityPattern                 sparsity_pattern;
	SparseMatrix<double>            system_matrix;
	SparseMatrix<double>            system_M2;
	SparseMatrix<double>            dd_matrix;
	SparseMatrix<double>            fd_matrix;
	SparseMatrix<double>            ff_matrix;
	
	Vector<double>                  solution;
	Vector<double>                  system_rhs;
	
	unsigned int refs, Nsteps;
	double time_step;
	
	Point<dim> Smin, Smax;
	double price;
	
	bool ran;
	
public:
	Opzione(Parametri const &par_, int Nsteps_,  int refinement):
	par(par_),
	fe (1),
	dof_handler (triangulation),
	refs(refinement), 
	Nsteps(Nsteps_), 
	time_step (par.T/double(Nsteps_)),
	price(0),  
	ran(false)
	{};
        
	double get_price();
        
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
        
	Smin[0]=par.S0*exp((par.r-par.sigma*par.sigma/2)*par.T
                           -par.sigma*sqrt(par.T)*6);
	Smax[0]=par.S0*exp((par.r-par.sigma*par.sigma/2)*par.T
                           +par.sigma*sqrt(par.T)*6);
        
	cout<< "Smin= "<< Smin<< "\t e Smax= "<< Smax<< endl;
	GridGenerator::subdivided_hyper_cube(triangulation,pow(2,refs)+3, Smin[0],Smax[0]);
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
                                if (std::fabs(cell->face(face)->center()(0) - (Smax[0])) < toll)
                                        cell->face(face)->set_boundary_indicator (1);
        
	cout<< "Controlling Boundary indicators\n";
	vector<types::boundary_id> info;
	info=triangulation.get_boundary_indicators();
	cout<< "Number of Boundaries: " << info.size()<< endl;
	cout<< "which are"<< endl;
	for (unsigned int i=0; i<info.size();++i)
                cout<< info[i] << endl;
        
	solution.reinit(dof_handler.n_dofs());
	system_rhs.reinit(dof_handler.n_dofs());
        
        
}




template<int dim>
void Opzione<dim>::assemble_system() {
        
        
        
	QGauss<dim> quadrature_formula(2);
	FEValues<dim> fe_values (fe, quadrature_formula, update_values   | update_gradients |
                                 update_JxW_values | update_quadrature_points);
        
	const unsigned int   dofs_per_cell = fe.dofs_per_cell;
	const unsigned int   n_q_points    = quadrature_formula.size();
        
	cout<< "Assembling System\n";
	cout<< "Degrees of freedom per cell: "<< dofs_per_cell<< endl;
	cout<< "Quadrature points per cell: "<< n_q_points<< endl;
        
	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        
        // 	FullMatrix<double> cell_dd(dofs_per_cell);
        // 	FullMatrix<double> cell_fd(dofs_per_cell);
	FullMatrix<double> cell_ff(dofs_per_cell);
	FullMatrix<double> cell_mat(dofs_per_cell);
        
	typename DoFHandler<dim>::active_cell_iterator
	cell=dof_handler.begin_active(),
	endc=dof_handler.end();
	Tensor< 1 , dim, double > trasp;
	Tensor<2, dim,  double > sig_mat;
	// 	Tensor< 1 , dim, double > increasing;
	vector<Point<dim> > quad_points(n_q_points);
        
	for (; cell !=endc;++cell) {
                fe_values.reinit(cell);
                cell_ff=0;
                cell_mat=0;
                
                quad_points=fe_values.get_quadrature_points();
                
                for (unsigned q_point=0;q_point<n_q_points;++q_point) {
                        
                        trasp[0]=(par.r-par.sigma*par.sigma)*quad_points[q_point][0];
                        sig_mat[0][0]=0.5*par.sigma*par.sigma
                        *quad_points[q_point][0]*quad_points[q_point][0];
                        
                        for (unsigned i=0;i<dofs_per_cell;++i)
                                for (unsigned j=0; j<dofs_per_cell;++j) {
                                        
                                        
                                        cell_mat(i, j)+=fe_values.JxW(q_point)*(
                                                                                (1/time_step+par.r)*fe_values.shape_value(i, q_point)*fe_values.shape_value(j,q_point)
                                                                                +fe_values.shape_grad(i, q_point)*sig_mat*fe_values.shape_grad(j, q_point)
                                                                                -fe_values.shape_value(i, q_point)*trasp*fe_values.shape_grad(j, q_point));
                                        
                                        cell_ff(i, j)+=fe_values.shape_value(i, q_point)*fe_values.shape_value(j, q_point)*fe_values.JxW(q_point);
                                        
                                }
                }
                
                cell->get_dof_indices (local_dof_indices);
                
                for (unsigned int i=0; i<dofs_per_cell;++i)
                        for (unsigned int j=0; j< dofs_per_cell; ++j) {
                                
                                system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_mat(i, j));
                                ff_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_ff(i, j));
                                
                        }
        }
        
	system_M2.add(1/time_step, ff_matrix);
        
}

template<int dim>
void Opzione<dim>::solve() {
        
	VectorTools::interpolate (dof_handler, PayOff<dim>(par.K), solution);
        
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
                
        }
        
	ran=true;
        
#ifdef __MATLAB__
	ofstream print;
	print.open("solution.m");
	vector<Point<dim> > grid_points(triangulation.get_vertices());
	if (print.is_open()) {
                print<<"x=[ ";
                for (int i=0; i<grid_points.size()-1; ++i) {
                        print<<grid_points[i][0]<<"; ";
                }
                print<<grid_points[grid_points.size()-1][0]<<" ];\n";
                
                print<<"sol=[ ";
                for (int i=0; i<solution.size()-1; ++i) {
                        print<<solution(i)<<"; ";
                }
                print<<solution(solution.size()-1)<<" ];\n";
        }
        
	print.close();
#endif
}

template<int dim>
double Opzione<dim>::get_price() {
        
	if (ran==false) {
                this->run();
        }
        
        Point<dim> p(par.S0);
	Functions::FEFieldFunction<dim> fe_function (dof_handler, solution);
	return fe_function.value(p);
}


int main() {
	Parametri par;
	par.T=1.;
	par.K=90;
	par.S0=95;
	par.r=0.0367;
	par.sigma=0.120381;
	
	cout<<"eps "<<eps<<"\n";
        
	Opzione<1> Call(par, 100, 7);
	double Prezzo=Call.run();
	cout<<"Prezzo "<<Prezzo<<"\n";
	
	cout<<"BS non cost\n";
	return 0;
}

