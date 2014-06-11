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

#include <deal.II/grid/grid_refinement.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/numerics/derivative_approximation.h>
#include <deal.II/numerics/solution_transfer.h>

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

const double eps=std::numeric_limits<double>::epsilon();
const double toll=eps;

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
class Kou_Density: public Function<dim>
{
public:
	Kou_Density(double p,  double lam, double lam_u,  double lam_d) : Function<dim>(),  _p(p),  _lam(lam), 
	_lam_u(lam_u),  _lam_d(lam_d) {};
	
        virtual double value (const Point<dim> &p,  const unsigned int component=0) const;
        virtual void value_list(const std::vector<Point<dim> > &points,
                                std::vector<double> &values,
                                const unsigned int component = 0) const;
private:
        double _p;
        double _lam;
        double _lam_u,  _lam_d;
};

template<int dim>
double Kou_Density<dim>::value(const Point<dim> &p,  const unsigned int component) const
{
	Assert (component == 0, ExcInternalError());
	if (p[0]>0)
                return _p*_lam*_lam_u*exp(-_lam_u*p[0]);
	else
                return (1-_p)*_lam*_lam_d*exp(_lam_d*p[0]);
        
}



template<int dim>
void Kou_Density<dim>::value_list(const std::vector<Point<dim> > &points, std::vector<double> &values, const unsigned int component) const
{
	Assert (values.size() == points.size(),
                ExcDimensionMismatch (values.size(), points.size()));
	Assert (component == 0, ExcInternalError());
        
	const unsigned int n_points=points.size();
        
	for (unsigned int i=0;i<n_points;++i)
                if (points[i][0]>0)
                        values[i]=_p*_lam*_lam_u*exp(-_lam_u*points[i][0]);
                else
                        values[i]=(1-_p)*_lam*_lam_d*exp(_lam_d*points[i][0]);
}

template<int dim>
class Kou_Density_logged: public Function<dim>
{
public:
	Kou_Density_logged(double p,  double lam, double lam_u,  double lam_d) : Function<dim>(),  _p(p),  _lam(lam), 
	_lam_u(lam_u),  _lam_d(lam_d) {};
	virtual double value (const Point<dim> &p,  const unsigned int component=0) const;
	virtual void value_list(const std::vector<Point<dim> > &points,
                                std::vector<double> &values,
                                const unsigned int component = 0) const;
private:
	double _p;
	double _lam;
	double _lam_u,  _lam_d;
};

template<int dim>
double Kou_Density_logged<dim>::value(const Point<dim> &p,  const unsigned int component) const
{
	Assert (component == 0, ExcInternalError());
	if (p[0]>0)
                return _p*_lam*_lam_u*(p[0], -_lam_u);
	else
                return (1-_p)*_lam*_lam_d*(p[0], _lam_d);
        
}



template<int dim>
void Kou_Density_logged<dim>::value_list(const std::vector<Point<dim> > &points, std::vector<double> &values, const unsigned int component) const
{
	Assert (values.size() == points.size(),
                ExcDimensionMismatch (values.size(), points.size()));
	Assert (component == 0, ExcInternalError());
        
	const unsigned int n_points=points.size();
        
	for (unsigned int i=0;i<n_points;++i)
                if (points[i][0]>0)
                        values[i]=_p*_lam*_lam_u*pow(points[i][0], -_lam_u);
                else
                        values[i]=(1-_p)*_lam*_lam_d*pow(points[i][0], +_lam_d);
}




template<int dim>
class Opzione{
private:
	Parametri par;
	void make_grid();
	void setup_system ();
	void assemble_system ();
	void solve_one_step(double time);
	void estimate_doubling(double time, Vector< float >& errors);
	void refine_grid();
	void output_results () const {};
	
	Kou_Density<dim>				k;
        // 	Kou_Density_logged<dim>			k_log;
	
	Triangulation<dim>              triangulation;
	FE_Q<dim>                       fe;
	DoFHandler<dim>                 dof_handler;
	
	SparsityPattern                 sparsity_pattern;
	SparseMatrix<double>            system_matrix;
	SparseMatrix<double>            system_M2;
	SparseMatrix<double>            dd_matrix;
	SparseMatrix<double>            fd_matrix;
	SparseMatrix<double>            ff_matrix;
	
	ConstraintMatrix				constraints;
	
	Vector<double>                  solution;
	Vector<double>                  system_rhs;
	
	std::vector< Point<dim> >       grid_points;
	
	unsigned int refs, Nsteps;
	double time_step;
	
	Point<dim> Smin, Smax;
	double price;
	double alpha;
	
	void Levy_integral_part1();
	void Levy_integral_part2(Vector<double> &J);
	
	bool ran;
	
public:
	Opzione(Parametri const &par_, int Nsteps_,  int refinement):
	par(par_),
	k(par.p, par.lambda, par.lambda_piu, par.lambda_meno),
        // 	k_log(par.p, par.lambda, par.lambda_piu, par.lambda_meno), 
	fe (1),
	dof_handler (triangulation),
	refs(refinement), 
	Nsteps(Nsteps_), 
	time_step (par.T/double(Nsteps_)),
	price(0),  
	ran(false)
	{};
        
	double get_price();
        
	double run();
};


template<int dim>
void Opzione<dim>::Levy_integral_part1() {
	
	alpha=0;
	Point<dim> Bmin(0.), Bmax(Smax);
	double step(0.5);
	
	while (k.value(Bmin)>toll)
                Bmin[0]+=-step;
	
	while (k.value(Bmax)>toll)
                Bmin[0]+=step;
        
	Triangulation<dim> integral_grid;
	FE_Q<dim> fe2(1);
	DoFHandler<dim> dof_handler2(integral_grid);
	
	GridGenerator::hyper_cube<dim>(integral_grid, Bmin[0], Bmax[0]);
	integral_grid.refine_global(15);
	
	
	dof_handler2.distribute_dofs(fe2);
	
	QGauss<dim> quadrature_formula(8);
	FEValues<dim> fe_values(fe2, quadrature_formula,  update_quadrature_points |update_JxW_values);
        
	typename DoFHandler<dim>::active_cell_iterator
	cell=dof_handler2.begin_active(),
	endc=dof_handler2.end();
        
	const unsigned int n_q_points(quadrature_formula.size());
	
	for (; cell !=endc;++cell) {
                fe_values.reinit(cell);
                std::vector< Point<dim> > quad_points(fe_values.get_quadrature_points());
                for (unsigned q_point=0;q_point<n_q_points;++q_point)
                        alpha+=fe_values.JxW(q_point)*(exp(quad_points[q_point][0])-1.)*k.value(quad_points[q_point]);
                
	}
	cout<< "alpha is "<< alpha<< endl;
	return;
        
}

template<int dim>
void Opzione<dim>::Levy_integral_part2(Vector<double> &J) {
        
	J.reinit(solution.size());
	
	QGauss<dim> quadrature_formula(5);
	FEValues<dim> fe_values(fe, quadrature_formula,  update_quadrature_points | update_values | update_JxW_values);
	
	const unsigned int n_q_points(quadrature_formula.size());
	typename DoFHandler<dim>::active_cell_iterator endc=dof_handler.end();
	
	vector<double> sol_cell(n_q_points);
	
	vector< Point <dim> > quad_points(n_q_points);
	Point<dim> logz(0.);
        
	std::map<types::global_dof_index, Point<dim> > vertices;
	DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dof_handler, vertices);

	for (unsigned iter=0;iter<solution.size();++iter) {
					Point<dim> actual_vertex=vertices[iter];
							  typename DoFHandler<dim>::active_cell_iterator cell=dof_handler.begin_active();
                                for (; cell!=endc;++cell) {
                                        // 	  cout<< "switching cell\n";
                                        fe_values.reinit(cell);
                                        quad_points=fe_values.get_quadrature_points();
                                        fe_values.get_function_values(solution, sol_cell);                        
                                        for (unsigned q_point=0;q_point<n_q_points;++q_point) {
                                                logz(0)=log(quad_points[q_point](0)/actual_vertex(0));
                                                J[iter]+=fe_values.JxW(q_point)*sol_cell[q_point]*k.value(logz)/quad_points[q_point](0);
                                        }
                                }
                        }
                        
                }
// 	}
        
// }

template<int dim>
void Opzione<dim>::make_grid() {
        
	Smin[0]=0.5*par.S0*exp((par.r-par.sigma*par.sigma/2)*par.T
                               -par.sigma*sqrt(par.T)*6);
	Smax[0]=1.5*par.S0*exp((par.r-par.sigma*par.sigma/2)*par.T
                               +par.sigma*sqrt(par.T)*6);
        
	cout<< "Smin= "<< Smin<< "\t e Smax= "<< Smax<< endl;
	GridGenerator::subdivided_hyper_cube(triangulation,pow(2,refs)+3, Smin[0],Smax[0]);
	
	grid_points=triangulation.get_vertices();/*
                                                  for (unsigned i=0;i<grid_points.size();++i)
                                                  cout<< grid_points[i]<< endl;*/
}

template<int dim>
void Opzione<dim>::setup_system() {
        
	dof_handler.distribute_dofs(fe);
        
        
	std::cout << "   Number of degrees of freedom: "
	<< dof_handler.n_dofs()
	<< std::endl;
        
	constraints.clear();
	DoFTools::make_hanging_node_constraints (dof_handler,
                                                 constraints);
        
	constraints.close();
        
	CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern (dof_handler, c_sparsity, constraints, false);
        
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
                        if (cell->face(face)->at_boundary()) {
                                if (std::fabs(cell->face(face)->center()(0) - (Smin[0])) < toll)
                                        cell->face(face)->set_boundary_indicator (0);
                                if (std::fabs(cell->face(face)->center()(0) - (Smax[0])) < toll)
                                        cell->face(face)->set_boundary_indicator (1);
                        }
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
        
	Levy_integral_part1();
        
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
                        
                        trasp[0]=(par.r-par.sigma*par.sigma-alpha)*quad_points[q_point][0];
                        sig_mat[0][0]=0.5*par.sigma*par.sigma
                        *quad_points[q_point][0]*quad_points[q_point][0];
                        
                        for (unsigned i=0;i<dofs_per_cell;++i)
                                for (unsigned j=0; j<dofs_per_cell;++j) {
                                        
                                        
                                        cell_mat(i, j)+=fe_values.JxW(q_point)*(
                                                                                (1/time_step+par.r+par.lambda)*fe_values.shape_value(i, q_point)*fe_values.shape_value(j,q_point)
                                                                                +fe_values.shape_grad(i, q_point)*sig_mat*fe_values.shape_grad(j, q_point)
                                                                                -fe_values.shape_value(i, q_point)*trasp*fe_values.shape_grad(j, q_point));
                                        
                                        cell_ff(i, j)+=fe_values.shape_value(i, q_point)*fe_values.shape_value(j, q_point)*fe_values.JxW(q_point);
                                        
                                }
                }
                
                cell->get_dof_indices (local_dof_indices);
                
                constraints.distribute_local_to_global(cell_mat, local_dof_indices, system_matrix);
                constraints.distribute_local_to_global(cell_ff, local_dof_indices, ff_matrix);
                
        }
        
	system_M2.add(1/time_step, ff_matrix);
        
}


template<int dim>
void Opzione<dim>::solve_one_step(double time) {
        
	Boundary_Right_Side<dim> right_bound(par.K, par.T, par.r);
        
	Vector<double> J;
	Levy_integral_part2(J);
	
	ff_matrix.vmult(system_rhs, J);
	{
                Vector<double> temp;
                
                temp.reinit(dof_handler.n_dofs());
                system_M2.vmult(temp,solution);
                
                system_rhs+=temp;
        }
        
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
        
	constraints.distribute (solution);
        
}


template<int dim>
double Opzione<dim>::run() {
        
	make_grid();
	setup_system();
	assemble_system();
        
	VectorTools::interpolate (dof_handler, PayOff<dim>(par.K), solution);
        
        cout<<this->solution<<"\n";
        
	{
                DataOut<dim> data_out;
                
                data_out.attach_dof_handler (dof_handler);
                data_out.add_data_vector (solution, "begin");
                
                data_out.build_patches ();
                
                std::ofstream output ("plot/begin.gpl");
                data_out.write_gnuplot (output);
        }
        
	unsigned int Step=Nsteps;
        
	for (double time=par.T-time_step;time >=0;time-=time_step, --Step) {
                cout<< "Step "<< Step<<"\t at time \t"<< time<< endl;
                
                if (!(Step%20) && !(Step==Nsteps)) {
                        refine_grid();
                        solve_one_step(time);
                }
                else
                {
                        solve_one_step(time);
                }
        }
        
	{
                DataOut<dim> data_out;
                
                data_out.attach_dof_handler (dof_handler);
                data_out.add_data_vector (solution, "end");
                
                data_out.build_patches ();
                
                std::ofstream output ("plot/end.gpl");
                data_out.write_gnuplot (output);
        }
        
	ran=true;
        
	return get_price();
}

template <int dim>
void Opzione<dim>::refine_grid (){
        
	Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
	Vector<float> estimated_error_per_cell2 (triangulation.n_active_cells());   
	KellyErrorEstimator<dim>::estimate (dof_handler,
                                            QGauss<dim-1>(3),
                                            typename FunctionMap<dim>::type(),
                                            solution,
                                            estimated_error_per_cell);
    double time=1.;
    estimate_doubling(time, estimated_error_per_cell2);
        
	// 	   GridRefinement::refine_and_coarsen_optimize (triangulation,estimated_error_per_cell);
    auto minimum=*std::min_element(estimated_error_per_cell2.begin(), estimated_error_per_cell2.end());    
	auto maximum=*std::max_element(estimated_error_per_cell2.begin(), estimated_error_per_cell2.end());    
	auto mid=0.5*(maximum+minimum);
	auto third=0.25*minimum+0.75*maximum;
	typename DoFHandler<dim>::active_cell_iterator cell=dof_handler.begin_active(), endc=dof_handler.end();
	
	for (unsigned i=0;cell !=endc;++cell, ++i)
	 if (estimated_error_per_cell2[i]<mid)
	  cell->set_coarsen_flag();
	 else if (estimated_error_per_cell2[i]>third)
	   cell->set_refine_flag();
        
// 	GridRefinement::refine_and_coarsen_fixed_number (triangulation, estimated_error_per_cell, 0.03, 0.1);
        
	SolutionTransfer<dim> solution_trans(dof_handler);
	Vector<double> previous_solution;
	previous_solution = solution;
	triangulation.prepare_coarsening_and_refinement();
	solution_trans.prepare_for_coarsening_and_refinement(previous_solution);
        
	triangulation.execute_coarsening_and_refinement ();
	setup_system ();
        
	solution_trans.interpolate(previous_solution, solution);
	assemble_system();
}

template<int dim>
void Opzione<dim>::estimate_doubling(double time,  Vector<float> & errors) {
	
	Triangulation<dim> old_tria;
	old_tria.copy_triangulation(triangulation);
	FE_Q<dim> old_fe(1);
	DoFHandler<dim> old_dof(old_tria);
	old_dof.distribute_dofs(old_fe);
	Vector<double> old_solution=solution;
	{
	Functions::FEFieldFunction<dim>	moveSol(old_dof,  old_solution);
	
	triangulation.refine_global(1);
	setup_system();
	VectorTools::interpolate(dof_handler, moveSol, solution);
	 }
	assemble_system();
	solve_one_step(time);
	{
	 Functions::FEFieldFunction<dim> moveSol(dof_handler, solution); 
	 cerr<< "dof size "<< dof_handler.n_dofs()<< " solution size "<< solution.size()<< endl;
	 cerr<< "olddof size "<< old_dof.n_dofs()<< " oldsolution size "<< old_solution.size()<< endl;

	 Vector<double> temp(old_dof.n_dofs());
	 cerr<< "this one2\n";
	 VectorTools::interpolate(old_dof, moveSol, temp);
	 cerr<< "this one2\n";
	 solution=temp;
	}
	triangulation.clear();
	triangulation.copy_triangulation(old_tria);
	setup_system();
	assemble_system();
	
	typename DoFHandler<dim>::active_cell_iterator cell=dof_handler.begin_active(),  endc=dof_handler.end();
	vector<types::global_dof_index> local_dof_indices(fe.dofs_per_cell);
	errors.reinit(old_tria.n_active_cells());
	double err(0);
	unsigned ind(0),  count(0);
	for (;cell !=endc;++cell) {
	 err=0;
	 cell->get_dof_indices(local_dof_indices);
	 for (unsigned i=0;i<fe.dofs_per_cell;++i) {
	  ind=local_dof_indices[i];
	  err+=(solution[ind]-old_solution[ind])*(solution[ind]-old_solution[ind]);
	 }
	 errors[count]=(err);
	 count++;
	 }
	 
	solution=old_solution;
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
	
	// Parametri della parte salto
	par.p=0.20761;                                             // Parametro 1 Kou
	par.lambda=0.330966;                                       // Parametro 2 Kou
	par.lambda_piu=9.65997;                                    // Parametro 3 Kou
	par.lambda_meno=3.13868;                                   // Parametro 4 Kou
	
	cout<<"eps "<<eps<<"\n";
        
	Opzione<1> Call(par, 100, 8);
	double Prezzo=Call.run();
	cout<<"Prezzo "<<Prezzo<<"\n";
	
	cout<<"Kou non cost\n";
	return 0;
}

