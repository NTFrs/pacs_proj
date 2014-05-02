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
#include <algorithm>

#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/precondition.h>

#include <cmath>
#include <algorithm>

#include <climits>

#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>

using namespace std;
using namespace dealii;

const double eps=std::numeric_limits<double>::epsilon();
const double toll=1e-8;

#define __CALL__
// max(s1+s2-k,0)

class Parametri2d{
public:
	//Dati
	double T;                                                  // Scadenza
	double K;                                                  // Strike price
	double S01;                                                // Spot price
	double S02;                                                // Spot price
	double r;                                                  // Tasso risk free

	// Parametri della parte continua
	double sigma1;                                             // Volatilità
	double sigma2;                                          // Volatilità
	double ro;                                              // Volatilità

	// Parametri della parte salto
	double p1;                                              // Parametro 1 Kou
	double lambda1;                                         // Parametro 2 Kou
	double lambda_piu_1;                                    // Parametro 3 Kou
	double lambda_meno_1;                                   // Parametro 4 Kou

	double p2;                                                 // Parametro 1 Kou
	double lambda2;                                            // Parametro 2 Kou
	double lambda_piu_2;                                       // Parametro 3 Kou
	double lambda_meno_2;                                      // Parametro 4 Kou


	Parametri2d()=default;
	Parametri2d(const Parametri2d &)=default;
  };

template<int dim>
class Boundary_Condition: public Function<dim>
{
public:
	Boundary_Condition(double S01, double S02, double K, double T,  double r) : Function< dim>(),
	_S01(S01), _S02(S02), _K(K), _T(T), _r(r) {};

	virtual double value (const Point<dim> &p, const unsigned int component =0) const;
private:
	double _S01;
	double _S02;
	double _K;
	double _T;
	double _r;
};

template<int dim>
double Boundary_Condition<dim>::value(const Point<dim> &p, const unsigned int component) const
{
	Assert (component == 0, ExcInternalError());
#ifdef __CALL__
	return max(_S01*exp(p[0])+_S02*exp(p[1])-_K*exp(-_r*(_T-this->get_time())),0.);
#else
	return max(_K*exp(-_r*(_T-this->get_time()))-(_S01*exp(p[0])+_S02*exp(p[1])),0.);
#endif
}

template<int dim>
class PayOff : public Function<dim>
{
public:
	PayOff (double K_, double S01_, double S02_) : Function<dim>(), K(K_), S01(S01_), S02(S02_) {};

	virtual double value (const Point<dim>   &p,
	 const unsigned int  component = 0) const;
private:
	double K;
	double S01;
	double S02;
};

template<int dim>
double PayOff<dim>::value (const Point<dim>  &p,
	const unsigned int component) const
{
	Assert (component == 0, ExcInternalError());
#ifdef __CALL__
	return max(S01*exp(p(0))+S02*exp(p(1))-K,0.);
#else
	return max(K-(S01*exp(p(0))+S02*exp(p(1))),0.);
#endif
}


template<int dim>
class Kou_Density: public Function<dim>
{

public:
	Kou_Density(unsigned int ax,  double p,  double lam, double lam_u,  double lam_d) : Function<dim>(),  _ax(ax),  _p(p),  _lam(lam), 
	_lam_u(lam_u),  _lam_d(lam_d) {};

	virtual double value (const Point<dim> &p,  const unsigned int component=0) const;
	virtual void value_list(const std::vector<Point<dim> > &points,
	 std::vector<double> &values,
	 const unsigned int component = 0) const;
private:
	unsigned int _ax;
	double _p;
	double _lam;
	double _lam_u,  _lam_d;
};

template<int dim>
double Kou_Density<dim>::value(const Point<dim> &p,  const unsigned int component) const
{
	Assert (component == 0, ExcInternalError());
	if (p[_ax]>0)
	return _p*_lam*_lam_u*exp(-_lam_u*p[_ax]);
	else
	return (1-_p)*_lam*_lam_d*exp(_lam_d*p[_ax]);

}

template<int dim>
void Kou_Density<dim>::value_list(const std::vector<Point<dim> > &points, std::vector<double> &values, const unsigned int component) const
{
	Assert (values.size() == points.size(),
	 ExcDimensionMismatch (values.size(), points.size()));
	Assert (component == 0, ExcInternalError());

	const unsigned int n_points=points.size();

	for (unsigned int i=0;i<n_points;++i)
	if (points[i][_ax]>0)
	values[i]=_p*_lam*_lam_u*exp(-_lam_u*points[i][_ax]);
	else
	values[i]=(1-_p)*_lam*_lam_d*exp(_lam_d*points[i][_ax]);
}

template<int dim>
class Solution_Trimmer: public Function<dim>
{
private:
	//check that this causes no memory leaks while keeping hereditariety
	unsigned int _ax;
	Function<dim> * _left;  
	Function<dim> * _right;
	DoFHandler<dim> const & _dof;
	Vector<double> const & _sol;
	Point<dim> _l_lim, _r_lim;
	Functions::FEFieldFunction<dim> _fe_func;

public:
	Solution_Trimmer(unsigned int ax, Function<dim> * left,  Function<dim> * right, DoFHandler<dim> const & dof, Vector<double> const & sol,  Point<dim> const & xmin, Point<dim> const & xmax): _ax(ax), _left(left),  _right(right),  _dof(dof), _sol(sol), _l_lim(xmin), _r_lim(xmax) , _fe_func(_dof, _sol){};

	virtual double value(const Point<dim> &p,  const unsigned int component=0) const;
	virtual void value_list(const std::vector<Point<dim> > &points,
	 std::vector<double> &values,
	 const unsigned int component = 0) const;
};

template<int dim>
double Solution_Trimmer<dim>::value(const Point<dim> &p,  const unsigned int component) const
{
	Assert (component == 0, ExcInternalError());

	if (p[_ax]<_l_lim[_ax])
	return _left->value(p);
	if (p[_ax]>_r_lim[_ax])
	return _right->value(p);
	return _fe_func.value(p);  

}

template<int dim>
void Solution_Trimmer<dim>::value_list(const std::vector<Point<dim> > &points, std::vector<double> &values, const unsigned int component) const
{
	Assert (values.size() == points.size(),
	 ExcDimensionMismatch (values.size(), points.size()));
	Assert (component == 0, ExcInternalError());

	const unsigned int n_points=points.size();

	for (unsigned int i=0;i<n_points;++i)
	{
	 if (points[i][_ax]<_l_lim[_ax])
	 values[i]=_left->value(points[i]);
	 else if (points[i][_ax]>_r_lim[_ax])
	 values[i]=_right->value(points[i]);
	 else
	 values[i]=_fe_func.value(points[i]);
 }
}



template<int dim>
class Opzione{
private:
	Parametri2d par;
	void make_grid();
	void setup_system () ;
	void assemble_system () ;
	void solve () ;
	void output_results () const {};

	Kou_Density<dim>				k_x;
	Kou_Density<dim>				k_y;

	Triangulation<dim>              triangulation;
	FE_Q<dim>                       fe;
	DoFHandler<dim>                 dof_handler;

	Triangulation<1>				integral_triangulation_x;
	FE_Q<1>							fe_integral_x;
	DoFHandler<1>					dof_handler_integral_x;

	Triangulation<1>				integral_triangulation_y;
	FE_Q<1>							fe_integral_y;
	DoFHandler<1>					dof_handler_integral_y;
	
	SparsityPattern                 sparsity_pattern;
	SparseMatrix<double>            system_matrix;
	SparseMatrix<double>            system_M2;
	SparseMatrix<double>            dd_matrix;
	SparseMatrix<double>            fd_matrix;
	SparseMatrix<double>            ff_matrix;

	

	Vector<double>                  solution;
	Vector<double>                  system_rhs;
	
	std::vector< Point<dim> >       grid_points;
	
	unsigned int refs, Nsteps;
	double time_step;
	double dx1, dx2;
	double Smin1, Smax1, Smin2, Smax2;
	double price;
	Point<dim> xmin, xmax;
	Point<dim> Bmin, Bmax;
	
	double alpha1,  alpha2;
	
	bool ran;
	
	void Levy_integral_part1();
	void Levy_integral_part2(Vector<double> &J);
	
public:
	Opzione(Parametri2d const &par_, int Nsteps_,  int refinement):
	par(par_),
	k_x(0, par_.p1, par_.lambda1,  par_.lambda_piu_1,  par_.lambda_meno_1),
	k_x(1, par_.p2, par_.lambda2,  par_.lambda_piu_2,  par_.lambda_meno_2),
	fe (1),
	dof_handler (triangulation),
	fe_integral_x(1), 
	dof_handler_integral_x(integral_triangulation_x), 
	fe_integral_y(1), 
	dof_handler_integral_y(integral_triangulation_y), 
	refs(refinement), 
	Nsteps(Nsteps_), 
	time_step (par.T/double(Nsteps_)),
	price(0), 
	ran(false)
	{};

	double get_price() ;

	double run(){
	 make_grid();
	 setup_system();
	 assemble_system();
	 solve();
	 output_results();
	 return get_price();

   };
  };

  
  
template<int dim>
void Opzione<dim>::Levy_integral_part1(){
	
	alpha1=0;
	alpha2=0;
	
	{
	 QGauss<dim> quadrature_formula2(7);
	 FEValues<dim> fe_values2 (fe_integral_x, quadrature_formula2, update_values | update_quadrature_points | update_JxW_values);

	 typename DoFHandler<dim>::active_cell_iterator
	 cell=dof_handler_integral_x .begin_active(),
	 endc=dof_handler_integral_x.end();

	 // 	const unsigned int   dofs_per_cell = fe2.dofs_per_cell;
	 const unsigned int   n_q_points    = quadrature_formula2.size();

	 //      const Coefficient<dim> coefficient;


	 for (; cell !=endc;++cell) {

	  fe_values2.reinit(cell);
	  std::vector< Point<dim> >    quad_points (fe_values2.get_quadrature_points());
	  for (unsigned q_point=0;q_point<n_q_points;++q_point) {
	   // 	  cerr<< "At point" << 
	   alpha1+=fe_values2.JxW(q_point)*(exp(quad_points[q_point][0])-1)*k_x.value(quad_points[q_point]);
	 }
	}
	}
	
	{
	 QGauss<dim> quadrature_formula2(7);
	 FEValues<dim> fe_values2 (fe_integral_y, quadrature_formula2, update_values | update_quadrature_points | update_JxW_values);

	 typename DoFHandler<dim>::active_cell_iterator
	 cell=dof_handler_integral_y .begin_active(),
	 endc=dof_handler_integral_y.end();

	 // 	const unsigned int   dofs_per_cell = fe2.dofs_per_cell;
	 const unsigned int   n_q_points    = quadrature_formula2.size();

	 //      const Coefficient<dim> coefficient;


	 for (; cell !=endc;++cell) {

	  fe_values2.reinit(cell);
	  std::vector< Point<dim> >    quad_points (fe_values2.get_quadrature_points());
	  for (unsigned q_point=0;q_point<n_q_points;++q_point) {
	   // 	  cerr<< "At point" << 
	   alpha2+=fe_values2.JxW(q_point)*(exp(quad_points[q_point][1])-1)*k_y.value(quad_points[q_point]);
	 }
	}
 }
	
}

template<int dim>
void Opzione<dim>::Levy_integral_part2(Vector<double> &J_x, Vector<double> &J_y) {
	
	J_x.reinit(solution.size());                                 // If fast is false, the vector is filled by zeros
	J_y.reinit(solution.size());                                 

	unsigned int N(grid_points.size());
	{
	QGauss<dim> quadrature_formula2(7);
	FEValues<dim> fe_values2 (fe_integral_x, quadrature_formula2, update_values | update_quadrature_points | update_JxW_values);

	 const unsigned int   n_q_points    = quadrature_formula2.size();
	
	 Boundary_Condition<dim> bc(par.S01, par.S02, par.K, par.T, par.r);
	 
	 Solution_Trimmer<dim> func(0, &bc, &bc, dof_handler, solution, xmin, xmax);
 
	 for (unsigned int it=0;it<N;++it)
	 {
	  typename DoFHandler<dim>::active_cell_iterator
	  cell=dof_handler_integral_x.begin_active(),
	  endc=dof_handler_integral_x.end();
 
	  for (; cell !=endc;++cell) {
 
		fe_values2.reinit(cell);
	   
		std::vector< Point<dim> > quad_points(fe_values2.get_quadrature_points());
	   
		std::vector<double> kern(n_q_points),  f_u(n_q_points);
 
		k_x.value_list(quad_points, kern);
		
	   for (unsigned int q_point=0;q_point<n_q_points;++q_point)
	   quad_points[q_point]+=grid_points[it];
	   
	   func.value_list(quad_points, f_u);
	   
	   for (unsigned q_point=0;q_point<n_q_points;++q_point)  J_x(it)+=fe_values2.JxW(q_point)*kern[q_point]*f_u[q_point];
	   
	 }
	}
	}

	{
	 QGauss<dim> quadrature_formula2(7);
	 FEValues<dim> fe_values2 (fe_integral_y, quadrature_formula2, update_values | update_quadrature_points | update_JxW_values);

	 const unsigned int   n_q_points    = quadrature_formula2.size();

	 Boundary_Condition<dim> bc(par.S01, par.S02, par.K, par.T, par.r);

	 Solution_Trimmer<dim> func(1, &bc, &bc, dof_handler, solution, xmin, xmax);

	 for (unsigned int it=0;it<N;++it)
	 {
	  typename DoFHandler<dim>::active_cell_iterator
	  cell=dof_handler_integral_y.begin_active(),
	  endc=dof_handler_integral_y.end();

	  for (; cell !=endc;++cell) {

	   fe_values2.reinit(cell);

	   std::vector< Point<dim> > quad_points(fe_values2.get_quadrature_points());

	   std::vector<double> kern(n_q_points),  f_u(n_q_points);

	   k_y.value_list(quad_points, kern);

	   for (unsigned int q_point=0;q_point<n_q_points;++q_point)
	   quad_points[q_point]+=grid_points[it];

	   func.value_list(quad_points, f_u);

	   for (unsigned q_point=0;q_point<n_q_points;++q_point)  J_y(it)+=fe_values2.JxW(q_point)*kern[q_point]*f_u[q_point];

	 }
  }
 }
	
}
  
  
  
template<int dim>
void Opzione<dim>::make_grid() {

	Smin1=par.S01*exp((par.r-par.sigma1*par.sigma1/2)*par.T
	 -par.sigma1*sqrt(par.T)*6);
	Smax1=par.S01*exp((par.r-par.sigma1*par.sigma1/2)*par.T
	 +par.sigma1*sqrt(par.T)*6);
	Smin2=par.S02*exp((par.r-par.sigma2*par.sigma2/2)*par.T
	 -par.sigma2*sqrt(par.T)*6);
	Smax2=par.S02*exp((par.r-par.sigma2*par.sigma2/2)*par.T
	 +par.sigma2*sqrt(par.T)*6);

	dx1=(log(Smax1/par.S01)-log(Smin1/par.S01))/pow(2., refs);
	dx2=(log(Smax2/par.S02)-log(Smin2/par.S02))/pow(2., refs);
	 
	xmin[0]=0;
	xmax[0]=0;
	xmin[1]=0;
	xmax[1]=0;

	while (xmin[0]>log(Smin1/par.S01))
	xmin[0]-=dx1;
	while (xmax[0]<log(Smax1/par.S01))
	xmax[0]+=dx1;
	
	xmin[0]-=dx1;
	xmax[0]+=dx1;
	
	while (xmin[1]>log(Smin2/par.S02))
	xmin[1]-=dx2;
	while (xmax[1]<log(Smax2/par.S02))
	xmax[1]+=dx2;
	
	xmin[1]-=dx2;
	xmax[1]+=dx2;
	
	cout<<"dx1 "<<dx1<<"\n";
	cout<<"dx2 "<<dx2<<"\n";
	
	cout<< "Smin1= "<< Smin1<< "\t e Smax1= "<< Smax1<< endl;
	cout<< "Smin2= "<< Smin2<< "\t e Smax2= "<< Smax2<< endl;
	cout<< "xmin= "<< xmin<< endl;
	cout<< "xmax= "<< xmax<< endl;

	Bmin=xmin;
	Bmax=xmax;
	
	while(k_x.value(Bmin)>toll)
	Bmin[0]-=dx1;

	while(k_x.value(Bmax)>toll)
	Bmax[0]+=dx1;
	
	while(k_y.value(Bmin)>toll)
	Bmin[1]-=dx2;

	while(k_y.value(Bmax)>toll)
	Bmax[1]+=dx2;

	cout<<"Bmin "<<Bmin<<" Bmax "<<Bmax<<"\n";
	
// 	Point<dim> p1(xmin1,xmin2);
// 	Point<dim> p2(xmax1,xmax2);

	std::vector<unsigned> refinement={static_cast<unsigned>(pow(2,refs))+3, static_cast<unsigned>(pow(2,refs))+3};

	GridGenerator::subdivided_hyper_rectangle(triangulation, refinement, xmin, xmax);

	grid_points=triangulation.get_vertices();
	
	GridGenerator::subdivided_hyper_cube(integral_triangulation_x, pow(2, refs-3), Bmin[0], Bmax[0]);
	
	GridGenerator::subdivided_hyper_cube(integral_triangulation_y, pow(2, refs-3), Bmin[1], Bmax[1]);

	std::ofstream out ("grid.eps");
	GridOut grid_out;
	grid_out.write_eps (triangulation, out);

  }

template<int dim>
void Opzione<dim>::setup_system() {

	dof_handler.distribute_dofs(fe);
	
	dof_handler_integral_x.distribute_dofs(fe_integral_x);
	dof_handler_integral_y.distribute_dofs(fe_integral_y);

	std::cout << " Number of degrees of freedom: "
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

	solution.reinit(dof_handler.n_dofs());
	system_rhs.reinit(dof_handler.n_dofs());

  }

template<int dim>
void Opzione<dim>::assemble_system() {
	
	Levy_integral_part1();
	
	QGauss<dim> quadrature_formula(2);                  // 2 nodes, 2d -> 4 quadrature points per cell

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
	FullMatrix<double> cell_system(dofs_per_cell);

	typename DoFHandler<dim>::active_cell_iterator
	cell=dof_handler.begin_active(),
	endc=dof_handler.end();

	// building tensors
	Tensor< dim , dim, double > sigma_matrix;

	sigma_matrix[0][0]=par.sigma1*par.sigma1;
	sigma_matrix[1][1]=par.sigma2*par.sigma2;
	sigma_matrix[0][1]=par.sigma1*par.sigma2*par.ro;
	sigma_matrix[1][0]=par.sigma1*par.sigma2*par.ro;
	/*
	Tensor< 1 , dim, double > ones;
	for (unsigned i=0;i<dim;++i)
	ones[i]=1;
	 */
	Tensor< 1, dim, double > trasp;
	trasp[0]=par.r-par.sigma1*par.sigma1/2-alpha1;
	trasp[1]=par.r-par.sigma2*par.sigma2/2-alpha2;

	// cell loop
	for (; cell !=endc;++cell) {
	 fe_values.reinit(cell);
	 cell_dd=0;
	 cell_fd=0;
	 cell_ff=0;
	 cell_system=0;
	 for (unsigned q_point=0;q_point<n_q_points;++q_point)
	 for (unsigned i=0;i<dofs_per_cell;++i)
	 for (unsigned j=0; j<dofs_per_cell;++j) {

	  // mass matrix
	  cell_ff(i, j)+=fe_values.shape_value(i, q_point)*fe_values.shape_value(j, q_point)*fe_values.JxW(q_point);

	  // system matrix
	  cell_system(i, j)+=fe_values.JxW(q_point)*
	  (0.5*fe_values.shape_grad(i, q_point)*sigma_matrix*fe_values.shape_grad(j, q_point)-
	   fe_values.shape_value(i, q_point)*(trasp*fe_values.shape_grad(j,q_point))+
	   (1/time_step+par.r+par.lambda1+par.lambda2)*
	   fe_values.shape_value(i, q_point)*fe_values.shape_value(j, q_point));

	}

	 cell->get_dof_indices (local_dof_indices);

	 for (unsigned int i=0; i<dofs_per_cell;++i)
	 for (unsigned int j=0; j< dofs_per_cell; ++j) {

	  ff_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_ff(i, j));
	  system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_system(i, j));

	}

   }

	system_M2.add(1/time_step, ff_matrix);

#ifdef __VERBOSE__
	cout<<"system_matrix\n";
	system_matrix.print_formatted(cout);
	cout<<"system_M2\n";
	system_M2.print_formatted(cout);
#endif
  }

template<int dim>
void Opzione<dim>::solve() {

	VectorTools::interpolate (dof_handler, PayOff<dim>(par.K, par.S01, par.S02), solution);

	unsigned int Step=Nsteps;

	// Printing beginning solution
	{
	 DataOut<2> data_out;

	 data_out.attach_dof_handler (dof_handler);
	 data_out.add_data_vector (solution, "begin");

	 data_out.build_patches ();

	 std::ofstream output ("begin.gpl");
	 data_out.write_gnuplot (output);
 }
	//

	Boundary_Condition<dim> bc(par.S01, par.S02, par.K, par.T, par.r);
	cout<< "time step is"<< time_step<< endl;
	for (double time=par.T-time_step;time >=0;time-=time_step, --Step) {
	 cout<< "Step "<< Step<<"\t at time \t"<< time<< endl;

	 system_M2.vmult(system_rhs, solution);

#ifdef __VERBOSE__
	 cout<<"rhs ";
	 system_rhs.print(cout);
	 cout<<"\n";
#endif

	 bc.set_time(time);

	 {

	  std::map<types::global_dof_index,double> boundary_values;

	  VectorTools::interpolate_boundary_values (dof_handler,
	   0,
	   bc,
	   boundary_values);

	  MatrixTools::apply_boundary_values (boundary_values,
	   system_matrix,
	   solution,
	   system_rhs, false);

  }

	 if (time==par.T-time_step) {
	  system_matrix.print(cout);
	}

	 SparseDirectUMFPACK solver;
	 solver.initialize(sparsity_pattern);
	 solver.factorize(system_matrix);
	 solver.solve(system_rhs);

	 solution=system_rhs;

   }

	// Printing final solution
	{
	 DataOut<2> data_out;

	 data_out.attach_dof_handler (dof_handler);
	 data_out.add_data_vector (solution, "end");

	 data_out.build_patches ();

	 std::ofstream output ("end.gpl");
	 data_out.write_gnuplot (output);
 }
	//

	ran=true;

  }

template<int dim>
double Opzione<dim>::get_price() {

	if (ran==false) {
	 this->run();
   }

	// Creo nuova grigla ( che passi da (0,0) )
	Triangulation<dim> price;
	// Creo degli fe
	FE_Q<dim> fe2 (1);
	// Creo un DoFHandler e lo attacco a price
	DoFHandler<dim> dof_handler_2 (price);
	// Costruisco la griglia, in modo che passi da (0,0) e non la rifinisco
	GridGenerator::hyper_rectangle(price, Point<dim> (0.,0.), Point<dim> (xmax1,xmax2));
	// Assegno a dof_handler_2 gli elementi finit fe2 appena creati
	dof_handler_2.distribute_dofs(fe2);
	// Definisco questa fantomatica funzione FEFieldFunction
	Functions::FEFieldFunction<dim> fe_function (dof_handler, solution);
	// Creo il vettore che conterrà i valori interpolati
	Vector<double> solution_vector(4);
	// Interpolo
	VectorTools::interpolate(dof_handler_2, fe_function, solution_vector);
	// Ritorno il valore interpolato della soluzione in (0,0)
	return solution_vector[0];
  }

int main() {
	Parametri2d par;
/*
	par.T=1.;
	par.K=200;
	par.S01=100;
	par.S02=100;
	par.r=0.0367;
	par.sigma1=0.120381;
	par.sigma2=0.09;
	par.ro=0.2;
 */

	par.T=1.;
	par.K=200;
	par.S01=100;
	par.S02=100;
	par.r=0.1;
	par.sigma1=0.2;
	par.sigma2=0.5;
	par.ro=-0.2;

	// Parametri della parte salto
	par.p=0.20761;                                      // Parametro 1 Kou
	par.lambda=0.330966;                                // Parametro 2 Kou
	par.lambda_piu=9.65997;                             // Parametro 3 Kou
	par.lambda_meno=3.13868;                            // Parametro 4 Kou

	// tempo // spazio
	Opzione<2> Call(par, 100, 0);
	double prezzo=Call.run();

	cout<<"Prezzo "<<prezzo<<"\n";
	cout<<"2d v001\n";

	return 0;
  }