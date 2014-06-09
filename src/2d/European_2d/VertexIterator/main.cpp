#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/base/qprojector.h>
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

#include <deal.II/grid/grid_refinement.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/numerics/derivative_approximation.h>
#include <deal.II/numerics/solution_transfer.h>

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

class Parametri2d{
public:
	//Dati
	double T;                                                   // Scadenza
	double K;                                                   // Strike price
	double S01;                                                 // Spot price
	double S02;                                                 // Spot price
	double r;                                                   // Tasso risk free

	// Parametri della parte continua
	double sigma1;                                              // Volatilità
	double sigma2;                                              // Volatilità
	double rho;                                                 // Volatilità

	// Parametri della parte salto
	double p1;                                                  // Parametro 1 Kou
	double lambda1;                                             // Parametro 2 Kou
	double lambda_piu_1;                                        // Parametro 3 Kou
	double lambda_meno_1;                                       // Parametro 4 Kou

	double p2;                                                  // Parametro 1 Kou
	double lambda2;                                             // Parametro 2 Kou
	double lambda_piu_2;                                        // Parametro 3 Kou
	double lambda_meno_2;                                       // Parametro 4 Kou


	Parametri2d()=default;
	Parametri2d(const Parametri2d &)=default;
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
	return max(p(0)+p(1)-K,0.);
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
	return max(p[0]+p[1]-_K*exp(-_r*(_T-this->get_time())), 0.);

}

template<int dim>
class Kou_Density: public Function<dim>
{

public:
	Kou_Density(unsigned int ax,  double p,  double lam, double lam_u,  double lam_d) : Function<dim>(),  _ax(ax),  _p(p),  _lam(lam), 
	_lam_u(lam_u),  _lam_d(lam_d) {};

	//value in the point p
	virtual double value (const Point<dim> &p,  const unsigned int component=0) const;
	//same but vector of points
	virtual void value_list(const std::vector<Point<dim> > &points,
	 std::vector<double> &values,
	 const unsigned int component = 0) const;
private:
	//indicates wich ax
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
class Vertex_Iterator {

private:
	typename dealii::DoFHandler<dim>::active_cell_iterator _cells;
	std::vector<bool> _used;
	unsigned _j;
	unsigned _nd;
	
public:
	Vertex_Iterator()=delete;
	Vertex_Iterator(dealii::DoFHandler<dim> const & _dof);
	
	dealii::types::global_dof_index get_global_index();
	
	static Vertex_Iterator last_vertex(dealii::DoFHandler<dim> const & dof);
	
	bool at_end();
	
	//opertors
	bool operator== (Vertex_Iterator<dim> const & rhs);
	bool operator!= (Vertex_Iterator<dim> const & rhs);
	Vertex_Iterator & operator++ ();

};


template<int dim>
Vertex_Iterator<dim>::Vertex_Iterator(dealii::DoFHandler<dim> const & dof) : _used(dof.n_dofs(), false), _j(0) {
	
	_nd=dof.get_fe().dofs_per_cell;
	_cells=dof.begin_active();
}

template<int dim>
dealii::types::global_dof_index Vertex_Iterator<dim>::get_global_index() {
	std::vector<dealii::types::global_dof_index> local_ind(_nd);
	
	_cells->get_dof_indices(local_ind);
	
	return local_ind[_j];
}

template<int dim>
Vertex_Iterator<dim> & Vertex_Iterator<dim>::operator++() {
	
	do{
	 if (_j<_nd)
	  _j++;
	else {
	 ++_cells;
	 _j=0;
	 }
   
	} while (_used[get_global_index()]);
	
	_used[get_global_index()]=true;
	
	return * this;
}

template<int dim>
bool Vertex_Iterator<dim>::operator== (Vertex_Iterator<dim> const & rhs) {
	return (_cells==rhs._cells && _j==rhs._j);
}


template<int dim>
bool Vertex_Iterator<dim>::operator != (Vertex_Iterator<dim> const & rhs) {
	return (_cells !=rhs._cells or _j !=rhs._j);
}

template<int dim>
Vertex_Iterator<dim> Vertex_Iterator<dim>::last_vertex(dealii::DoFHandler<dim> const & dof) {
	
	Vertex_Iterator<dim> last(dof);
	last._cells=dof.end();
	last._j=last._nd-1;
	
	return last;
	
}

template<int dim>
bool Vertex_Iterator<dim>::at_end() {
	
}

template<int dim>
class Opzione{
private:
	Parametri2d par;
	void make_grid();
	void setup_system ();
	void assemble_system ();
	void solve_one_step(double time);
	void refine_grid ();
	// 	void solve ();
	void output_results () const {};

	Kou_Density<dim>				k_x;
	Kou_Density<dim>				k_y;

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

	ConstraintMatrix                constraints;

	Vector<double>                  solution;
	Vector<double>                  system_rhs;

	std::vector< Point<dim> >       grid_points;

	unsigned int refs, Nsteps;
	double time_step;

	Point<dim> Smin, Smax;
	double price;
	double alpha_x;
	double alpha_y;

	void Levy_integral_part1();
	void Levy_integral_part2(Vector<double> &J_x, Vector<double> &J_y);

	bool ran;

public:
	Opzione(Parametri2d const &par_, int Nsteps_,  int refinement):
	par(par_),
	k_x(0, par.p1 , par.lambda1, par.lambda_piu_1, par.lambda_meno_1),
	k_y(1, par.p2 , par.lambda2, par.lambda_piu_2, par.lambda_meno_2),
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

	alpha_x=0;
	alpha_y=0;
	Point<dim> Bmin(0., 0.), Bmax(Smax);
	double step(0.5);

	while (k_x.value(Bmin)>toll)
	Bmin[0]+=-step;

	while (k_y.value(Bmin)>toll)
	Bmin[1]+=-step;

	while (k_x.value(Bmax)>toll)
	Bmin[0]+=step;

	while (k_y.value(Bmax)>toll)
	Bmin[1]+=step;

	{
	 Triangulation<1> integral_grid;
	 FE_Q<1> fe2(1);
	 DoFHandler<1> dof_handler2(integral_grid);

	 GridGenerator::hyper_cube<1>(integral_grid, Bmin[0], Bmax[0]);
	 integral_grid.refine_global(15);

	 dof_handler2.distribute_dofs(fe2);

	 QGauss<1> quadrature_formula(8);
	 FEValues<1> fe_values(fe2, quadrature_formula,  update_quadrature_points |update_JxW_values);

	 typename DoFHandler<1>::active_cell_iterator
	 cell=dof_handler2.begin_active(),
	 endc=dof_handler2.end();

	 const unsigned int n_q_points(quadrature_formula.size());

	 for (; cell !=endc;++cell) {
	  fe_values.reinit(cell);
	  std::vector< Point<1> > quad_points_1D(fe_values.get_quadrature_points());

	  for (unsigned q_point=0;q_point<n_q_points;++q_point) {
	   Point<dim> p(quad_points_1D[q_point][0], 0.);
	   alpha_x+=fe_values.JxW(q_point)*(exp(p[0])-1.)*k_x.value(p);
	 }
	}

	 cout<< "alpha_x is "<< alpha_x<< endl;
 }

	{
	 Triangulation<1> integral_grid;
	 FE_Q<1> fe2(1);
	 DoFHandler<1> dof_handler2(integral_grid);

	 GridGenerator::hyper_cube<1>(integral_grid, Bmin[1], Bmax[1]);
	 integral_grid.refine_global(15);

	 dof_handler2.distribute_dofs(fe2);

	 QGauss<1> quadrature_formula(8);
	 FEValues<1> fe_values(fe2, quadrature_formula,  update_quadrature_points |update_JxW_values);

	 typename DoFHandler<1>::active_cell_iterator
	 cell=dof_handler2.begin_active(),
	 endc=dof_handler2.end();

	 const unsigned int n_q_points(quadrature_formula.size());

	 for (; cell !=endc;++cell) {
	  fe_values.reinit(cell);
	  std::vector< Point<1> > quad_points_1D(fe_values.get_quadrature_points());

	  for (unsigned q_point=0;q_point<n_q_points;++q_point) {
	   Point<dim> p(0., quad_points_1D[q_point][0]); alpha_y+=fe_values.JxW(q_point)*(exp(p[1])-1.)*k_y.value(p);
	 }
	}

	 cout<< "alpha_y is "<< alpha_y<< endl;
 }



	return;

  }

template<int dim>
void Opzione<dim>::Levy_integral_part2(Vector<double> &J_x, Vector<double> &J_y) {

	double grid_tol(1.0e-6);
	J_x.reinit(solution.size());
	J_y.reinit(solution.size());

	unsigned int N(solution.size());

	QGauss<1> quad1D(3);    
	FEFaceValues<dim> fe_face(fe, quad1D, update_values  | update_quadrature_points | update_JxW_values);

	const unsigned int n_q_points=quad1D.size();

	Point<dim> z, karg;

	typename DoFHandler<dim>::active_cell_iterator cell=dof_handler.begin_active(),  endc=dof_handler.end();
	const unsigned int   dofs_per_cell = fe.dofs_per_cell;
	vector<bool> used(N, false);
	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
	
	Vertex_Iterator<dim> verts(dof_handler);
	Vertex_Iterator<dim> last=Vertex_Iterator<dim>::last_vertex(dof_handler);
	
	Vertex_Iterator<dim> last2=Vertex_Iterator<dim>::last_vertex(dof_handler);

// 	if (verts==last)
// 	 cout<< "differents at start\n";
	if (last !=last2)
	 cout<< "really different\n";
	for (unsigned i=0;verts !=last;++verts) {
	 i++;
	 cout<< " i is \t"<< i<< endl;
	 }
	for (; cell !=endc;++cell) {

	 cell->get_dof_indices(local_dof_indices);

	 for (unsigned int j=0;j<dofs_per_cell;++j) {
	  unsigned int it=local_dof_indices[j];
	  //                 cout << "Nodo globale "<< it<< endl;
	  if (used[it]==false)
	  {

	   used[it]=true;
	   Point<dim> actual_vertex=cell->vertex(j);
	   // 		cout<< "At point N "<< it<<" wich is "<<  actual_vertex<< endl;
	   typename DoFHandler<dim>::active_cell_iterator inner_cell=dof_handler.begin_active();
	   bool left(false),  bottom(false);
	   if (fabs(actual_vertex[0]-Smin[0])<grid_tol) {
		left=true;
	  // 		cout<< "it's a left node\n";
	  }
	   if (fabs(actual_vertex[1]-Smin[1])<grid_tol) {
		bottom=true;
	  // 	    cout<< "it's a bottom node \n";
	  }
	   for (;inner_cell !=endc;++inner_cell)
	   {
		if (left && inner_cell->face(0)->at_boundary())
		{
	  // 		  cout<< "\t boundary cell left\n";
	  unsigned actual_face(0);
	  fe_face.reinit(inner_cell, actual_face);
	  vector<Point <dim> > quad_points=fe_face.get_quadrature_points();

	  vector<double> sol_values(n_q_points);
	  fe_face.get_function_values(solution,  sol_values);

	  for (unsigned q_point=0;q_point<n_q_points;++q_point) {
	   z=quad_points[q_point];
	   karg(1)=log(z(1)/actual_vertex(1));
	   J_y[it]+=fe_face.JxW(q_point)*k_y.value(karg)*sol_values[q_point]/z(1);
	 }

  }

		if (bottom && inner_cell->face(2)->at_boundary())
		{
	  // 		  cout<< "\t boundary cell bottom\n";
	  unsigned actual_face(2);
	  fe_face.reinit(inner_cell, actual_face);
	  vector<Point <dim> > quad_points=fe_face.get_quadrature_points();

	  vector<double> sol_values(n_q_points);
	  fe_face.get_function_values(solution,  sol_values);

	  for (unsigned q_point=0;q_point<n_q_points;++q_point) {
	   z=quad_points[q_point];
	   karg(0)=log(z(0)/actual_vertex(0));
	   J_x[it]+=fe_face.JxW(q_point)*k_x.value(karg)*sol_values[q_point]/z(0);
	 }

  }


		if (fabs(inner_cell->face(3)->center()(1)-actual_vertex(1))<grid_tol) 
		{
	  // 		  cout<< "\t\t operating on upper face\n";
	  unsigned face(3);
	  fe_face.reinit(inner_cell, face);
	  vector<Point <dim> > quad_points=fe_face.get_quadrature_points();

	  vector<double> sol_values(n_q_points);
	  fe_face.get_function_values(solution,  sol_values);

	  for (unsigned q_point=0;q_point<n_q_points;++q_point) {
	   z=quad_points[q_point];
	   karg(0)=log(z(0)/actual_vertex(0));

	   J_x[it]+=fe_face.JxW(q_point)*k_x.value(karg)*sol_values[q_point]/z(0);
	 }

  }

		if (fabs(inner_cell->face(1)->center()(0)-actual_vertex(0))<grid_tol) 
		{
	  // 			cout<< "\t\t Operating on right face\n";
	  unsigned face(1);
	  fe_face.reinit(inner_cell, face);
	  vector<Point <dim> > quad_points=fe_face.get_quadrature_points();

	  vector<double> sol_values(n_q_points);
	  fe_face.get_function_values(solution,  sol_values);

	  for (unsigned q_point=0;q_point<n_q_points;++q_point) {
	   z=quad_points[q_point];
	   karg(1)=log(z(1)/actual_vertex(1));

	   J_y[it]+=fe_face.JxW(q_point)*k_y.value(karg)*sol_values[q_point]/z(1);
	 }

  }


	}
   }

	}
   }
  }


template<int dim>
void Opzione<dim>::make_grid() {

	Smin[0]=0.5*par.S01*exp((par.r-par.sigma1*par.sigma1/2)*par.T
	 -par.sigma1*sqrt(par.T)*6);

	Smax[0]=1.5*par.S01*exp((par.r-par.sigma1*par.sigma1/2)*par.T
	 +par.sigma1*sqrt(par.T)*6);

	Smin[1]=0.5*par.S02*exp((par.r-par.sigma2*par.sigma2/2)*par.T
	 -par.sigma2*sqrt(par.T)*6);

	Smax[1]=1.5*par.S02*exp((par.r-par.sigma2*par.sigma2/2)*par.T
	 +par.sigma2*sqrt(par.T)*6);


	cout<< "Smin= "<< Smin<< "\t e Smax= "<< Smax<< endl;

	std::vector<unsigned> refinement={static_cast<unsigned>(pow(2,refs))+3, static_cast<unsigned>(pow(2,refs))+3};

	GridGenerator::subdivided_hyper_rectangle(triangulation, refinement,Smin, Smax);

	Levy_integral_part1();

	std::ofstream out ("grid.eps");
	GridOut grid_out;
	grid_out.write_eps (triangulation, out);
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
	if (cell->face(face)->at_boundary())
	cell->face(face)->set_boundary_indicator (0);

	cout<< "Controlling Boundary indicators\n";
	vector<types::boundary_id> info;
	info=triangulation.get_boundary_indicators();
	cout<< "Number of Boundaries: " << info.size()<< endl;
	cout<< "which are"<< endl;
	for (unsigned int i=0; i<info.size();++i)
	cout<< info[i] << endl;

	solution.reinit(dof_handler.n_dofs());
	system_rhs.reinit(dof_handler.n_dofs());

	grid_points=triangulation.get_vertices();
  }




template<int dim>
void Opzione<dim>::assemble_system() {


	QGauss<dim> quadrature_formula(4);
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

	  trasp[0]=par.sigma1*par.sigma1*quad_points[q_point][0]
	  +0.5*par.rho*par.sigma1*par.sigma2*quad_points[q_point][0]+(alpha_x-par.r)*quad_points[q_point][0];
	  trasp[1]=par.sigma2*par.sigma2*quad_points[q_point][1]
	  +0.5*par.rho*par.sigma1*par.sigma2*quad_points[q_point][1]+(alpha_y-par.r)*quad_points[q_point][1];

	  sig_mat[0][0]=0.5*par.sigma1*par.sigma1
	  *quad_points[q_point][0]*quad_points[q_point][0];
	  sig_mat[1][1]=0.5*par.sigma2*par.sigma2
	  *quad_points[q_point][1]*quad_points[q_point][1];
	  sig_mat[0][1]=0.5*par.rho*par.sigma1*par.sigma2*
	  quad_points[q_point][0]*quad_points[q_point][1];
	  sig_mat[1][0]=sig_mat[0][1];


	  for (unsigned i=0;i<dofs_per_cell;++i)
	  for (unsigned j=0; j<dofs_per_cell;++j) {


	   cell_mat(i, j)+=fe_values.JxW(q_point)*(
		(1/time_step+par.r+par.lambda1+par.lambda2)*fe_values.shape_value(i, q_point)*fe_values.shape_value(j,q_point)
		+fe_values.shape_grad(i, q_point)*sig_mat*fe_values.shape_grad(j, q_point)
		+fe_values.shape_value(i, q_point)*trasp*fe_values.shape_grad(j, q_point));

	   cell_ff(i, j)+=fe_values.shape_value(i, q_point)*fe_values.shape_value(j, q_point)*fe_values.JxW(q_point);

	 }
	}

	 cell->get_dof_indices (local_dof_indices);
	 constraints.distribute_local_to_global(cell_mat, local_dof_indices, system_matrix);
	 constraints.distribute_local_to_global(cell_ff, local_dof_indices, ff_matrix);
// 	 for (unsigned int i=0; i<dofs_per_cell;++i)
// 	 for (unsigned int j=0; j< dofs_per_cell; ++j) {

	 
// 	  system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_mat(i, j));
// 	  ff_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_ff(i, j));

// 	}
   }

	system_M2.add(1/time_step, ff_matrix);

  }

template<int dim>
void Opzione<dim>::solve_one_step(double time) {

	Boundary_Right_Side<dim> right_bound(par.K, par.T, par.r);

	Vector<double> J_x, J_y;
	Levy_integral_part2(J_x, J_y);

	system_M2.vmult(system_rhs, solution);
	{
	 Vector<double> temp;
	 temp.reinit(dof_handler.n_dofs());
	 ff_matrix.vmult(temp, J_x);

	 system_rhs+=temp;

	 temp.reinit(dof_handler.n_dofs());
	 ff_matrix.vmult(temp, J_y);

	 system_rhs+=temp;
 }


	right_bound.set_time(time);

	{

	 std::map<types::global_dof_index,double> boundary_values;
	 VectorTools::interpolate_boundary_values (dof_handler,
	  0,
	  right_bound, 
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

	KellyErrorEstimator<dim>::estimate (dof_handler,
	 QGauss<dim-1>(3),
	 typename FunctionMap<dim>::type(),
	 solution,
	 estimated_error_per_cell);

	// 	   GridRefinement::refine_and_coarsen_optimize (triangulation,estimated_error_per_cell);

	GridRefinement::refine_and_coarsen_fixed_number (triangulation, estimated_error_per_cell, 0.03, 0.1);

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
double Opzione<dim>::get_price() {

	if (ran==false) {
	 this->run();
   }

	Point<dim> p(par.S01, par.S02);
	Functions::FEFieldFunction<dim> fe_function (dof_handler, solution);
	return fe_function.value(p);
  }


int main() {
	Parametri2d par;
	par.T=1.;
	par.K=200;
	par.S01=80;
	par.S02=120;
	par.r=0.1;
	par.sigma1=0.1256;
	par.sigma2=0.2;
	par.rho=-0.2;

	// Parametri della parte salto
	par.p1=0.20761;                                             // Parametro 1 Kou
	par.lambda1=0.330966;                                       // Parametro 2 Kou
	par.lambda_piu_1=9.65997;                                   // Parametro 3 Kou
	par.lambda_meno_1=3.13868;                                  // Parametro 4 Kou

	// Parametri della parte salto

	par.p2=0.20761;                                             // Parametro 1 Kou
	par.lambda2=0.330966;                                       // Parametro 2 Kou
	par.lambda_piu_2=9.65997;                                   // Parametro 3 Kou
	par.lambda_meno_2=3.13868;                                  // Parametro 4 Kou

	cout<<"eps "<<eps<<"\n";

	Opzione<2> Call(par, 100, 4);
	double Prezzo=Call.run();
	cout<<"Prezzo "<<Prezzo<<"\n";

	cout<<"Kou non cost\n";
	return 0;
  }