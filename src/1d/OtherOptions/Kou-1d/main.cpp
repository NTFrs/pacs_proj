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

// laguerre stuff found on http://people.sc.fsu.edu/~jburkardt/cpp_src/laguerre_rule/laguerre_rule.cpp
// This code is distributed under the GNU LGPL license.
void cdgqf ( int nt, int kind, double alpha, double beta, double t[], 
            double wts[] );
void cgqf ( int nt, int kind, double alpha, double beta, double a, double b, 
           double t[], double wts[] );
double class_matrix ( int kind, int m, double alpha, double beta, double aj[], 
                     double bj[] );
void imtqlx ( int n, double d[], double e[], double z[] );
void parchk ( int kind, int m, double alpha, double beta );
double r8_abs ( double x );
double r8_epsilon ( );
double r8_huge ( );
double r8_sign ( double x );
void r8mat_write ( string output_filename, int m, int n, double table[] );
void rule_write ( int order, string filename, double x[], double w[], 
                 double r[] );
void scqf ( int nt, double t[], int mlt[], double wts[], int nwts, int ndx[], 
           double swts[], double st[], int kind, double alpha, double beta, double a, 
           double b );
void sgqf ( int nt, double aj[], double bj[], double zemu, double t[], 
           double wts[] );
void timestamp ( );
// laguerre stuff

//#define dim 1

//#define __PIDE__
#define __MATLAB__
//#define __INTERPOLATION__

const double toll=1e-8;
const double eps=std::numeric_limits<double>::epsilon();
/*
 auto greater = [] (Point<dim> p1, Point<dim> p2) {
 return p1[0]<p2[0];
 };
 
 sort(integral_points.begin(), integral_points.end(), greater);
 */

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

// auto grid_order = [] (Point<1> p1, Point<1> p2) {return p1[0]<p2[0];};

template<int dim>
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

template<int dim>
double PayOff<dim>::value (const Point<dim>  &p,
                           const unsigned int component) const
{
	Assert (component == 0, ExcInternalError());
	return max(K-S0*exp(p(0)),0.);
}

template<int dim>
class Boundary_Right_Side : public Function<dim>
{
public:
	Boundary_Right_Side() : Function< dim>() {};
        
	virtual double value (const Point<dim> &p, const unsigned int component =0) const;
        
};

template<int dim>
double Boundary_Right_Side<dim>::value(const Point<dim> &p, const unsigned int component) const
{
	Assert (component == 0, ExcInternalError());
	return 0;
        
}

template<int dim>
class Boundary_Left_Side: public Function<dim>
{
public:
	Boundary_Left_Side(double S0, double K, double T,  double r) : Function< dim>(), _S0(S0), _K(K), _T(T), _r(r) {};
        
	virtual double value (const Point<dim> &p, const unsigned int component =0) const;
private:
	double _S0;
	double _K;
	double _T;
	double _r;
};

template<int dim>
double Boundary_Left_Side<dim>::value(const Point<dim> &p, const unsigned int component) const
{
	Assert (component == 0, ExcInternalError());
	return _K*exp(-_r*(_T-this->get_time()))-_S0*exp(p[0]);
        
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
class Solution_Trimmer: public Function<dim>
{
private:
	//check that this causes no memory leaks while keeping hereditariety
	Function<dim> * _left;  
	Function<dim> * _right;
	DoFHandler<dim> const & _dof;
	Vector<double> const & _sol;
	Point<dim> _l_lim, _r_lim;
	Functions::FEFieldFunction<dim> _fe_func;
	
public:
	Solution_Trimmer(       Function<dim> * left,
                                Function<dim> * right,
                                DoFHandler<dim> const & dof,
                                Vector<double> const & sol, 
                                Point<dim> const & xmin,
                                Point<dim> const & xmax         )
                                :
                                _left(left),
                                _right(right),
                                _dof(dof),
                                _sol(sol),
                                _l_lim(xmin),
                                _r_lim(xmax),
                                _fe_func(_dof, _sol)
                                {};
	
	virtual double value(const Point<dim> &p,  const unsigned int component=0) const;
	virtual void value_list(const std::vector<Point<dim> > &points,
                                std::vector<double> &values,
                                const unsigned int component = 0) const;
};

template<int dim>
double Solution_Trimmer<dim>::value(const Point<dim> &p,  const unsigned int component) const
{
	Assert (component == 0, ExcInternalError());
	
	if (p[0]<_l_lim[0])
                return _left->value(p);
	if (p[0]>_r_lim[0])
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
                if (points[i][0]<_l_lim[0])
                        values[i]=_left->value(points[i]);
                else if (points[i][0]>_r_lim[0])
                        values[i]=_right->value(points[i]);
                else
                        values[i]=_fe_func.value(points[i]);
        }
}

class Quadrature_Laguerre{
private:
        
        std::vector<double> nodes;
        std::vector<double> weights;
        unsigned order; 
        
public:
        
        Quadrature_Laguerre()=default;
        
        // il costruttore costruisce nodi e pesi
        Quadrature_Laguerre(unsigned n, double lambda){
                
                order=n;
                
                nodes=std::vector<double> (order);
                weights=std::vector<double> (order);
                
                unsigned kind = 5; // kind=5, Generalized Laguerre, (a,+oo) (x-a)^alpha*exp(-b*(x-a))
                
                //cgqf ( int nt, int kind, double alpha, double beta, double a, double b, double t[], double wts[] )
                cgqf ( order, kind, 0., 0., 0., lambda, nodes.data(), weights.data() );
        }
        
        inline unsigned get_order () {return order;}
        inline std::vector<double> const & get_nodes () {return nodes;}
        inline std::vector<double> const & get_weights () {return weights;}
};


template<int dim>
class Opzione{
private:
	Parametri par;
	void make_grid();
	void setup_system ();
	void assemble_system ();
	void solve ();
	void output_results () const {};
        
	
        
	Kou_Density<dim>				k;    
        
	Triangulation<dim>              triangulation;
	FE_Q<dim>                       fe;
	DoFHandler<dim>                 dof_handler;
        
	Triangulation<dim>              integral_triangulation;
	//     Triangulation<dim>              integral_triangulation2;
        
	FE_Q<dim>                       fe2;
	DoFHandler<dim>                 dof_handler_2;
        
	SparsityPattern                 sparsity_pattern;
	SparseMatrix<double>            system_matrix;
	SparseMatrix<double>            system_M2;
	SparseMatrix<double>            dd_matrix;
	SparseMatrix<double>            fd_matrix;
	SparseMatrix<double>            ff_matrix;
        
	Vector<double>                  solution;
	Vector<double>                  system_rhs;
        
        // 	std::vector<unsigned int>       index;
        // 	std::vector<double>             u_array;
        // 	double *                        x_array;
        
	std::vector< Point<dim> >       grid_points;
        // 	std::vector< Point<dim> >       integral_grid_points;
        
        // 	std::vector< double >           integral_weights;
        // 	std::vector< Point<dim> >       integral_points;
        
        // quadrature di laguerre
        Quadrature_Laguerre right_quad;
        Quadrature_Laguerre left_quad;
        
        std::vector<double> right_quad_nodes;
        std::vector<double> left_quad_nodes;
        std::vector<double> right_quad_weights;
        std::vector<double> left_quad_weights;
        
        std::vector<Point<dim> > quadrature_points;
        
	unsigned int refs, Nsteps;
	double time_step;
	double dx;
	double Smin, Smax;
	double                          price;
	Point<dim> xmin, xmax, Bmin, Bmax;
        
	double alpha;
        
        
        
	void Levy_integral_part1();
	void Levy_integral_part2(Vector<double> &J,  double time);
        // 	void f_u(std::vector<Point<dim> > &val, std::vector<Point<dim> > const &y);
        // 	inline double payoff(double x, double K, double S0){return max(S0*exp(x)-K, 0.);}
        // 	void calculate_weights();
        
	bool ran;
        
public:
	Opzione(Parametri const &par_, int Nsteps_,  int refinement):
	par(par_),
	k(par.p, par.lambda, par.lambda_piu, par.lambda_meno),
	fe (1),
	dof_handler (triangulation),
	fe2 (1),
	dof_handler_2 (integral_triangulation),
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
void Opzione<dim>::Levy_integral_part1(){
        
	alpha=0;
        
        for (int i=0; i<right_quad.get_order(); ++i) {
                alpha+=(exp(right_quad_nodes[i])-1)*par.p*par.lambda*par.lambda_piu*right_quad_weights[i];
        }

        for (int i=0; i<left_quad.get_order(); ++i) {
                        // il - è perché i nodi sono positivi (Quadrature_Laguerre integra da 0 a \infty)
                alpha+=(exp(-left_quad_nodes[i])-1)*(1-par.p)*par.lambda*par.lambda_meno*left_quad_weights[i];
        }
        
	return;
}

template<int dim>
void Opzione<dim>::Levy_integral_part2(Vector<double> &J,  double time) {
        
	J.reinit(solution.size());// If fast is false, the vector is filled by zeros
        
	Boundary_Left_Side<dim>         leftie(par.S0, par.K, par.T, par.r);
	Boundary_Right_Side<dim>        rightie;
	leftie.set_time(time);rightie.set_time(time);
	
	Solution_Trimmer<dim> func(&leftie, &rightie, dof_handler, solution, xmin, xmax);

//#pragma omp parallel for
        for (int it=0; it<J.size(); ++it) {
                
                std::vector< Point<dim> > quad_points(left_quad.get_order()+right_quad.get_order());
                
                std::vector<double> f_u(left_quad.get_order()+right_quad.get_order());
                
                // Inserisco in quad_points tutti i punti di quadrature shiftati
                for (int i=0; i<quad_points.size(); ++i) {
                        quad_points[i]=quadrature_points[i] + grid_points[it];
                }
                
                // valuto f_u in quad_points
                func.value_list(quad_points, f_u);
                
                // Integro dividendo fra parte sinistra e parte destra dell'integrale
                for (int i=0; i<left_quad.get_order(); ++i) {
                        J(it)+=f_u[i]*(1-par.p)*par.lambda*par.lambda_meno*left_quad_weights[i];
                }
                for (int i=0; i<right_quad.get_order(); ++i) {
                        J(it)+=f_u[i+left_quad.get_order()]*par.p*par.lambda*par.lambda_piu*right_quad_weights[i];
                }

        }

}


template<int dim>
void Opzione<dim>::make_grid() {
        
	Smin=0.5*par.S0*exp((par.r-par.sigma*par.sigma/2)*par.T
                            -par.sigma*sqrt(par.T)*6);
	Smax=1.5*par.S0*exp((par.r-par.sigma*par.sigma/2)*par.T
                            +par.sigma*sqrt(par.T)*6);
        
	cout<< "Smin= "<< Smin<< "\t e Smax= "<< Smax<< endl;
	xmin[0]=0;
	xmax[0]=0;
        
	dx=(log(Smax/par.S0)-log(Smin/par.S0))/pow(2., refs);
	cout<<"dx "<<dx<<"\n";
        
	while (xmin[0]>log(Smin/par.S0))
                xmin[0]-=dx;
	while (xmax[0]<log(Smax/par.S0))
                xmax[0]+=dx;
	xmin[0]-=dx;
	xmax[0]+=dx;
        
	cout<<"dx "<<dx<<"\n";
        
	Bmin[0]=xmin[0];
	Bmax[0]=xmax[0];
        
        
	while(k.value(Bmin)>toll)
                Bmin[0]-=dx;
        
	while(k.value(Bmax)>toll)
                Bmax[0]+=dx;
        
	cout<<"Bmin "<<Bmin<<" Bmax "<<Bmax<<"\n";
        
	GridGenerator::subdivided_hyper_cube(triangulation,pow(2,refs)+3, xmin[0],xmax[0]);
        
	grid_points=triangulation.get_vertices();
	
	GridGenerator::subdivided_hyper_cube(integral_triangulation, pow(2, refs-3), Bmin[0], Bmax[0]);
	
        
}

template<int dim>
void Opzione<dim>::setup_system() {
        
	dof_handler.distribute_dofs(fe);
        
	dof_handler_2.distribute_dofs(fe2);
        
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
                                if (std::fabs(cell->face(face)->center()(0) - (xmax[0])) < toll)
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
        
        // Costruisco punti e nodi di Laguerre una volta per tutte (tanto non cambiano)
        right_quad=Quadrature_Laguerre(static_cast<unsigned>(round(Bmax[0]/dx)), par.lambda_piu);
        left_quad=Quadrature_Laguerre(static_cast<unsigned>(round(-Bmin[0]/dx)), par.lambda_meno);
        
        // Costruisco i vettori dei nodi e dei pesi per la parte destra e sinistra
        right_quad_nodes=right_quad.get_nodes();
        right_quad_weights=right_quad.get_weights();
        
        left_quad_nodes=left_quad.get_nodes();
        left_quad_weights=left_quad.get_weights();
        
        quadrature_points=std::vector<Point<dim> > (left_quad.get_order()+right_quad.get_order());
        
        // Costruisco un unico vettore con tutti i nodi di quadratura (quelli di sinistra cambiati di segno)
        for (int i=0; i<left_quad.get_order(); ++i) {
                quadrature_points[i]=static_cast< Point<dim> > (-left_quad_nodes[i]);
        }
        for (int i=0; i<right_quad.get_order(); ++i) {
                quadrature_points[i+left_quad.get_order()]=static_cast< Point<dim> > (right_quad_nodes[i]);
        }
}

template<int dim>
void Opzione<dim>::assemble_system() {
        
	Levy_integral_part1();
        
	cout<<"alpha "<<alpha<<" Bmin "<<Bmin<<" Bmax "<<Bmax<<"\n";
        
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
        
        
#ifdef __PIDE__
	double diff=par.sigma*par.sigma/2;
	double trasp=par.r-par.sigma*par.sigma/2-alpha;
	double reaz=-par.r-par.lambda;
        
	system_matrix.add(1/time_step-0.5*reaz, ff_matrix); 
	system_matrix.add(0.5*diff, dd_matrix);
	system_matrix.add(-0.5*trasp, fd_matrix);
        
	system_M2.add(1/time_step+0.5*reaz, ff_matrix); 
	system_M2.add(-0.5*diff, dd_matrix);
	system_M2.add(0.5*trasp, fd_matrix);
#else
	system_M2.add(1, ff_matrix);
	system_matrix.add(1, ff_matrix);
	system_matrix.add(par.sigma*par.sigma*time_step/2, dd_matrix);
	system_matrix.add(-time_step*(par.r-par.sigma*par.sigma/2), fd_matrix);
	system_matrix.add(par.r*time_step, ff_matrix);
        
#endif
        
}

template<int dim>
void Opzione<dim>::solve() {
        
	VectorTools::interpolate (dof_handler, PayOff<dim>(par.K, par.S0), solution);
        
	{
	 DataOut<1> data_out;

	 data_out.attach_dof_handler (dof_handler);
	 data_out.add_data_vector (solution, "begin");

	 data_out.build_patches ();

	 std::ofstream output ("begin.gpl");
	 data_out.write_gnuplot (output);
	 }
        
	unsigned int Step=Nsteps;
        
	Boundary_Left_Side<dim> left_bound(par.S0, par.K, par.T, par.r);
	Boundary_Right_Side<dim> right_bound;
	cout<< "time step is"<< time_step<< endl;
	for (double time=par.T-time_step;time >=0;time-=time_step, --Step) {
                cout<< "Step "<< Step<<"\t at time \t"<< time<< endl;
                
#ifdef __PIDE__
                Vector<double> J;
                J.reinit(solution.size());
                Levy_integral_part2(J, time);
                
                ff_matrix.vmult(system_rhs, J);
                Vector<double> temp;
                temp.reinit(dof_handler.n_dofs());
                system_M2.vmult(temp,solution);
                system_rhs+=temp;
#else
                system_M2.vmult(system_rhs, solution);
#endif
                left_bound.set_time(time);
                right_bound.set_time(time);
                
                
                {
                        
                        std::map<types::global_dof_index,double> boundary_values;
                        VectorTools::interpolate_boundary_values (dof_handler,
                                                                  0,
                                                                  left_bound,
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
         
		 {
		  DataOut<1> data_out;
		  std::string name("step-");
		  name.append(to_string(Step));
		  data_out.attach_dof_handler (dof_handler);
		  data_out.add_data_vector (solution, "end");

		  data_out.build_patches ();
		  name.append(".gpl");
		  std::ofstream output (name);
		  data_out.write_gnuplot (output);
		  }
         
         
                
        }
        
	{
	 DataOut<1> data_out;

	 data_out.attach_dof_handler (dof_handler);
	 data_out.add_data_vector (solution, "end");

	 data_out.build_patches ();

	 std::ofstream output ("end.gpl");
	 data_out.write_gnuplot (output);
	 }
        
	ran=true;
        
#ifdef __MATLAB__
	ofstream print;
        print.open("solution.m");
        
        if (print.is_open()) {
                print<<"x=[ ";
                for (int i=0; i<grid_points.size()-1; ++i) {
                        print<<par.S0*exp(grid_points[i][0])<<"; ";
                }
                print<<par.S0*exp(grid_points[grid_points.size()-1][0])<<" ];\n";
                
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
	
        // Creo nuova grigla ( che passi da 0 )
//         Triangulation<dim> price;
        // Creo degli fe
//         FE_Q<dim> fe3 (1);
        // Creo un DoFHandler e lo attacco a price
//         DoFHandler<dim> dof_handler_3 (price);
        // Costruisco la griglia, in modo che passi da 0 e non la rifinisco
//         GridGenerator::hyper_rectangle(price, Point<dim> (0.), Point<dim> (xmax));
        // Assegno a dof_handler_3 gli elementi finit fe3 appena creati
//         dof_handler_3.distribute_dofs(fe3);
        // Definisco questa fantomatica funzione FEFieldFunction
        Functions::FEFieldFunction<dim> fe_function (dof_handler, solution);
        // Creo il vettore che conterrà i valori interpolati
//         Vector<double> solution_vector(2);
        // Interpolo
//         VectorTools::interpolate(dof_handler_3, fe_function, solution_vector);
        // Ritorno il valore interpolato della soluzione in 0
        Point<dim> p(0.);
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
        
        Opzione<1> Call(par, 100, 10);
        double Prezzo=Call.run();
        cout<<"Prezzo "<<Prezzo<<"\n";
        
        /*
	// tempo // spazio
        // 4 a 9 *100 time_step
        const int top=9-4+1;
        double T[top], ratio[top], result[top], real_T[top];
	
	clock_t inizio,fine;
        struct timeval start, end;
        	
        for (int i=0; i<top; i++) {
                
                Opzione<1> Call(par, 10, i+4);
                
                gettimeofday(&start, NULL);
                inizio=clock();
                Call.run();
                gettimeofday(&end, NULL);
                fine=clock();
                
                result[i]=Call.get_price();
                
                T[i]=static_cast<double> (((fine-inizio)*1.e6)/CLOCKS_PER_SEC);
                real_T[i]=((end.tv_sec  - start.tv_sec) * 1000000u + 
                           end.tv_usec - start.tv_usec);
                
        }
        
        cout<<"Results for 100 time iterations:\n";
	for (int i=0; i<top; ++i) {
                cout<<"Grid\t"<<pow(2,i+4)<<"\tPrice\t"<<result[i]<<"\tclocktime\t"<<
                T[i]/1e6<<" s\trealtime\t"<<real_T[i]/1e6<<"\n";
        }*/
	cout<<"Kou-1d\n";
        
	return 0;
}

// Laguerre stuff
//****************************************************************************80

void cdgqf ( int nt, int kind, double alpha, double beta, double t[], 
            double wts[] )

//****************************************************************************80
//
//  Purpose:
//
//    CDGQF computes a Gauss quadrature formula with default A, B and simple knots.
//
//  Discussion:
//
//    This routine computes all the knots and weights of a Gauss quadrature
//    formula with a classical weight function with default values for A and B,
//    and only simple knots.
//
//    There are no moments checks and no printing is done.
//
//    Use routine EIQFS to evaluate a quadrature computed by CGQFS.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    08 January 2010
//
//  Author:
//
//    Original FORTRAN77 version by Sylvan Elhay, Jaroslav Kautsky.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Sylvan Elhay, Jaroslav Kautsky,
//    Algorithm 655: IQPACK, FORTRAN Subroutines for the Weights of 
//    Interpolatory Quadrature,
//    ACM Transactions on Mathematical Software,
//    Volume 13, Number 4, December 1987, pages 399-415.
//
//  Parameters:
//
//    Input, int NT, the number of knots.
//
//    Input, int KIND, the rule.
//    1, Legendre,             (a,b)       1.0
//    2, Chebyshev,            (a,b)       ((b-x)*(x-a))^(-0.5)
//    3, Gegenbauer,           (a,b)       ((b-x)*(x-a))^alpha
//    4, Jacobi,               (a,b)       (b-x)^alpha*(x-a)^beta
//    5, Generalized Laguerre, (a,inf)     (x-a)^alpha*exp(-b*(x-a))
//    6, Generalized Hermite,  (-inf,inf)  |x-a|^alpha*exp(-b*(x-a)^2)
//    7, Exponential,          (a,b)       |x-(a+b)/2.0|^alpha
//    8, Rational,             (a,inf)     (x-a)^alpha*(x+b)^beta
//
//    Input, double ALPHA, the value of Alpha, if needed.
//
//    Input, double BETA, the value of Beta, if needed.
//
//    Output, double T[NT], the knots.
//
//    Output, double WTS[NT], the weights.
//
{
        double *aj;
        double *bj;
        double zemu;
        
        parchk ( kind, 2 * nt, alpha, beta );
        //
        //  Get the Jacobi matrix and zero-th moment.
        //
        aj = new double[nt];
        bj = new double[nt];
        
        zemu = class_matrix ( kind, nt, alpha, beta, aj, bj );
        //
        //  Compute the knots and weights.
        //
        sgqf ( nt, aj, bj, zemu, t, wts );
        
        delete [] aj;
        delete [] bj;
        
        return;
}
//****************************************************************************80

void cgqf ( int nt, int kind, double alpha, double beta, double a, double b, 
           double t[], double wts[] )

//****************************************************************************80
//
//  Purpose:
//
//    CGQF computes knots and weights of a Gauss quadrature formula.
//
//  Discussion:
//
//    The user may specify the interval (A,B).
//
//    Only simple knots are produced.
//
//    Use routine EIQFS to evaluate this quadrature formula.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    16 February 2010
//
//  Author:
//
//    Original FORTRAN77 version by Sylvan Elhay, Jaroslav Kautsky.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Sylvan Elhay, Jaroslav Kautsky,
//    Algorithm 655: IQPACK, FORTRAN Subroutines for the Weights of 
//    Interpolatory Quadrature,
//    ACM Transactions on Mathematical Software,
//    Volume 13, Number 4, December 1987, pages 399-415.
//
//  Parameters:
//
//    Input, int NT, the number of knots.
//
//    Input, int KIND, the rule.
//    1, Legendre,             (a,b)       1.0
//    2, Chebyshev Type 1,     (a,b)       ((b-x)*(x-a))^-0.5)
//    3, Gegenbauer,           (a,b)       ((b-x)*(x-a))^alpha
//    4, Jacobi,               (a,b)       (b-x)^alpha*(x-a)^beta
//    5, Generalized Laguerre, (a,+oo)     (x-a)^alpha*exp(-b*(x-a))
//    6, Generalized Hermite,  (-oo,+oo)   |x-a|^alpha*exp(-b*(x-a)^2)
//    7, Exponential,          (a,b)       |x-(a+b)/2.0|^alpha
//    8, Rational,             (a,+oo)     (x-a)^alpha*(x+b)^beta
//    9, Chebyshev Type 2,     (a,b)       ((b-x)*(x-a))^(+0.5)
//
//    Input, double ALPHA, the value of Alpha, if needed.
//
//    Input, double BETA, the value of Beta, if needed.
//
//    Input, double A, B, the interval endpoints, or
//    other parameters.
//
//    Output, double T[NT], the knots.
//
//    Output, double WTS[NT], the weights.
//
{
        int i;
        int *mlt;
        int *ndx;
        //
        //  Compute the Gauss quadrature formula for default values of A and B.
        //
        cdgqf ( nt, kind, alpha, beta, t, wts );
        //
        //  Prepare to scale the quadrature formula to other weight function with 
        //  valid A and B.
        //
        mlt = new int[nt];
        for ( i = 0; i < nt; i++ )
        {
                mlt[i] = 1;
        }
        ndx = new int[nt];
        for ( i = 0; i < nt; i++ )
        {
                ndx[i] = i + 1;
        }
        scqf ( nt, t, mlt, wts, nt, ndx, wts, t, kind, alpha, beta, a, b );
        
        delete [] mlt;
        delete [] ndx;
        
        return;
}
//****************************************************************************80

double class_matrix ( int kind, int m, double alpha, double beta, double aj[], 
                     double bj[] )

//****************************************************************************80
//
//  Purpose:
//
//    CLASS_MATRIX computes the Jacobi matrix for a quadrature rule.
//
//  Discussion:
//
//    This routine computes the diagonal AJ and sub-diagonal BJ
//    elements of the order M tridiagonal symmetric Jacobi matrix
//    associated with the polynomials orthogonal with respect to
//    the weight function specified by KIND.
//
//    For weight functions 1-7, M elements are defined in BJ even
//    though only M-1 are needed.  For weight function 8, BJ(M) is
//    set to zero.
//
//    The zero-th moment of the weight function is returned in ZEMU.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    08 January 2010
//
//  Author:
//
//    Original FORTRAN77 version by Sylvan Elhay, Jaroslav Kautsky.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Sylvan Elhay, Jaroslav Kautsky,
//    Algorithm 655: IQPACK, FORTRAN Subroutines for the Weights of 
//    Interpolatory Quadrature,
//    ACM Transactions on Mathematical Software,
//    Volume 13, Number 4, December 1987, pages 399-415.
//
//  Parameters:
//
//    Input, int KIND, the rule.
//    1, Legendre,             (a,b)       1.0
//    2, Chebyshev,            (a,b)       ((b-x)*(x-a))^(-0.5)
//    3, Gegenbauer,           (a,b)       ((b-x)*(x-a))^alpha
//    4, Jacobi,               (a,b)       (b-x)^alpha*(x-a)^beta
//    5, Generalized Laguerre, (a,inf)     (x-a)^alpha*exp(-b*(x-a))
//    6, Generalized Hermite,  (-inf,inf)  |x-a|^alpha*exp(-b*(x-a)^2)
//    7, Exponential,          (a,b)       |x-(a+b)/2.0|^alpha
//    8, Rational,             (a,inf)     (x-a)^alpha*(x+b)^beta
//
//    Input, int M, the order of the Jacobi matrix.
//
//    Input, double ALPHA, the value of Alpha, if needed.
//
//    Input, double BETA, the value of Beta, if needed.
//
//    Output, double AJ[M], BJ[M], the diagonal and subdiagonal
//    of the Jacobi matrix.
//
//    Output, double CLASS_MATRIX, the zero-th moment.
//
{
        double a2b2;
        double ab;
        double aba;
        double abi;
        double abj;
        double abti;
        double apone;
        int i;
        double pi = 3.14159265358979323846264338327950;
        double temp;
        double temp2;
        double zemu;
        
        temp = r8_epsilon ( );
        
        parchk ( kind, 2 * m - 1, alpha, beta );
        
        temp2 = 0.5;
        
        if ( 500.0 * temp < r8_abs ( pow ( tgamma ( temp2 ), 2 ) - pi ) )
        {
                cout << "\n";
                cout << "CLASS_MATRIX - Fatal error!\n";
                cout << "  Gamma function does not match machine parameters.\n";
                exit ( 1 );
        }
        
        if ( kind == 1 )
        {
                ab = 0.0;
                
                zemu = 2.0 / ( ab + 1.0 );
                
                for ( i = 0; i < m; i++ )
                {
                        aj[i] = 0.0;
                }
                
                for ( i = 1; i <= m; i++ )
                {
                        abi = i + ab * ( i % 2 );
                        abj = 2 * i + ab;
                        bj[i-1] = sqrt ( abi * abi / ( abj * abj - 1.0 ) );
                }
        }
        else if ( kind == 2 )
        {
                zemu = pi;
                
                for ( i = 0; i < m; i++ )
                {
                        aj[i] = 0.0;
                }
                
                bj[0] =  sqrt ( 0.5 );
                for ( i = 1; i < m; i++ )
                {
                        bj[i] = 0.5;
                }
        }
        else if ( kind == 3 )
        {
                ab = alpha * 2.0;
                zemu = pow ( 2.0, ab + 1.0 ) * pow ( tgamma ( alpha + 1.0 ), 2 )
                / tgamma ( ab + 2.0 );
                
                for ( i = 0; i < m; i++ )
                {
                        aj[i] = 0.0;
                }
                
                bj[0] = sqrt ( 1.0 / ( 2.0 * alpha + 3.0 ) );
                for ( i = 2; i <= m; i++ )
                {
                        bj[i-1] = sqrt ( i * ( i + ab ) / ( 4.0 * pow ( i + alpha, 2 ) - 1.0 ) );
                }
        }
        else if ( kind == 4 )
        {
                ab = alpha + beta;
                abi = 2.0 + ab;
                zemu = pow ( 2.0, ab + 1.0 ) * tgamma ( alpha + 1.0 ) 
                * tgamma ( beta + 1.0 ) / tgamma ( abi );
                aj[0] = ( beta - alpha ) / abi;
                bj[0] = sqrt ( 4.0 * ( 1.0 + alpha ) * ( 1.0 + beta ) 
                              / ( ( abi + 1.0 ) * abi * abi ) );
                a2b2 = beta * beta - alpha * alpha;
                
                for ( i = 2; i <= m; i++ )
                {
                        abi = 2.0 * i + ab;
                        aj[i-1] = a2b2 / ( ( abi - 2.0 ) * abi );
                        abi = abi * abi;
                        bj[i-1] = sqrt ( 4.0 * i * ( i + alpha ) * ( i + beta ) * ( i + ab ) 
                                        / ( ( abi - 1.0 ) * abi ) );
                }
        }
        else if ( kind == 5 )
        {
                zemu = tgamma ( alpha + 1.0 );
                
                for ( i = 1; i <= m; i++ )
                {
                        aj[i-1] = 2.0 * i - 1.0 + alpha;
                        bj[i-1] = sqrt ( i * ( i + alpha ) );
                }
        }
        else if ( kind == 6 )
        {
                zemu = tgamma ( ( alpha + 1.0 ) / 2.0 );
                
                for ( i = 0; i < m; i++ )
                {
                        aj[i] = 0.0;
                }
                
                for ( i = 1; i <= m; i++ )
                {
                        bj[i-1] = sqrt ( ( i + alpha * ( i % 2 ) ) / 2.0 );
                }
        }
        else if ( kind == 7 )
        {
                ab = alpha;
                zemu = 2.0 / ( ab + 1.0 );
                
                for ( i = 0; i < m; i++ )
                {
                        aj[i] = 0.0;
                }
                
                for ( i = 1; i <= m; i++ )
                {
                        abi = i + ab * ( i % 2 );
                        abj = 2 * i + ab;
                        bj[i-1] = sqrt ( abi * abi / ( abj * abj - 1.0 ) );
                }
        }
        else if ( kind == 8 )
        {
                ab = alpha + beta;
                zemu = tgamma ( alpha + 1.0 ) * tgamma ( - ( ab + 1.0 ) ) 
                / tgamma ( - beta );
                apone = alpha + 1.0;
                aba = ab * apone;
                aj[0] = - apone / ( ab + 2.0 );
                bj[0] = - aj[0] * ( beta + 1.0 ) / ( ab + 2.0 ) / ( ab + 3.0 );
                for ( i = 2; i <= m; i++ )
                {
                        abti = ab + 2.0 * i;
                        aj[i-1] = aba + 2.0 * ( ab + i ) * ( i - 1 );
                        aj[i-1] = - aj[i-1] / abti / ( abti - 2.0 );
                }
                
                for ( i = 2; i <= m - 1; i++ )
                {
                        abti = ab + 2.0 * i;
                        bj[i-1] = i * ( alpha + i ) / ( abti - 1.0 ) * ( beta + i ) 
                        / ( abti * abti ) * ( ab + i ) / ( abti + 1.0 );
                }
                bj[m-1] = 0.0;
                for ( i = 0; i < m; i++ )
                {
                        bj[i] =  sqrt ( bj[i] );
                }
        }
        
        return zemu;
}
//****************************************************************************80

void imtqlx ( int n, double d[], double e[], double z[] )

//****************************************************************************80
//
//  Purpose:
//
//    IMTQLX diagonalizes a symmetric tridiagonal matrix.
//
//  Discussion:
//
//    This routine is a slightly modified version of the EISPACK routine to 
//    perform the implicit QL algorithm on a symmetric tridiagonal matrix. 
//
//    The authors thank the authors of EISPACK for permission to use this
//    routine. 
//
//    It has been modified to produce the product Q' * Z, where Z is an input 
//    vector and Q is the orthogonal matrix diagonalizing the input matrix.  
//    The changes consist (essentialy) of applying the orthogonal transformations
//    directly to Z as they are generated.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    08 January 2010
//
//  Author:
//
//    Original FORTRAN77 version by Sylvan Elhay, Jaroslav Kautsky.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Sylvan Elhay, Jaroslav Kautsky,
//    Algorithm 655: IQPACK, FORTRAN Subroutines for the Weights of 
//    Interpolatory Quadrature,
//    ACM Transactions on Mathematical Software,
//    Volume 13, Number 4, December 1987, pages 399-415.
//
//    Roger Martin, James Wilkinson,
//    The Implicit QL Algorithm,
//    Numerische Mathematik,
//    Volume 12, Number 5, December 1968, pages 377-383.
//
//  Parameters:
//
//    Input, int N, the order of the matrix.
//
//    Input/output, double D(N), the diagonal entries of the matrix.
//    On output, the information in D has been overwritten.
//
//    Input/output, double E(N), the subdiagonal entries of the 
//    matrix, in entries E(1) through E(N-1).  On output, the information in
//    E has been overwritten.
//
//    Input/output, double Z(N).  On input, a vector.  On output,
//    the value of Q' * Z, where Q is the matrix that diagonalizes the
//    input symmetric tridiagonal matrix.
//
{
        double b;
        double c;
        double f;
        double g;
        int i;
        int ii;
        int itn = 30;
        int j;
        int k;
        int l;
        int m;
        int mml;
        double p;
        double prec;
        double r;
        double s;
        
        prec = r8_epsilon ( );
        
        if ( n == 1 )
        {
                return;
        }
        
        e[n-1] = 0.0;
        
        for ( l = 1; l <= n; l++ )
        {
                j = 0;
                for ( ; ; )
                {
                        for ( m = l; m <= n; m++ )
                        {
                                if ( m == n )
                                {
                                        break;
                                }
                                
                                if ( r8_abs ( e[m-1] ) <= prec * ( r8_abs ( d[m-1] ) + r8_abs ( d[m] ) ) )
                                {
                                        break;
                                }
                        }
                        p = d[l-1];
                        if ( m == l )
                        {
                                break;
                        }
                        if ( itn <= j )
                        {
                                cout << "\n";
                                cout << "IMTQLX - Fatal error!\n";
                                cout << "  Iteration limit exceeded\n";
                                exit ( 1 );
                        }
                        j = j + 1;
                        g = ( d[l] - p ) / ( 2.0 * e[l-1] );
                        r =  sqrt ( g * g + 1.0 );
                        g = d[m-1] - p + e[l-1] / ( g + r8_abs ( r ) * r8_sign ( g ) );
                        s = 1.0;
                        c = 1.0;
                        p = 0.0;
                        mml = m - l;
                        
                        for ( ii = 1; ii <= mml; ii++ )
                        {
                                i = m - ii;
                                f = s * e[i-1];
                                b = c * e[i-1];
                                
                                if ( r8_abs ( g ) <= r8_abs ( f ) )
                                {
                                        c = g / f;
                                        r =  sqrt ( c * c + 1.0 );
                                        e[i] = f * r;
                                        s = 1.0 / r;
                                        c = c * s;
                                }
                                else
                                {
                                        s = f / g;
                                        r =  sqrt ( s * s + 1.0 );
                                        e[i] = g * r;
                                        c = 1.0 / r;
                                        s = s * c;
                                }
                                g = d[i] - p;
                                r = ( d[i-1] - g ) * s + 2.0 * c * b;
                                p = s * r;
                                d[i] = g + p;
                                g = c * r - b;
                                f = z[i];
                                z[i] = s * z[i-1] + c * f;
                                z[i-1] = c * z[i-1] - s * f;
                        }
                        d[l-1] = d[l-1] - p;
                        e[l-1] = g;
                        e[m-1] = 0.0;
                }
        }
        //
        //  Sorting.
        //
        for ( ii = 2; ii <= m; ii++ )
        {
                i = ii - 1;
                k = i;
                p = d[i-1];
                
                for ( j = ii; j <= n; j++ )
                {
                        if ( d[j-1] < p )
                        {
                                k = j;
                                p = d[j-1];
                        }
                }
                
                if ( k != i )
                {
                        d[k-1] = d[i-1];
                        d[i-1] = p;
                        p = z[i-1];
                        z[i-1] = z[k-1];
                        z[k-1] = p;
                }
        }
        return;
}
//****************************************************************************80

void parchk ( int kind, int m, double alpha, double beta )

//****************************************************************************80
//
//  Purpose:
//
//    PARCHK checks parameters ALPHA and BETA for classical weight functions. 
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    07 January 2010
//
//  Author:
//
//    Original FORTRAN77 version by Sylvan Elhay, Jaroslav Kautsky.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Sylvan Elhay, Jaroslav Kautsky,
//    Algorithm 655: IQPACK, FORTRAN Subroutines for the Weights of 
//    Interpolatory Quadrature,
//    ACM Transactions on Mathematical Software,
//    Volume 13, Number 4, December 1987, pages 399-415.
//
//  Parameters:
//
//    Input, int KIND, the rule.
//    1, Legendre,             (a,b)       1.0
//    2, Chebyshev,            (a,b)       ((b-x)*(x-a))^(-0.5)
//    3, Gegenbauer,           (a,b)       ((b-x)*(x-a))^alpha
//    4, Jacobi,               (a,b)       (b-x)^alpha*(x-a)^beta
//    5, Generalized Laguerre, (a,inf)     (x-a)^alpha*exp(-b*(x-a))
//    6, Generalized Hermite,  (-inf,inf)  |x-a|^alpha*exp(-b*(x-a)^2)
//    7, Exponential,          (a,b)       |x-(a+b)/2.0|^alpha
//    8, Rational,             (a,inf)     (x-a)^alpha*(x+b)^beta
//
//    Input, int M, the order of the highest moment to
//    be calculated.  This value is only needed when KIND = 8.
//
//    Input, double ALPHA, BETA, the parameters, if required
//    by the value of KIND.
//
{
        double tmp;
        
        if ( kind <= 0 )
        {
                cout << "\n";
                cout << "PARCHK - Fatal error!\n";
                cout << "  KIND <= 0.\n";
                exit ( 1 );
        }
        //
        //  Check ALPHA for Gegenbauer, Jacobi, Laguerre, Hermite, Exponential.
        //
        if ( 3 <= kind && alpha <= -1.0 )
        {
                cout << "\n";
                cout << "PARCHK - Fatal error!\n";
                cout << "  3 <= KIND and ALPHA <= -1.\n";
                exit ( 1 );
        }
        //
        //  Check BETA for Jacobi.
        //
        if ( kind == 4 && beta <= -1.0 )
        {
                cout << "\n";
                cout << "PARCHK - Fatal error!\n";
                cout << "  KIND == 4 and BETA <= -1.0.\n";
                exit ( 1 );
        }
        //
        //  Check ALPHA and BETA for rational.
        //
        if ( kind == 8 )
        {
                tmp = alpha + beta + m + 1.0;
                if ( 0.0 <= tmp || tmp <= beta )
                {
                        cout << "\n";
                        cout << "PARCHK - Fatal error!\n";
                        cout << "  KIND == 8 but condition on ALPHA and BETA fails.\n";
                        exit ( 1 );
                }
        }
        return;
}
//****************************************************************************80

double r8_abs ( double x )

//****************************************************************************80
//
//  Purpose:
//
//    R8_ABS returns the absolute value of an R8.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    14 November 2006
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double X, the quantity whose absolute value is desired.
//
//    Output, double R8_ABS, the absolute value of X.
//
{
        double value;
        
        if ( 0.0 <= x )
        {
                value = x;
        } 
        else
        {
                value = -x;
        }
        return value;
}
//****************************************************************************80

double r8_epsilon ( )

//****************************************************************************80
//
//  Purpose:
//
//    R8_EPSILON returns the R8 roundoff unit.
//
//  Discussion:
//
//    The roundoff unit is a number R which is a power of 2 with the 
//    property that, to the precision of the computer's arithmetic,
//      1 < 1 + R
//    but 
//      1 = ( 1 + R / 2 )
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    01 July 2004
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Output, double R8_EPSILON, the R8 round-off unit.
//
{
        double value;
        
        value = 1.0;
        
        while ( 1.0 < ( double ) ( 1.0 + value )  )
        {
                value = value / 2.0;
        }
        
        value = 2.0 * value;
        
        return value;
}
//****************************************************************************80

double r8_huge ( )

//****************************************************************************80
//
//  Purpose:
//
//    R8_HUGE returns a "huge" R8.
//
//  Discussion:
//
//    The value returned by this function is NOT required to be the
//    maximum representable R8.  This value varies from machine to machine,
//    from compiler to compiler, and may cause problems when being printed.
//    We simply want a "very large" but non-infinite number.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    06 October 2007
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Output, double R8_HUGE, a "huge" R8 value.
//
{
        double value;
        
        value = 1.0E+30;
        
        return value;
}
//****************************************************************************80

double r8_sign ( double x )

//****************************************************************************80
//
//  Purpose:
//
//    R8_SIGN returns the sign of an R8.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    18 October 2004
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double X, the number whose sign is desired.
//
//    Output, double R8_SIGN, the sign of X.
//
{
        double value;
        
        if ( x < 0.0 )
        {
                value = -1.0;
        } 
        else
        {
                value = 1.0;
        }
        return value;
}
//****************************************************************************80

void r8mat_write ( string output_filename, int m, int n, double table[] )

//****************************************************************************80
//
//  Purpose:
//
//    R8MAT_WRITE writes an R8MAT file with no header.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    29 June 2009
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, string OUTPUT_FILENAME, the output filename.
//
//    Input, int M, the spatial dimension.
//
//    Input, int N, the number of points.
//
//    Input, double TABLE[M*N], the table data.
//
{
        int i;
        int j;
        ofstream output;
        //
        //  Open the file.
        //
        output.open ( output_filename.c_str ( ) );
        
        if ( !output )
        {
                cerr << "\n";
                cerr << "R8MAT_WRITE - Fatal error!\n";
                cerr << "  Could not open the output file.\n";
                return;
        }
        //
        //  Write the data.
        //
        for ( j = 0; j < n; j++ )
        {
                for ( i = 0; i < m; i++ )
                {
                        output << "  " << setw(24) << setprecision(16) << table[i+j*m];
                }
                output << "\n";
        }
        //
        //  Close the file.
        //
        output.close ( );
        
        return;
}
//****************************************************************************80

void rule_write ( int order, string filename, double x[], double w[], 
                 double r[] )

//****************************************************************************80
//
//  Purpose:
//
//    RULE_WRITE writes a quadrature rule to three files.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    18 February 2010
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int ORDER, the order of the rule.
//
//    Input, double A, the left endpoint.
//
//    Input, double B, the right endpoint.
//
//    Input, string FILENAME, specifies the output filenames.
//    "filename_w.txt", "filename_x.txt", "filename_r.txt" 
//    defining weights, abscissas, and region.
// 
{
        string filename_r;
        string filename_w;
        string filename_x;
        int i;
        int kind;
        
        filename_w = filename + "_w.txt";
        filename_x = filename + "_x.txt";
        filename_r = filename + "_r.txt";
        
        cout << "\n";
        cout << "  Creating quadrature files.\n";
        cout << "\n";
        cout << "  Root file name is     \"" << filename   << "\".\n";
        cout << "\n";
        cout << "  Weight file will be   \"" << filename_w << "\".\n";
        cout << "  Abscissa file will be \"" << filename_x << "\".\n";
        cout << "  Region file will be   \"" << filename_r << "\".\n";
        
        r8mat_write ( filename_w, 1, order, w );
        r8mat_write ( filename_x, 1, order, x );
        r8mat_write ( filename_r, 1, 2,     r );
        
        return;
}
//****************************************************************************80

void scqf ( int nt, double t[], int mlt[], double wts[], int nwts, int ndx[], 
           double swts[], double st[], int kind, double alpha, double beta, double a, 
           double b )

//****************************************************************************80
//
//  Purpose:
//
//    SCQF scales a quadrature formula to a nonstandard interval.
//
//  Discussion:
//
//    The arrays WTS and SWTS may coincide.
//
//    The arrays T and ST may coincide.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    16 February 2010
//
//  Author:
//
//    Original FORTRAN77 version by Sylvan Elhay, Jaroslav Kautsky.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Sylvan Elhay, Jaroslav Kautsky,
//    Algorithm 655: IQPACK, FORTRAN Subroutines for the Weights of 
//    Interpolatory Quadrature,
//    ACM Transactions on Mathematical Software,
//    Volume 13, Number 4, December 1987, pages 399-415.
//
//  Parameters:
//
//    Input, int NT, the number of knots.
//
//    Input, double T[NT], the original knots.
//
//    Input, int MLT[NT], the multiplicity of the knots.
//
//    Input, double WTS[NWTS], the weights.
//
//    Input, int NWTS, the number of weights.
//
//    Input, int NDX[NT], used to index the array WTS.  
//    For more details see the comments in CAWIQ.
//
//    Output, double SWTS[NWTS], the scaled weights.
//
//    Output, double ST[NT], the scaled knots.
//
//    Input, int KIND, the rule.
//    1, Legendre,             (a,b)       1.0
//    2, Chebyshev Type 1,     (a,b)       ((b-x)*(x-a))^(-0.5)
//    3, Gegenbauer,           (a,b)       ((b-x)*(x-a))^alpha
//    4, Jacobi,               (a,b)       (b-x)^alpha*(x-a)^beta
//    5, Generalized Laguerre, (a,+oo)     (x-a)^alpha*exp(-b*(x-a))
//    6, Generalized Hermite,  (-oo,+oo)   |x-a|^alpha*exp(-b*(x-a)^2)
//    7, Exponential,          (a,b)       |x-(a+b)/2.0|^alpha
//    8, Rational,             (a,+oo)     (x-a)^alpha*(x+b)^beta
//    9, Chebyshev Type 2,     (a,b)       ((b-x)*(x-a))^(+0.5)
//
//    Input, double ALPHA, the value of Alpha, if needed.
//
//    Input, double BETA, the value of Beta, if needed.
//
//    Input, double A, B, the interval endpoints.
//
{
        double al;
        double be;
        int i;
        int k;
        int l;
        double p;
        double shft;
        double slp;
        double temp;
        double tmp;
        
        temp = r8_epsilon ( );
        
        parchk ( kind, 1, alpha, beta );
        
        if ( kind == 1 )
        {
                al = 0.0;
                be = 0.0;
                if ( r8_abs ( b - a ) <= temp )
                {
                        cout << "\n";
                        cout << "SCQF - Fatal error!\n";
                        cout << "  |B - A| too small.\n";
                        exit ( 1 );
                }
                shft = ( a + b ) / 2.0;
                slp = ( b - a ) / 2.0;
        }
        else if ( kind == 2 )
        {
                al = -0.5;
                be = -0.5;
                if ( r8_abs ( b - a ) <= temp )
                {
                        cout << "\n";
                        cout << "SCQF - Fatal error!\n";
                        cout << "  |B - A| too small.\n";
                        exit ( 1 );
                }
                shft = ( a + b ) / 2.0;
                slp = ( b - a ) / 2.0;
        }
        else if ( kind == 3 )
        {
                al = alpha;
                be = alpha;
                if ( r8_abs ( b - a ) <= temp )
                {
                        cout << "\n";
                        cout << "SCQF - Fatal error!\n";
                        cout << "  |B - A| too small.\n";
                        exit ( 1 );
                }
                shft = ( a + b ) / 2.0;
                slp = ( b - a ) / 2.0;
        }
        else if ( kind == 4 )
        {
                al = alpha;
                be = beta;
                
                if ( r8_abs ( b - a ) <= temp )
                {
                        cout << "\n";
                        cout << "SCQF - Fatal error!\n";
                        cout << "  |B - A| too small.\n";
                        exit ( 1 );
                }
                shft = ( a + b ) / 2.0;
                slp = ( b - a ) / 2.0;
        }
        else if ( kind == 5 )
        {
                if ( b <= 0.0 )
                {
                        cout << "\n";
                        cout << "SCQF - Fatal error!\n";
                        cout << "  B <= 0\n";
                        exit ( 1 );
                }
                shft = a;
                slp = 1.0 / b;
                al = alpha;
                be = 0.0;
        }
        else if ( kind == 6 )
        {
                if ( b <= 0.0 )
                {
                        cout << "\n";
                        cout << "SCQF - Fatal error!\n";
                        cout << "  B <= 0.\n";
                        exit ( 1 );
                }
                shft = a;
                slp = 1.0 / sqrt ( b );
                al = alpha;
                be = 0.0;
        }
        else if ( kind == 7 )
        {
                al = alpha;
                be = 0.0;
                if ( r8_abs ( b - a ) <= temp )
                {
                        cout << "\n";
                        cout << "SCQF - Fatal error!\n";
                        cout << "  |B - A| too small.\n";
                        exit ( 1 );
                }
                shft = ( a + b ) / 2.0;
                slp = ( b - a ) / 2.0;
        }
        else if ( kind == 8 )
        {
                if ( a + b <= 0.0 )
                {
                        cout << "\n";
                        cout << "SCQF - Fatal error!\n";
                        cout << "  A + B <= 0.\n";
                        exit ( 1 );
                }
                shft = a;
                slp = a + b;
                al = alpha;
                be = beta;
        }
        else if ( kind == 9 )
        {
                al = 0.5;
                be = 0.5;
                if ( r8_abs ( b - a ) <= temp )
                {
                        cout << "\n";
                        cout << "SCQF - Fatal error!\n";
                        cout << "  |B - A| too small.\n";
                        exit ( 1 );
                }
                shft = ( a + b ) / 2.0;
                slp = ( b - a ) / 2.0;
        }
        
        p = pow ( slp, al + be + 1.0 );
        
        for ( k = 0; k < nt; k++ )
        {
                st[k] = shft + slp * t[k];
                l = abs ( ndx[k] );
                
                if ( l != 0 )
                {
                        tmp = p;
                        for ( i = l - 1; i <= l - 1 + mlt[k] - 1; i++ )
                        {
                                swts[i] = wts[i] * tmp;
                                tmp = tmp * slp;
                        }
                }
        }
        return;
}
//****************************************************************************80

void sgqf ( int nt, double aj[], double bj[], double zemu, double t[], 
           double wts[] )

//****************************************************************************80
//
//  Purpose:
//
//    SGQF computes knots and weights of a Gauss Quadrature formula.
//
//  Discussion:
//
//    This routine computes all the knots and weights of a Gauss quadrature
//    formula with simple knots from the Jacobi matrix and the zero-th
//    moment of the weight function, using the Golub-Welsch technique.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    08 January 2010
//
//  Author:
//
//    Original FORTRAN77 version by Sylvan Elhay, Jaroslav Kautsky.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Sylvan Elhay, Jaroslav Kautsky,
//    Algorithm 655: IQPACK, FORTRAN Subroutines for the Weights of 
//    Interpolatory Quadrature,
//    ACM Transactions on Mathematical Software,
//    Volume 13, Number 4, December 1987, pages 399-415.
//
//  Parameters:
//
//    Input, int NT, the number of knots.
//
//    Input, double AJ[NT], the diagonal of the Jacobi matrix.
//
//    Input/output, double BJ[NT], the subdiagonal of the Jacobi 
//    matrix, in entries 1 through NT-1.  On output, BJ has been overwritten.
//
//    Input, double ZEMU, the zero-th moment of the weight function.
//
//    Output, double T[NT], the knots.
//
//    Output, double WTS[NT], the weights.
//
{
        int i;
        //
        //  Exit if the zero-th moment is not positive.
        //
        if ( zemu <= 0.0 )
        {
                cout << "\n";
                cout << "SGQF - Fatal error!\n";
                cout << "  ZEMU <= 0.\n";
                exit ( 1 );
        }
        //
        //  Set up vectors for IMTQLX.
        //
        for ( i = 0; i < nt; i++ )
        {
                t[i] = aj[i];
        }
        wts[0] = sqrt ( zemu );
        for ( i = 1; i < nt; i++ )
        {
                wts[i] = 0.0;
        }
        //
        //  Diagonalize the Jacobi matrix.
        //
        imtqlx ( nt, t, bj, wts );
        
        for ( i = 0; i < nt; i++ )
        {
                wts[i] = wts[i] * wts[i];
        }
        
        return;
}
//****************************************************************************80

void timestamp ( )

//****************************************************************************80
//
//  Purpose:
//
//    TIMESTAMP prints the current YMDHMS date as a time stamp.
//
//  Example:
//
//    31 May 2001 09:45:54 AM
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    08 July 2009
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    None
//
{
# define TIME_SIZE 40
        
        static char time_buffer[TIME_SIZE];
        const struct std::tm *tm_ptr;
        size_t len;
        std::time_t now;
        
        now = std::time ( NULL );
        tm_ptr = std::localtime ( &now );
        
        len = std::strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm_ptr );
        
        std::cout << time_buffer << "\n";
        
        return;
# undef TIME_SIZE
}
// laguerre stuff