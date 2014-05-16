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
#include <boost/graph/buffer_concepts.hpp>

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

#define __CALL__
// max(s1+s2-k,0)

//selfexplanatory,  just extended it
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


//added a new private variable,  that indicates with ax the density uses
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

//same,  added a private variable indicating wih ax
//TODO delete default constructor
template<int dim>
class Solution_Trimmer: public Function<dim>
{
private:
	unsigned int _ax;
	//check that this causes no memory leaks while keeping hereditariety
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
        
	//two densities that work on 2D points
	Kou_Density<dim>				k_x;
	Kou_Density<dim>				k_y;
        
	//domain triangulation 2D
	Triangulation<dim>              triangulation;
	FE_Q<dim>                       fe;
	DoFHandler<dim>                 dof_handler;
        
	//integral triangulations 1D,  and relatives FE and DoF_handlers
	Triangulation<1>        integral_triangulation_x;
	FE_Q<1>                 fe_integral_x;
	DoFHandler<1>           dof_handler_integral_x;
        
	Triangulation<1>        integral_triangulation_y;
	FE_Q<1>			fe_integral_y;
	DoFHandler<1>		dof_handler_integral_y;
	
	SparsityPattern                 sparsity_pattern;
	SparseMatrix<double>            system_matrix;
	SparseMatrix<double>            system_M2;
	SparseMatrix<double>            dd_matrix;
	SparseMatrix<double>            fd_matrix;
	SparseMatrix<double>            ff_matrix;
        
        // quadrature di laguerre
        Quadrature_Laguerre right_quad_x;
        Quadrature_Laguerre left_quad_x;
        
        std::vector<double> right_quad_nodes_x;
        std::vector<double> left_quad_nodes_x;
        std::vector<double> right_quad_weights_x;
        std::vector<double> left_quad_weights_x;
        
        std::vector<Point<1> > quadrature_points_x;
        
        Quadrature_Laguerre right_quad_y;
        Quadrature_Laguerre left_quad_y;
        
        std::vector<double> right_quad_nodes_y;
        std::vector<double> left_quad_nodes_y;
        std::vector<double> right_quad_weights_y;
        std::vector<double> left_quad_weights_y;
        
        std::vector<Point<1> > quadrature_points_y;
        
	Vector<double>                  solution;
	Vector<double>                  system_rhs;
	
	std::vector< Point<dim> >       grid_points;
	
	unsigned int refs, Nsteps;
	double time_step;
	double dx1, dx2;
	double Smin1, Smax1, Smin2, Smax2;
	double price;
	//using as points
	Point<dim> xmin, xmax;
	Point<dim> Bmin, Bmax;
	
	double alpha1,  alpha2;
	
	bool ran;
	
	//redefined these parts
	void Levy_integral_part1();
	void Levy_integral_part2(Vector<double> &J_x, Vector<double> &J_y );
	
public:
	//standard initialization.
	Opzione(Parametri2d const &par_, int Nsteps_,  int refinement):
	par(par_),
	k_x(0, par_.p1, par_.lambda1,  par_.lambda_piu_1,  par_.lambda_meno_1),
	k_y(1, par_.p2, par_.lambda2,  par_.lambda_piu_2,  par_.lambda_meno_2),
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


//we calculate both alpha 1 and 2 in two separate cycles. Since the integral
//grid is 1 dimensional,  we need to transform the 1D points of quadrature
//q_i in 2D points (q_i, 0),  or (0, q_i) since all functions work on points.

template<int dim>
void Opzione<dim>::Levy_integral_part1(){
	
	alpha1=0;
	alpha2=0;
	
        for (int i=0; i<right_quad_x.get_order(); ++i) {
                alpha1+=(exp(right_quad_nodes_x[i])-1)*par.p1*par.lambda1*par.lambda_piu_1*right_quad_weights_x[i];
        }
        
        for (int i=0; i<left_quad_x.get_order(); ++i) {
                // il - è perché i nodi sono positivi (Quadrature_Laguerre integra da 0 a \infty)
                alpha1+=(exp(-left_quad_nodes_x[i])-1)*(1-par.p1)*par.lambda1*par.lambda_meno_1*left_quad_weights_x[i];
        }
        
        for (int i=0; i<right_quad_y.get_order(); ++i) {
                alpha2+=(exp(right_quad_nodes_y[i])-1)*par.p2*par.lambda2*par.lambda_piu_2*right_quad_weights_y[i];
        }
        
        for (int i=0; i<left_quad_y.get_order(); ++i) {
                // il - è perché i nodi sono positivi (Quadrature_Laguerre integra da 0 a \infty)
                alpha2+=(exp(-left_quad_nodes_y[i])-1)*(1-par.p2)*par.lambda2*par.lambda_meno_2*left_quad_weights_y[i];
        }
        
        cout<<"alpha1 "<<alpha1<<" alpha2 "<<alpha2<<"\n";
	
}

template<int dim>
void Opzione<dim>::Levy_integral_part2(Vector<double> &J_x, Vector<double> &J_y) {
	
	//initialize
	J_x.reinit(solution.size());    // If fast is false, the vector is filled by zeros
	J_y.reinit(solution.size());
        
	unsigned int N(grid_points.size());
        
        //we need a BC in 2d
        Boundary_Condition<dim> bc(par.S01, par.S02, par.K, par.T, par.r);
	
	//we start the integration here
	{
                //we need a solution trimmer
                Solution_Trimmer<dim> func(0, &bc, &bc, dof_handler, solution, xmin, xmax);
#pragma omp parallel for
                for (int it=0; it<J_x.size(); ++it) {
                        
                        // and some vectors
                        std::vector< Point<dim> > quad_points(left_quad_x.get_order()+right_quad_x.get_order());
                        std::vector<double> f_u(left_quad_x.get_order()+right_quad_x.get_order());
                        
                        // Inserisco in quad_points tutti i punti di quadrature shiftati
                        for (int i=0; i<quad_points.size(); ++i) {
                                quad_points[i][0]=quadrature_points_x[i][0] + grid_points[it][0];
                                quad_points[i][1]=grid_points[it][1];
                        }
                        
                        // valuto f_u in quad_points
                        func.value_list(quad_points, f_u);
                        
                        // Integro dividendo fra parte sinistra e parte destra dell'integrale
                        for (int i=0; i<left_quad_x.get_order(); ++i) {
                                J_x(it)+=f_u[i]*(1-par.p1)*par.lambda1*par.lambda_meno_1*left_quad_weights_x[i];
                        }
                        for (int i=0; i<right_quad_x.get_order(); ++i) {
                                J_x(it)+=f_u[i+left_quad_x.get_order()]*par.p1*par.lambda1
                                *par.lambda_piu_1*right_quad_weights_x[i];
                        }
                        
                }
                
	}
        // here we do the same but inverting x and y
	{
                
                Solution_Trimmer<dim> func(1, &bc, &bc, dof_handler, solution, xmin, xmax);
#pragma omp parallel for
                for (int it=0; it<J_y.size(); ++it) {
                        
                        std::vector< Point<dim> > quad_points(left_quad_y.get_order()+right_quad_y.get_order());
                        std::vector<double> f_u(left_quad_y.get_order()+right_quad_y.get_order());
                        
                        // Inserisco in quad_points tutti i punti di quadrature shiftati
                        for (int i=0; i<quad_points.size(); ++i) {
                                quad_points[i][0]=grid_points[it][0];
                                quad_points[i][1]=quadrature_points_y[i][0] + grid_points[it][1];
                        }
                        
                        // valuto f_u in quad_points
                        func.value_list(quad_points, f_u);
                        
                        // Integro dividendo fra parte sinistra e parte destra dell'integrale
                        for (int i=0; i<left_quad_y.get_order(); ++i) {
                                J_y(it)+=f_u[i]*(1-par.p2)*par.lambda2*par.lambda_meno_2*left_quad_weights_y[i];
                        }
                        for (int i=0; i<right_quad_y.get_order(); ++i) {
                                J_y(it)+=f_u[i+left_quad_y.get_order()]*par.p2*par.lambda2
                                *par.lambda_piu_2*right_quad_weights_y[i];
                        }
                        
                }
        }
	
}


//   here I've changed double to points,  makes it easier in some parts
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
        
        // we can then create all triangulations
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
        
        // Costruisco punti e nodi di Laguerre una volta per tutte (tanto non cambiano)
        right_quad_x=Quadrature_Laguerre(static_cast<unsigned>(round(Bmax[0]/dx1)/10), par.lambda_piu_1);
        left_quad_x=Quadrature_Laguerre(static_cast<unsigned>(round(-Bmin[0]/dx1)/10), par.lambda_meno_1);
        
        // Costruisco i vettori dei nodi e dei pesi per la parte destra e sinistra
        right_quad_nodes_x=right_quad_x.get_nodes();
        right_quad_weights_x=right_quad_x.get_weights();
        
        left_quad_nodes_x=left_quad_x.get_nodes();
        left_quad_weights_x=left_quad_x.get_weights();
        
        quadrature_points_x=std::vector<Point<1> > (left_quad_x.get_order()+right_quad_x.get_order());
        
        // Costruisco un unico vettore con tutti i nodi di quadratura (quelli di sinistra cambiati di segno)
        for (int i=0; i<left_quad_x.get_order(); ++i) {
                quadrature_points_x[i]=static_cast< Point<1> > (-left_quad_nodes_x[i]);
        }
        for (int i=0; i<right_quad_x.get_order(); ++i) {
                quadrature_points_x[i+left_quad_x.get_order()]=static_cast< Point<1> > (right_quad_nodes_x[i]);
        }
        
        // Costruisco punti e nodi di Laguerre una volta per tutte (tanto non cambiano)
        right_quad_y=Quadrature_Laguerre(static_cast<unsigned>(round(Bmax[1]/dx2)/10), par.lambda_piu_2);
        left_quad_y=Quadrature_Laguerre(static_cast<unsigned>(round(-Bmin[1]/dx2)/10), par.lambda_meno_2);
        
        // Costruisco i vettori dei nodi e dei pesi per la parte destra e sinistra
        right_quad_nodes_y=right_quad_y.get_nodes();
        right_quad_weights_y=right_quad_y.get_weights();
        
        left_quad_nodes_y=left_quad_y.get_nodes();
        left_quad_weights_y=left_quad_y.get_weights();
        
        quadrature_points_y=std::vector<Point<1> > (left_quad_y.get_order()+right_quad_y.get_order());
        
        // Costruisco un unico vettore con tutti i nodi di quadratura (quelli di sinistra cambiati di segno)
        for (int i=0; i<left_quad_y.get_order(); ++i) {
                quadrature_points_y[i]=static_cast< Point<1> > (-left_quad_nodes_y[i]);
        }
        for (int i=0; i<right_quad_y.get_order(); ++i) {
                quadrature_points_y[i+left_quad_y.get_order()]=static_cast< Point<1> > (right_quad_nodes_y[i]);
        }
        
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
                cout<< "Step "<< Step<<"\t at time \t"<< time << endl;
                
                // 	 pretty much the same: we calculate J_x and J_y
                Vector<double> J_x, J_y;
                Levy_integral_part2(J_x, J_y);
                
                // 	 And we use the same old way
                //   M2*old_solution+FF*J_x+FF*J_y to calculate rhs
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
                
#ifdef __VERBOSE__
                if (time==par.T-time_step) {
                        system_matrix.print(cout);
                }
#endif
                
                SparseDirectUMFPACK solver;
                solver.initialize(sparsity_pattern);
                solver.factorize(system_matrix);
                solver.solve(system_rhs);
                
                solution=system_rhs;
                
                DataOut<2> data_out;
                
                data_out.attach_dof_handler (dof_handler);
                data_out.add_data_vector (solution, "end");
                
                data_out.build_patches ();
                
                std::string name("plot/step-");
                name.append(to_string(Step));
                name.append(".gpl");
                std::ofstream output (name);
                data_out.write_gnuplot (output);
                
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
        
        //    No need to create more poits since xmin and xmax are already points
	
	// Creo nuova grigla ( che passi da (0,0) )
	Triangulation<dim> price;
	// Creo degli fe
	FE_Q<dim> fe2 (1);
	// Creo un DoFHandler e lo attacco a price
	DoFHandler<dim> dof_handler_2 (price);
	// Costruisco la griglia, in modo che passi da (0,0) e non la rifinisco
	GridGenerator::hyper_rectangle(price, Point<dim> (0.,0.), xmax);
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
        
	par.T=1.;
	par.K=200;
	par.S01=80;
	par.S02=120;
	par.r=0.1;
	par.sigma1=0.1256;
	par.sigma2=0.2;
	par.ro=-0.2;
        
	// Parametri della parte salto
	par.p1=0.20761;                                      // Parametro 1 Kou
	par.lambda1=0.330966;                                // Parametro 2 Kou
	par.lambda_piu_1=9.65997;                             // Parametro 3 Kou
	par.lambda_meno_1=3.13868;                            // Parametro 4 Kou
        
	// Parametri della parte salto
        
	par.p2=0.20761;                                            // Parametro 1 Kou
	par.lambda2=0.330966;                                      // Parametro 2 Kou
	par.lambda_piu_2=9.65997;                                  // Parametro 3 Kou
	par.lambda_meno_2=3.13868;                                 // Parametro 4 Kou
        
        // tempo // spazio
        // 4 a 7 * 10/50/100 time_step
        const int top=7-4+1;
        double T[3][top], ratio[3][top], result[3][top], real_T[3][top];
	
	clock_t inizio,fine;
        struct timeval start, end;
        
        for (int i=0; i<top; i++) {
                
                {
                        
                        Opzione<2> Call(par, 10, i+4);
                        
                        gettimeofday(&start, NULL);
                        inizio=clock();
                        Call.run();
                        gettimeofday(&end, NULL);
                        fine=clock();
                        
                        result[0][i]=Call.get_price();
                        
                        T[0][i]=static_cast<double> (((fine-inizio)*1.e6)/CLOCKS_PER_SEC);
                        real_T[0][i]=((end.tv_sec  - start.tv_sec) * 1000000u + 
                                      end.tv_usec - start.tv_usec);
                        
                }
                
                {
                        
                        Opzione<2> Call(par, 50, i+4);
                        
                        gettimeofday(&start, NULL);
                        inizio=clock();
                        Call.run();
                        gettimeofday(&end, NULL);
                        fine=clock();
                        
                        result[1][i]=Call.get_price();
                        
                        T[1][i]=static_cast<double> (((fine-inizio)*1.e6)/CLOCKS_PER_SEC);
                        real_T[1][i]=((end.tv_sec  - start.tv_sec) * 1000000u + 
                                   end.tv_usec - start.tv_usec);
                        
                }
                
                {
                        
                        Opzione<2> Call(par, 100, i+4);
                        
                        gettimeofday(&start, NULL);
                        inizio=clock();
                        Call.run();
                        gettimeofday(&end, NULL);
                        fine=clock();
                        
                        result[2][i]=Call.get_price();
                        
                        T[2][i]=static_cast<double> (((fine-inizio)*1.e6)/CLOCKS_PER_SEC);
                        real_T[2][i]=((end.tv_sec  - start.tv_sec) * 1000000u + 
                                   end.tv_usec - start.tv_usec);
                        
                }
                
        }
        
        cout<<"Results for 10 time iterations:\n";
	for (int i=0; i<top; ++i) {
                cout<<"Grid\t"<<pow(2,2*i+8)<<"\tPrice\t"<<result[0][i]<<"\tclocktime\t"<<
                T[0][i]/1e6<<" s\trealtime\t"<<real_T[0][i]/1e6<<"s\n";
        }
        cout<<"Results for 50 time iterations:\n";
	for (int i=0; i<top; ++i) {
                cout<<"Grid\t"<<pow(2,2*i+8)<<"\tPrice\t"<<result[1][i]<<"\tclocktime\t"<<
                T[1][i]/1e6<<" s\trealtime\t"<<real_T[1][i]/1e6<<"s\n";
        }
        cout<<"Results for 100 time iterations:\n";
	for (int i=0; i<top; ++i) {
                cout<<"Grid\t"<<pow(2,2*i+8)<<"\tPrice\t"<<result[2][i]<<"\tclocktime\t"<<
                T[2][i]/1e6<<" s\trealtime\t"<<real_T[2][i]/1e6<<"s\n";
        }
        
        cout<<"Kou 2d SpeedTest\n";
        
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