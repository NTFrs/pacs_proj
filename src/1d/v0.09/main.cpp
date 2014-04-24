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

#include <climits>

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

#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>

// #define __VERBOSE__


using namespace std;
using namespace dealii;

//#define dim 1

#define __PIDE__
//#define __MATLAB__
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
	double T;                                               // Scadenza
	double K;                                               // Strike price
	double S0;                                              // Spot price
	double r;                                               // Tasso risk free
        
	// Parametri della parte continua
	double sigma;                                           // Volatilità
        
        // Parametri della parte salto
        double p;                                               // Parametro 1 Kou
        double lambda;                                          // Parametro 2 Kou
        double lambda_piu;                                      // Parametro 3 Kou
        double lambda_meno;                                     // Parametro 4 Kou
        
	Parametri()=default;
	Parametri(const Parametri &)=default;
};

auto grid_order = [] (Point<1> p1, Point<1> p2) {return p1[0]<p2[0];};

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
        return max(S0*exp(p(0))-K,0.);
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
	Boundary_Right_Side(double S0, double K, double T,  double r) : Function< dim>(), _S0(S0), _K(K), _T(T), _r(r) {};
        
	virtual double value (const Point<dim> &p, const unsigned int component =0) const;
private:
        double _S0;
	double _K;
	double _T;
	double _r;
};

template<int dim>
double Boundary_Right_Side<dim>::value(const Point<dim> &p, const unsigned int component) const
{
	Assert (component == 0, ExcInternalError());
        return _S0*exp(p[0])-_K*exp(-_r*(_T-this->get_time()));
        
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
        
        double                          price;
        
	Triangulation<dim>              triangulation;
	FE_Q<dim>                       fe;
	DoFHandler<dim>                 dof_handler;
        
        Triangulation<dim>              integral_triangulation;
        Triangulation<dim>              integral_triangulation2;
        
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
        
        std::vector<unsigned int>       index;
        std::vector<double>             u_array;
        double *                        x_array;
        
        std::vector< Point<dim> >       grid_points;
        std::vector< Point<dim> >       integral_grid_points;
        
        std::vector< double >           integral_weights;
        std::vector< Point<dim> >       integral_points;
	
	unsigned int refs, Nsteps;
	double time_step;
        double dx;
	double Smin, Smax, xmin, xmax;
        
        double alpha, Bmin, Bmax;
        
        double k(double y);
        void Levy_integral_part1();
        void Levy_integral_part2(Vector<double> &J);
        void f_u(std::vector<Point<dim> > &val, std::vector<Point<dim> > const &y);
        inline double payoff(double x, double K, double S0){return max(S0*exp(x)-K, 0.);}
        void calculate_weights();
        
        bool ran;
        
public:
	Opzione(Parametri const &par_, int Nsteps_,  int refinement):
	par(par_),
	fe (1),
	dof_handler (triangulation),
        fe2 (1),
        dof_handler_2 (integral_triangulation),
	refs(refinement), 
	Nsteps(Nsteps_), 
	time_step (par.T/double(Nsteps_)),
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
double Opzione<dim>::k(double y){
        if (y>0)
                return par.p*par.lambda*par.lambda_piu*exp(-par.lambda_piu*y);
        else
                return (1-par.p)*par.lambda*par.lambda_meno*exp(par.lambda_meno*y);
}

template<int dim>
void Opzione<dim>::calculate_weights(){
        integral_weights=std::vector<double>(integral_grid_points.size(), dx);
        integral_weights[0]=dx/2;
        integral_weights[integral_weights.size()-1]=dx/2;
}


template <int dim>
class Coefficient : public Function<dim>
{
public:
        Coefficient ()  : Function<dim>() {}
        virtual double value (const Point<dim>   &p,
                              const unsigned int  component = 0) const;
        virtual void value_list (const std::vector<Point<dim> > &points,
                                 std::vector<double>            &values,
                                 const unsigned int              component = 0) const;
};

template <int dim>
double Coefficient<dim>::value (const Point<dim> &p,
                                const unsigned int /* component */) const
{
        return p(0);
}

template <int dim>
void Coefficient<dim>::value_list (const std::vector<Point<dim> > &points,
                                   std::vector<double>            &values,
                                   const unsigned int              component) const
{
        Assert (values.size() == points.size(),
                ExcDimensionMismatch (values.size(), points.size()));
        Assert (component == 0,
                ExcIndexRange (component, 0, 1));
        const unsigned int n_points = points.size();
        for (unsigned int i=0; i<n_points; ++i)
        {
                values[i]=points[i][0];
        }
}

template <int dim>
class Zero : public Function<dim>
{
public:
        Zero ()  : Function<dim>() {}
        virtual double value (const Point<dim>   &p,
                              const unsigned int  component = 0) const;
        virtual void value_list (const std::vector<Point<dim> > &points,
                                 std::vector<double>            &values,
                                 const unsigned int              component = 0) const;
};

template <int dim>
double Zero<dim>::value (const Point<dim> &p,
                         const unsigned int /* component */) const
{
        return 0.;
}

template <int dim>
void Zero<dim>::value_list (const std::vector<Point<dim> > &points,
                            std::vector<double>            &values,
                            const unsigned int              component) const
{
        Assert (values.size() == points.size(),
                ExcDimensionMismatch (values.size(), points.size()));
        Assert (component == 0,
                ExcIndexRange (component, 0, 1));
        const unsigned int n_points = points.size();
        for (unsigned int i=0; i<n_points; ++i)
        {
                values[i]=points[i][0];
        }
}

template<int dim>
void Opzione<dim>::Levy_integral_part1(){
        
        alpha=0;
        
        //GridGenerator::subdivided_hyper_cube(integral_triangulation2, pow(2, refs), Bmin, Bmax);
        
        QGauss<dim> quadrature_formula2(2);
	FEValues<dim> fe_values2 (fe2, quadrature_formula2, update_values | update_quadrature_points | update_JxW_values);
        
        typename DoFHandler<dim>::active_cell_iterator
	cell=dof_handler_2.begin_active(),
	endc=dof_handler_2.end();
        
        const unsigned int   dofs_per_cell = fe2.dofs_per_cell;
	const unsigned int   n_q_points    = quadrature_formula2.size();
        
        const Coefficient<dim> coefficient;
        std::vector<double>    coefficient_values (n_q_points);
        
        for (; cell !=endc;++cell) {
                
                fe_values2.reinit(cell);
                
                coefficient.value_list (fe_values2.get_quadrature_points(),
                                        coefficient_values);
                
                for (unsigned q_point=0;q_point<n_q_points;++q_point)
                        alpha+=fe_values2.JxW(q_point)*(exp(coefficient_values[q_point])-1)*k(coefficient_values[q_point]);
        }
        
        return;
}

template<int dim>
void Opzione<dim>::Levy_integral_part2(Vector<double> &J) {
        
        J.reinit(solution.size()); // If fast is false, the vector is filled by zeros
        
        Vector<double> J2;
        J2.reinit(solution.size());
        
#ifdef __INTERPOLATION__
        solution.extract_subvector_to(index, u_array);
#endif
        
        std::vector<Point<dim> > z(integral_grid_points);
        
        for (int i=1; i<J.size()-1; ++i) {
                
                // One way
                {
                        
                        std::vector<Point<dim> > x_i(z.size(),grid_points[i]);
                        for (int k=0; k<x_i.size(); ++k) {
                                x_i[k][0]=x_i[k][0]+z[k][0];
                        }
                        
                        std::vector<Point<dim> > val(z.size());
                        f_u(val, x_i);
                        
                        for (int j=0; j<integral_grid_points.size(); ++j) {
                                J(i)+=integral_weights[j]*k(integral_grid_points[j][0])*val[j][0];
                        }
                        
                }
                
                
                // or another...
                {
                        // left
                        {
                                Triangulation<dim> left_tri;
                                FE_Q<dim> fe_left (1);
                                DoFHandler<dim> dof_handler_left (left_tri);
                                GridGenerator::hyper_cube(left_tri, Bmin+grid_points[i][0], xmin);
                                
                                dof_handler_left.distribute_dofs(fe_left);
                                
                                QGauss<dim> quadrature_formula_left(2);
                                FEValues<dim> fe_values_left (fe_left, quadrature_formula_left, update_values |
                                                              update_quadrature_points | update_JxW_values);
                                
                                typename DoFHandler<dim>::active_cell_iterator
                                cell=dof_handler_2.begin_active(),
                                endc=dof_handler_2.end();
                                
                                const unsigned int   dofs_per_cell = fe_left.dofs_per_cell;
                                const unsigned int   n_q_points    = quadrature_formula_left.size();
                                
                                const Coefficient<dim> coefficient;
                                std::vector<double>    coefficient_values (n_q_points);
                                
                                const Zero<dim>         zero;
                                std::vector<double>     zero_values (n_q_points);
                                
                                for (; cell !=endc;++cell) {
                                        
                                        fe_values_left.reinit(cell);
                                        
                                        coefficient.value_list (fe_values_left.get_quadrature_points(),
                                                                coefficient_values);
                                        
                                        for (unsigned q_point=0;q_point<n_q_points;++q_point)
                                                J2(i)+=fe_values_left.JxW(q_point)*zero_values[q_point]
                                                *k(coefficient_values[q_point]);
                                }
                                
                        }
                        
                        // middle
                        {
                                QGauss<dim> quadrature_formula(2);
                                
                                FEValues<dim> fe_values (fe, quadrature_formula, update_values   | update_quadrature_points |
                                                         update_JxW_values);
                                
                                typename DoFHandler<dim>::active_cell_iterator
                                cell=dof_handler.begin_active(),
                                endc=dof_handler.end();
                                
                                const unsigned int   dofs_per_cell = fe.dofs_per_cell;
                                const unsigned int   n_q_points    = quadrature_formula.size();
                                
                                const Coefficient<dim> coefficient;
                                std::vector<double>    coefficient_values (n_q_points);
                                
                                for (; cell !=endc;++cell) {
                                        
                                        fe_values.reinit(cell);
                                        
                                        coefficient.value_list (fe_values.get_quadrature_points(),
                                                                coefficient_values);
                                        
                                        Vector<double> interpolated_points(n_q_points);
                                        cout<<"ora vado in sf\n";
                                        VectorTools::interpolate(dof_handler, fe_function, interpolated_points);
                                        cout<<"sono andato in sf\n";
                                        for (unsigned q_point=0;q_point<n_q_points;++q_point)
                                                J2(i)+=fe_values.JxW(q_point)*interpolated_points(q_point)
                                                *k(coefficient_values[q_point]);
                                }
                                
                        }
                        
                }
                
        }
}

template<int dim>
void Opzione<dim>::f_u(std::vector<Point<dim> > &val, std::vector<Point<dim> > const &y){
        
#ifdef __INTERPOLATION__
        gsl_interp_accel *my_accel_ptr = gsl_interp_accel_alloc ();
        gsl_spline *my_spline_ptr = gsl_spline_alloc (gsl_interp_cspline, grid_points.size());
        gsl_spline_init (my_spline_ptr, x_array, &u_array[0], grid_points.size());
#endif
        
        int k=0;
        
        while (y[k][0]<x_array[0]) {
                val[k]=Point<dim> (0.);
                ++k;
        }
        int j=0;
        while (k<y.size() && y[k][0]<x_array[grid_points.size()-1]) {
#ifdef __INTERPOLATION__
                val[k]=Point<dim> (gsl_spline_eval(my_spline_ptr, y[k][0] , my_accel_ptr));
#else
                val[k]=Point<dim> (solution(j));
#endif
                ++k;
                ++j;
        }
        
        for (int i=k; i<y.size(); ++i) {
                val[i]=Point<dim> (payoff(x_array[grid_points.size()-1],par.K,par.S0));
        }
        
#ifdef __INTERPOLATION__
        gsl_spline_free(my_spline_ptr);
        gsl_interp_accel_free(my_accel_ptr);
#endif
        
        return;
}

template<int dim>
void Opzione<dim>::make_grid() {
	
        Smin=0.5*par.S0*exp((par.r-par.sigma*par.sigma/2)*par.T
                            -par.sigma*sqrt(par.T)*6);
	Smax=1.5*par.S0*exp((par.r-par.sigma*par.sigma/2)*par.T
                            +par.sigma*sqrt(par.T)*6);
        
	cout<< "Smin= "<< Smin<< "\t e Smax= "<< Smax<< endl;
	xmin=0;
        xmax=0;
        
        dx=(log(Smax/par.S0)-log(Smin/par.S0))/pow(2., refs);
        cout<<"dx "<<dx<<"\n";
        
        while (xmin>log(Smin/par.S0))
                xmin-=dx;
        while (xmax<log(Smax/par.S0))
                xmax+=dx;
        xmin-=dx;
        xmax+=dx;
        
        cout<<"dx "<<dx<<"\n";
        
        Bmin=xmin;
        Bmax=xmax;
        
        while(k(Bmin)>toll)
                Bmin-=dx;
        
        while(k(Bmax)>toll)
                Bmax+=dx;
        
        cout<<"Bmin "<<Bmin<<" Bmax "<<Bmax<<"\n";
        
        GridGenerator::subdivided_hyper_cube(triangulation,pow(2,refs)+3, xmin,xmax);
        
        grid_points=triangulation.get_vertices();
        
        if (Bmin==xmin && Bmax==xmax) {
                
                GridGenerator::subdivided_hyper_cube(integral_triangulation, pow(2,refs), Bmin, Bmax);
                
                integral_grid_points=integral_triangulation.get_vertices();
                
                sort(integral_grid_points.begin(), integral_grid_points.end(), grid_order);
                
        }
        else if (Bmin==xmin && Bmax>=xmax) {
                
                Triangulation<dim> right_triangulation;
                
                GridGenerator::subdivided_hyper_cube(right_triangulation, round((Bmax-xmax)/dx), xmax, Bmax);
                
                GridGenerator::merge_triangulations(triangulation, right_triangulation, integral_triangulation);
                
                integral_grid_points=integral_triangulation.get_vertices();
                
                sort(integral_grid_points.begin(), integral_grid_points.end(), grid_order);
                
        }
        else if (Bmin<=xmin && Bmax==xmax) {
                
                Triangulation<dim> left_triangulation;
                
                GridGenerator::subdivided_hyper_cube(left_triangulation, round((xmin-Bmin)/dx), Bmin, xmin);
                
                GridGenerator::merge_triangulations(left_triangulation, triangulation, integral_triangulation);
                
                integral_grid_points=integral_triangulation.get_vertices();
                
                sort(integral_grid_points.begin(), integral_grid_points.end(), grid_order);
                
        }
        else {
                // Left & Right Triangulation
                Triangulation<dim> left_triangulation;
                Triangulation<dim> right_triangulation;
                
                GridGenerator::subdivided_hyper_cube(left_triangulation, round((xmin-Bmin)/dx), Bmin, xmin);
                
                GridGenerator::subdivided_hyper_cube(right_triangulation, round((Bmax-xmax)/dx), xmax, Bmax);
                
                Triangulation<dim> temp;
                GridGenerator::merge_triangulations(left_triangulation, triangulation, temp);
                
                GridGenerator::merge_triangulations(temp, right_triangulation, integral_triangulation);
                
                integral_grid_points=integral_triangulation.get_vertices();
                
                sort(integral_grid_points.begin(), integral_grid_points.end(), grid_order);
                
        }
        
        x_array=new double[grid_points.size()];
        
        index=std::vector<unsigned int>(grid_points.size());
        u_array=std::vector<double> (grid_points.size());
        
        for (int i=0; i<grid_points.size(); ++i) {
                x_array[i]=grid_points[i][0];
                index[i]=i;
        }
        
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
                                if (std::fabs(cell->face(face)->center()(0) - (xmax)) < toll)
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
        
        calculate_weights();
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
	
	unsigned int Step=Nsteps;
	
	Boundary_Right_Side<dim> right_bound(par.S0, par.K, par.T, par.r);
	cout<< "time step is"<< time_step<< endl;
	for (double time=par.T-time_step;time >=0;time-=time_step, --Step) {
                cout<< "Step "<< Step<<"\t at time \t"<< time<< endl;
#ifdef __PIDE__
                Vector<double> J;
                J.reinit(solution.size());
                Levy_integral_part2(J);
                
                ff_matrix.vmult(system_rhs, J);
                Vector<double> temp;
                temp.reinit(dof_handler.n_dofs());
                system_M2.vmult(temp,solution);
                system_rhs+=temp;
#else
                system_M2.vmult(system_rhs, solution);
#endif
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
        cout<<"x=[ ";
        for (int i=0; i<grid_points.size()-1; ++i) {
                cout<<grid_points[i][0]<<"; ";
        }
	cout<<grid_points[grid_points.size()-1][0]<<" ];\n";
        
	cout<<"sol=[ ";
	for (int i=0; i<solution.size()-1; ++i) {
                cout<<solution(i)<<"; ";
        }
	cout<<solution(solution.size()-1)<<" ];\n";
#endif
}

template<int dim>
double Opzione<dim>::get_price() {
        
        if (ran==false) {
                this->run();
        }
        
        // find 0 in grid
        unsigned position=grid_points.size();
        
        for (unsigned i=0; i<grid_points.size(); ++i) {
                if (grid_points[i][0]<=100*eps && grid_points[i][0]>=0) {
                        position=i;                     // if 0 found, set the position
                        i=grid_points.size();           // quit the loop
                }
        }
        
        // if position has not been set, exit
        if (position==grid_points.size()) {
                cerr<<"An error occurred.\n";
                std::exit(-1);
        }
        
        // return price
        return solution(position);
}


int main() {
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
        
        cout<<"eps "<<eps<<"\n";
        
        // tempo // spazio
	Opzione<1> Call(par, 100, 10);
	Call.run();
        
        cout<<"Prezzo "<<Call.get_price()<<"\n";
        cout<<"v009\n";
	
	return 0;
}