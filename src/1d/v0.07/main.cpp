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
	double get_price();
	void output_results () const {};
        
	Triangulation<dim>   triangulation;
	FE_Q<dim>            fe;
	DoFHandler<dim>      dof_handler;
        
        Triangulation<dim>   integral_triangulation;
        
	SparsityPattern      sparsity_pattern;
	SparseMatrix<double> system_matrix;
	SparseMatrix<double> system_M2;
	SparseMatrix<double> dd_matrix;
	SparseMatrix<double> fd_matrix;
	SparseMatrix<double> ff_matrix;
        
	Vector<double>       solution;
	Vector<double>       system_rhs;
        
        std::vector<unsigned int>     index;
        std::vector<double>  u_array;
        double *             x_array;
        
        std::vector< Point<dim> > grid_points;
        std::vector< Point<dim> > integral_grid_points;
        
        std::vector< double > integral_weights;
        std::vector< Point<dim> > integral_points;
	
	unsigned int refs, Nsteps;
	double time_step;
        double dx;
	double Smin, Smax, xmin, xmax;
        
        std::vector<int> grid_size;
        
        double alpha, Bmin, Bmax;
        double k(double y);
        void Levy_integral_part1();
        void integrale2_Levy(Vector<double> &J, Vector<double> const &x);
        void f_u(std::vector<Point<dim> > &val, std::vector<Point<dim> > const &y);
        inline double payoff(double x, double K, double S0){return max(S0*exp(x)-K, 0.);}
        void calculate_weights();
        
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

template<int dim>
void Opzione<dim>::Levy_integral_part1(){
                
        alpha=0;
        
        for (int i=0; i<integral_grid_points.size(); ++i) {
                alpha+=integral_weights[i]*(exp(integral_grid_points[i][0])-1)*k(integral_grid_points[i][0]);
        }
        
        return;
}

template<int dim>
void Opzione<dim>::integrale2_Levy(Vector<double> &J, Vector<double> const &x) {
        /*
        int n=integral_grid_points.size();
        
        Vector<double> J3(J.size());
        
        Vector<double> y(n); // Nodi di quadratura
        y.reinit(n);
        
        for (int i=0; i<n; ++i) {
                y(i)=integral_grid_points[i][0];
        }*/
        
        for (int i=0; i<J.size(); ++i) {
                J(i)=0;
                //J3(i)=0;
        }
        
        //double x_array[x.size()];
        
        solution.extract_subvector_to(index, u_array);
        /*
        cout<<"index ";
        for (int k=0; k<index.size(); ++k) {
                cout<<index[k]<<" ";
        }
        cout<<"\n";
         */
        /*
        cout<<"temp ";
        for (int k=0; k<temp.size(); ++k) {
                cout<<temp[k]<<" ";
        }
        cout<<"\n";
        */
        /*
        for (int i=0; i<x.size(); ++i) {
                //x_array[i]=x(i);
                u_array[i]=solution(i);
        }*/
        //u_array=&temp[0];
        /*
        cout<<"x ";
        for (int k=0; k<x.size(); ++k) {
                cout<<x[k]<<" ";
        }
        cout<<"\n";
        cout<<"grid ";
        for (int k=0; k<grid_points.size(); ++k) {
                cout<<grid_points[k]<<" ";
        }
        cout<<"\n";
        cout<<"solution ";
        for (int k=0; k<solution.size(); ++k) {
                cout<<solution[k]<<" ";
        }
        cout<<"\n";
        */
        std::vector<Point<dim> > z_(integral_grid_points);
        
        for (int i=1; i<J.size()-1; ++i) {
                /*
                Vector<double> val;
                Vector<double> z(y);
                z.add(x(i));
                */
                // calcolo z+x(i)
                std::vector<Point<dim> > x_i(z_.size(),grid_points[i]);
                for (int k=0; k<x_i.size(); ++k) {
                        x_i[k][0]=x_i[k][0]+z_[k][0];
                }
                
                // stampo
                /*
                cout<<"z ";
                for (int k=0; k<z.size(); ++k) {
                        cout<<z[k]<<" ";
                }
                cout<<"\nz_ ";
                for (int k=0; k<x_i.size(); ++k) {
                        cout<<x_i[k]<<" ";
                }
                cout<<"\n z.size "<<z.size()<<" z_.size() "<<z_.size()<<"\n";
                */
                std::vector<Point<dim> > val(z_.size());
                f_u(val, x_i);
                /*
                f_u(val,x_array,&u_array[0],z,x.size());
                 */
                /*
                cout<<"val ";
                for (int k=0; k<val.size(); ++k) {
                        cout<<val[k]<<" ";
                }
                cout<<"\n";
                cout<<"val2 ";
                for (int k=0; k<val2.size(); ++k) {
                        cout<<val2[k]<<" ";
                }
                cout<<"\n";
                cout<<"val3 ";
                for (int k=0; k<val3.size(); ++k) {
                        cout<<val3[k]<<" ";
                }
                cout<<"\n";*/
                for (int j=0; j<integral_grid_points.size(); ++j) {
                        J(i)+=integral_weights[j]*k(integral_grid_points[j][0])*val[j][0];
                        //J3(i)+=integral_weights[j]*k(integral_grid_points[j][0])*val3[j][0];
                }
        }
        /*
        cout<<"J ";
        for (int k=0; k<J.size(); ++k) {
                cout<<J[k]<<" ";
        }
        cout<<"\n";*/
        cout<<"J ";
        for (int k=0; k<J.size(); ++k) {
                cout<<J[k]<<" ";
        }
        cout<<"\n";
        //J=J3;
}

template<int dim>
void Opzione<dim>::f_u(std::vector<Point<dim> > &val, std::vector<Point<dim> > const &y){
        
        gsl_interp_accel *my_accel_ptr = gsl_interp_accel_alloc ();
        gsl_spline *my_spline_ptr = gsl_spline_alloc (gsl_interp_cspline, grid_points.size());
        gsl_spline_init (my_spline_ptr, x_array, &u_array[0], grid_points.size());
        
        int k=0;
        
        while (y[k][0]<x_array[0]) {
                val[k]=Point<dim> (0.);
                ++k;
        }
        int j=0;
        while (k<y.size() && y[k][0]<x_array[grid_points.size()-1]) {
                val[k]=Point<dim> (gsl_spline_eval(my_spline_ptr, y[k][0] , my_accel_ptr));
                //val[k]=Point<dim> (solution(j));
                ++k;
                ++j;
        }
        
        for (int i=k; i<y.size(); ++i) {
                val[i]=Point<dim> (payoff(x_array[grid_points.size()-1],par.K,par.S0));
        }
        
        gsl_spline_free(my_spline_ptr);
        gsl_interp_accel_free(my_accel_ptr);
        
        return;
}

//template<int dim>
//void Opzione<dim>::f_u(Vector<double> &val, double * x_array, double * u_array, Vector<double> const &y, int n){
//        
//        //66.2425 45.1334 5.26828 0.800746 0
//        /*
//        u_array[0]=66.2425;
//        u_array[1]=45.1334;
//        u_array[2]=5.26828;
//        u_array[3]=0.800746;
//        u_array[4]=0;
//        */
//        
//        gsl_interp_accel *my_accel_ptr = gsl_interp_accel_alloc ();
//        gsl_spline *my_spline_ptr = gsl_spline_alloc (gsl_interp_cspline, n);
//        gsl_spline_init (my_spline_ptr, x_array, u_array, n);
//        
//        /*
//        //gsl_interp * workspace = gsl_interp_alloc(gsl_interp_polynomial, n);
//        gsl_interp * workspace = gsl_interp_alloc(gsl_interp_linear, n);
//        gsl_interp_accel * accel = gsl_interp_accel_alloc();
//        gsl_interp_init(workspace, x_array, u_array, n);
//         */
//        /*
//        cout<<"***in f_u\nn "<<n<<"\nx ";
//        for (int i=0; i<n; ++i) {
//                cout<<x_array[i]<<" ";
//        }
//        cout<<"\n";
//        cout<<"u ";
//        for (int i=0; i<n; ++i) {
//                cout<<u_array[i]<<" ";
//        }
//        cout<<"\n";
//        cout<<"y.size "<<y.size()<<"y ";
//        y.print(cout);
//        cout<<"\n";
//        */
//        val=Vector<double>(y.size());
//        int j=0;
//        int k=0;
//        
//        while (y(j)<x_array[0]) {
//                val(k)=0;
//                ++k;
//                ++j;
//        }
//        //cout<<"k "<<k<<"\n";
//        while (j<y.size() && y(j)<x_array[n-1]) {
//                val(k)=gsl_spline_eval(my_spline_ptr, y(k) , my_accel_ptr);
//                //val(k)=gsl_interp_eval(workspace, x_array, u_array, y(k) , NULL);
//                //val(k)=interp_linear(y(k), x_array, u_array);
//                ++j;
//                ++k;
//        }
//        
//        for (int i=k; i<y.size(); ++i) {
//			val(i)=payoff(x_array[n-1],par.K,par.S0);
//        }
//        /*
//        gsl_interp_free(workspace);
//        gsl_interp_accel_free(accel);
//        */
//        gsl_spline_free(my_spline_ptr);
//        gsl_interp_accel_free(my_accel_ptr);
//        
//        return;
//}

template<int dim>
void Opzione<dim>::make_grid() {
	
        Smin=0.5*par.S0*exp((par.r-par.sigma*par.sigma/2)*par.T
                        -par.sigma*sqrt(par.T)*6);
	Smax=1.5*par.S0*exp((par.r-par.sigma*par.sigma/2)*par.T
                        +par.sigma*sqrt(par.T)*6);
        
	cout<< "Smin= "<< Smin<< "\t e Smax= "<< Smax<< endl;
	xmin=log(Smin/par.S0);
        xmax=log(Smax/par.S0);
        
        dx=(xmax-xmin)/pow(2., refs);
        double tol=10e-9;
        
        cout<<"dx "<<dx<<"\n";
        
        Bmin=xmin;
        Bmax=xmax;
        
        while(k(Bmin)>tol)
                Bmin-=dx;
        
        while(k(Bmax)>tol)
                Bmax+=dx;
        
        if (Bmin==xmin && Bmax==xmax) {
                
                GridGenerator::subdivided_hyper_cube(triangulation,pow(2,refs), xmin,xmax);
                
                grid_points=triangulation.get_vertices();
                
                cout<<"grid ";
                for (int i=0; i<grid_points.size(); ++i) {
                        cout<<grid_points[i][0]<<" ";
                }
                cout<<"\n";
                
                cout<<"here!\n";
                
                GridGenerator::subdivided_hyper_cube(integral_triangulation, pow(2,refs), Bmin, Bmax);
                
                integral_grid_points=integral_triangulation.get_vertices();
                
                sort(integral_grid_points.begin(), integral_grid_points.end(), grid_order);
                
        }
        else {
                GridGenerator::subdivided_hyper_cube(triangulation,pow(2,refs), xmin,xmax);
                
                grid_points=triangulation.get_vertices();
                
                cout<<"grid ";
                for (int i=0; i<grid_points.size(); ++i) {
                        cout<<grid_points[i][0]<<" ";
                }
                cout<<"\n";
                
                // Left & Right Triangulation
                Triangulation<dim> left_triangulation;
                Triangulation<dim> right_triangulation;
                
                cout<<"error "<<(Bmin-xmin)/dx<<"\n";
                
                GridGenerator::subdivided_hyper_cube(left_triangulation, round((xmin-Bmin)/dx), Bmin, xmin);
                
                GridGenerator::subdivided_hyper_cube(right_triangulation, round((Bmax-xmax)/dx), xmax, Bmax);
                
                auto left_grid_points=left_triangulation.get_vertices();
                auto right_grid_points=right_triangulation.get_vertices();
                
                sort(left_grid_points.begin(), left_grid_points.end(), grid_order);
                sort(right_grid_points.begin(), right_grid_points.end(), grid_order);
                /*
                cout<<"left grid ";
                for (int i=0; i<left_grid_points.size(); ++i) {
                        cout<<left_grid_points[i]<<" ";
                }
                cout<<"\n";
                
                cout<<"right grid ";
                for (int i=0; i<right_grid_points.size(); ++i) {
                        cout<<right_grid_points[i]<<" ";
                }
                cout<<"\n";
                */
                Triangulation<dim> temp;
                GridGenerator::merge_triangulations(left_triangulation, triangulation, temp);
                
                auto temp_grid_points=temp.get_vertices();
                sort(temp_grid_points.begin(), temp_grid_points.end(), grid_order);
                
                /*
                cout<<"temp grid ";
                for (int i=0; i<temp_grid_points.size(); ++i) {
                        cout<<temp_grid_points[i]<<" ";
                }
                cout<<"\n";
                
                grid_points=triangulation.get_vertices();
                
                cout<<"grid ";
                for (int i=0; i<grid_points.size(); ++i) {
                        cout<<grid_points[i][0]<<" ";
                }
                cout<<"\n";
                */
                std::cout << "   Number of active cells: "
                << temp.n_active_cells()
                << std::endl
                << "   Total number of cells: "
                << temp.n_cells()
                << std::endl;
                
                GridGenerator::merge_triangulations(temp, right_triangulation, integral_triangulation);
                
                integral_grid_points=integral_triangulation.get_vertices();
                
                sort(integral_grid_points.begin(), integral_grid_points.end(), grid_order);
                
                /*
                cout<<"integral grid ";
                for (int i=0; i<integral_grid_points.size(); ++i) {
                        cout<<integral_grid_points[i]<<" ";
                }
                cout<<"\n";
                cout<<"differences ";
                for (int i=0; i<integral_grid_points.size()-1; ++i) {
                        cout<<integral_grid_points[i+1]-integral_grid_points[i]<<" ";
                }
                cout<<"\n";
                */
                grid_size.reserve(3);
                grid_size[0]=left_grid_points.size()-1;
                grid_size[1]=grid_points.size();
                grid_size[2]=right_grid_points.size()-1;
                
                x_array=new double[grid_points.size()];
                //u_array=new double[grid_points.size()];
                
                index=std::vector<unsigned int>(grid_points.size());
                u_array=std::vector<double> (grid_points.size());
                //index.reserve(grid_points.size());
                cout<<"grid size "<<grid_points.size()<<" index size "<<index.size()<<"\n";
                
                for (int i=0; i<grid_points.size(); ++i) {
                        x_array[i]=grid_points[i][0];
                        index[i]=i;
                }
                /*
                cout<<"index ";
                for (int i=0; i<index.size(); ++i) {
                        cout<<index[i]<<" ";
                }
                cout<<"\n";*/
        }
        
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
	
	cout<< "xmin e xmx\n";
	cout<<xmin<<"\t"<<xmax<<"\n";
        
        Vector<double> x;
        x.reinit(dof_handler.n_dofs());
        for (int i=0; i<x.size(); ++i) {
                x(i)=xmin+i*(xmax-xmin)/(x.size()-1);
        }
        
        double dx=x(2)-x(1);
        cout<<"dx "<<dx<<"\n";
        
        cout<<"x=[ ";
        for (int i=0; i<x.size()-1; ++i) {
                cout<<x(i)<<"; ";
        }
	cout<<x(x.size()-1)<<" ]\n";
        
	VectorTools::interpolate (dof_handler, PayOff<dim>(par.K, par.S0),solution);
	/*cout<<"solution:\n";
	solution.print(cout);
	cout<<"\n";*/
	unsigned int Step=Nsteps;
	
	Boundary_Right_Side<dim> right_bound(par.S0, par.K, par.T, par.r);
	cout<< "time step is"<< time_step<< endl;
	for (double time=par.T-time_step;time >=0;time-=time_step, --Step) {
                cout<< "Step "<< Step<<"\t at time \t"<< time<< endl;
#ifdef __PIDE__/*
                cout<<"x ";
                x.print(cout);
                cout<<"\n";*/
                Vector<double> J;
                J.reinit(solution.size());
                integrale2_Levy(J, x);
                /*cout<<"J ";
                J.print(cout);
                cout<<"\n";*/
                ff_matrix.vmult(system_rhs, J);
                Vector<double> temp;
                temp.reinit(dof_handler.n_dofs());
                system_M2.vmult(temp,solution);
                system_rhs+=temp;
                /*
                cout<<"rhs before ";
                system_rhs.print(cout);
                cout<<"\n";
                cout<<"rhs after ";
                system_rhs.print(cout);
                cout<<"\n";
                cout<<"J ";
                J.print(cout);
                cout<<"\n";*/
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
                
#ifdef __VERBOSE__
                cout<<"solution:\n";
                solution.print(cout);
                cout<<"\n";
# endif
                
                
                
	}
	
        
	cout<<"sol=[ ";
	for (int i=0; i<solution.size()-1; ++i) {
                cout<<solution(i)<<"; ";
        }
	cout<<solution(solution.size()-1)<<" ]\n";
}

template<int dim>
double Opzione<dim>::get_price() {
	
	return 0;
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
        
                        // tempo // spazio
	Opzione<1> Call(par, 2, 2);
	Call.run();
        
        cout<<"v007\n";
	
	return 0;
}