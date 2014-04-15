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

using namespace std;
using namespace dealii;

const double toll=1e-8;

// max(s1+s2-k,0)

class Parametri2d{
public:
	//Dati
	double T;                                               // Scadenza
	double K;                                               // Strike price
	double S01;                                             // Spot price
	double S02;                                             // Spot price
	double r;                                               // Tasso risk free
        
	// Parametri della parte continua
	double sigma1;                                          // Volatilità
        double sigma2;                                          // Volatilità
        double ro;                                              // Volatilità
        
        // Parametri della parte salto
        double p;                                               // Parametro 1 Kou
        double lambda;                                          // Parametro 2 Kou
        double lambda_piu;                                      // Parametro 3 Kou
        double lambda_meno;                                     // Parametro 4 Kou
        
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
        return max(_S01*exp(p[0])+_S02*exp(p[1])-_K*exp(-_r*(_T-this->get_time())),0.);
        
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
        return max(S01*exp(p(0))+S02*exp(p(1))-K,0.);
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
        
        double                          price;
        
	Triangulation<dim>              triangulation;
	FE_Q<dim>                       fe;
	DoFHandler<dim>                 dof_handler;
        
        Triangulation<dim>              integral_triangulation;
        
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
        double dx1, dx2;
	double Smin1, Smax1, Smin2, Smax2;
        double xmin1, xmax1, xmin2, xmax2;
        
        bool ran;
public:
        Opzione(Parametri2d const &par_, int Nsteps_,  int refinement):
	par(par_),
	fe (1),
	dof_handler (triangulation),
	refs(refinement), 
	Nsteps(Nsteps_), 
	time_step (par.T/double(Nsteps_)),
        ran(false)
	{};
        
        double get_price(){ return 0; };
        
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
void Opzione<dim>::make_grid() {
	
        Smin1=par.S01*exp((par.r-par.sigma1*par.sigma1/2)*par.T
                            -par.sigma1*sqrt(par.T)*6);
	Smax1=par.S01*exp((par.r-par.sigma1*par.sigma1/2)*par.T
                            +par.sigma1*sqrt(par.T)*6);
        Smin2=par.S02*exp((par.r-par.sigma2*par.sigma2/2)*par.T
                         -par.sigma2*sqrt(par.T)*6);
	Smax2=par.S02*exp((par.r-par.sigma2*par.sigma2/2)*par.T
                         +par.sigma2*sqrt(par.T)*6);
        
	xmin1=log(Smin1/par.S01);
        xmax1=log(Smax1/par.S01);
        xmin2=log(Smin2/par.S02);
        xmax2=log(Smax2/par.S02);
        
        cout<< "Smin1= "<< Smin1<< "\t e Smax1= "<< Smax1<< endl;
	cout<< "Smin2= "<< Smin2<< "\t e Smax2= "<< Smax2<< endl;
	cout<< "xmin1= "<< xmin1<< "\t e xmax1= "<< xmax1<< endl;
	cout<< "xmin2= "<< xmin2<< "\t e xmax2= "<< xmax2<< endl;
        
        dx1=(xmax1-xmin1)/pow(2., refs);
        dx2=(xmax2-xmin2)/pow(2., refs);
        
        Point<dim> p1(xmin1,xmin2);
        Point<dim> p2(xmax1,xmax2);
        
        std::vector<unsigned> refinement={pow(2,refs), pow(2,refs)};
        
        GridGenerator::subdivided_hyper_rectangle(triangulation, refinement, p1, p2);
        
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
        
        QGauss<dim> quadrature_formula(4); // 2 nodes, 2d -> 4 quadrature points per cell
        
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
        
        sigma_matrix[0][0]=par.sigma1;
        sigma_matrix[1][1]=par.sigma2;
        sigma_matrix[0][1]=par.sigma1*par.sigma2*par.ro;
        sigma_matrix[1][0]=par.sigma1*par.sigma2*par.ro;
        
        Tensor< 1 , dim, double > ones;
	for (unsigned i=0;i<dim;++i)
                ones[i]=1;
        
        Tensor< 1, dim, double > trasp;
        trasp[0]=par.r-par.sigma1*par.sigma1/2;
        trasp[1]=par.r-par.sigma2*par.sigma2/2;
        
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
                                        
                                        cell_dd(i, j)+=fe_values.shape_grad(i, q_point)*sigma_matrix*fe_values.shape_grad(j, q_point)*fe_values.JxW(q_point);
                                        cell_fd(i, j)+=fe_values.shape_value(i, q_point)*(ones*fe_values.shape_grad(j,q_point))*fe_values.JxW(q_point);
                                        cell_ff(i, j)+=fe_values.shape_value(i, q_point)*fe_values.shape_value(j, q_point)*fe_values.JxW(q_point);
                                        cell_system(i, j)+=fe_values.JxW(q_point)*
                                        (-0.5*fe_values.shape_grad(i, q_point)*sigma_matrix*fe_values.shape_grad(j, q_point)-
                                         fe_values.shape_value(i, q_point)*(trasp*fe_values.shape_grad(j,q_point))+
                                         (1/time_step+par.r)*
                                         fe_values.shape_value(i, q_point)*fe_values.shape_value(j, q_point));
                                        
                                }
                
                cell->get_dof_indices (local_dof_indices);
                
                for (unsigned int i=0; i<dofs_per_cell;++i)
                        for (unsigned int j=0; j< dofs_per_cell; ++j) {
                                
                                dd_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_dd(i, j));
                                fd_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_fd(i, j));
                                ff_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_ff(i, j));
                                system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_system(i, j));
                                
                        }
                
        }
        /*
        cout<<"dd_matrix\n";
        dd_matrix.print(cout);
        cout<<"\nfd_matrix\n";
        fd_matrix.print(cout);
        cout<<"ff_matrix\n";
        ff_matrix.print(cout);*/
        
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
                        VectorTools::interpolate_boundary_values (dof_handler,
                                                                  1,
                                                                  bc,
                                                                  boundary_values);
                        VectorTools::interpolate_boundary_values (dof_handler,
                                                                  2,
                                                                  bc,
                                                                  boundary_values);
                        VectorTools::interpolate_boundary_values (dof_handler,
                                                                  3,
                                                                  bc,
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

int main() {
	Parametri2d par;
	par.T=1.;
	par.K=200;
	par.S01=100;
        par.S02=100;
	par.r=0.0367;
	par.sigma1=0.120381;
        par.sigma2=0.09;
        par.ro=0.1;
        
        // Parametri della parte salto
        par.p=0.20761;           // Parametro 1 Kou
        par.lambda=0.330966;     // Parametro 2 Kou
        par.lambda_piu=9.65997;  // Parametro 3 Kou
        par.lambda_meno=3.13868; // Parametro 4 Kou
        
        // tempo // spazio
	Opzione<2> Call(par, 10, 5);
	double prezzo=Call.run();
        
        cout<<"Prezzo "<<prezzo<<"\n";
        cout<<"2d v000\n";
	
	return 0;
}