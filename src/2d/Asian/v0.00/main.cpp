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

// template <int dim>
// class Coefficient : public Function<dim>
// {
// public:
//         Coefficient (double _r, double _S0)  : Function<dim>(), r(_r), S0(_S0) {}
//         virtual double value (const Point<dim>   &p,
//                               const unsigned int  component = 0) const;
//         virtual void value_list (const std::vector<Point<dim> > &points,
//                                  std::vector<double>            &values,
//                                  const unsigned int              component = 0) const;
// private:
//         double r;
//         double S0;
// };
// 
// template <int dim>
// double Coefficient<dim>::value (const Point<dim> &p,
//                                 const unsigned int /*component*/) const
// {
//         return S0*exp(p(0));
// }
// 
// template <int dim>
// void Coefficient<dim>::value_list (const std::vector<Point<dim> > &points,
//                                    std::vector<double>            &values,
//                                    const unsigned int              component) const
// {
//         Assert (values.size() == points.size(),
//                 ExcDimensionMismatch (values.size(), points.size()));
//         Assert (component == 0,
//                 ExcIndexRange (component, 0, 1));
//         const unsigned int n_points = points.size();
//         for (unsigned int i=0; i<n_points; ++i)
//         {
//                 values[i]=S0*exp(points[i][0]);
//         }
// }

template<int dim>
class Boundary_Condition: public Function<dim>
{
public:
	Boundary_Condition() : Function< dim>() {};
        
	virtual double value (const Point<dim> &p, const unsigned int component =0) const;
};

template<int dim>
double Boundary_Condition<dim>::value(const Point<dim> &p, const unsigned int component) const
{
	Assert (component == 0, ExcInternalError());
        return max(p[0]-p[1]/this->get_time(),0.);
        
}

template<int dim>
class PayOff : public Function<dim>
{
public:
        PayOff (double T_) : Function<dim>(), T(T_) {};
        
        virtual double value (const Point<dim>   &p,
                              const unsigned int  component = 0) const;
private:
        double T;
};

template<int dim>
double PayOff<dim>::value (const Point<dim>  &p,
                           const unsigned int component) const
{
        Assert (component == 0, ExcInternalError());
        return max(p[0]-p[1]/T,0.);
}

template<int dim>
class Opzione{
private:
	Parametri par;
	void make_grid();
	void setup_system () ;
	void assemble_system () ;
	void solve () ;
	void output_results () const {};
        
        double                          price;
        
	Triangulation<dim>              triangulation;
	FE_Q<dim>                       fe;
	DoFHandler<dim>                 dof_handler;
        
//         Triangulation<dim>              integral_triangulation;
        
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
//         double dx;
	double Smin, Smax;
//         double xmin, xmax;
        double Amin, Amax;
        
        bool ran;
public:
        Opzione(Parametri const &par_, int Nsteps_,  int refinement):
	par(par_),
	fe (1),
	dof_handler (triangulation),
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
                output_results();
                return get_price();
                
        };
};

template<int dim>
void Opzione<dim>::make_grid() {
	
    Smin=par.S0*exp((par.r-par.sigma*par.sigma/2)*par.T
                            -par.sigma*sqrt(par.T)*6);
	Smax=par.S0*exp((par.r-par.sigma*par.sigma/2)*par.T
                            +par.sigma*sqrt(par.T)*6);
        
	cout<< "Smin= "<< Smin<< "\t e Smax= "<< Smax<< endl;
// 	xmin=0;
//         xmax=0;
        
//         dx=(log(Smax/par.S0)-log(Smin/par.S0))/pow(2., refs);
//         cout<<"dx "<<dx<<"\n";
        
//         while (xmin>log(Smin/par.S0))
//                 xmin-=dx;
//         while (xmax<log(Smax/par.S0))
//                 xmax+=dx;
//         xmin-=dx;
//         xmax+=dx;
        
        Amin=0;
        Amax=Smax*par.T;
        
        Point<dim> p1(Smin,Amin);
        Point<dim> p2(Smax,Amax);
        
        std::vector<unsigned> refinement={pow(2,refs)+3, pow(2,refs)};
        
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
	
	
	typename Triangulation<dim>::cell_iterator
		cell = triangulation.begin (),
		endc = triangulation.end();
		
	for (; cell!=endc; ++cell)
		for (unsigned int face=0;
			 face<GeometryInfo<dim>::faces_per_cell;++face)
				if (cell->face(face)->at_boundary()) {
					if ((std::fabs(cell->face(face)->center()(0) - Smin) < toll) ||
						(std::fabs(cell->face(face)->center()(1) - Amax) < toll))
						cell->face(face)->set_boundary_indicator (0);
					if ((std::fabs(cell->face(face)->center()(0) - Smax) < toll) ||
						(std::fabs(cell->face(face)->center()(1) - Amin) < toll))
						cell->face(face)->set_boundary_indicator (1);
				}

}

template<int dim>
void Opzione<dim>::assemble_system() {
        
        QGauss<dim> quadrature_formula(4); // 2 nodes, 2d -> 4 quadrature points per cell
        
	FEValues<dim> fe_values (fe, quadrature_formula, update_values   | update_gradients | update_quadrature_points |
                                 update_JxW_values | update_jacobians); //  update_quadrature_points ?
        
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
        
        sigma_matrix[0][0]=0.;
        sigma_matrix[1][1]=0.;
        sigma_matrix[0][1]=0.;
        sigma_matrix[1][0]=0.;
        
        Tensor< 1 , dim, double > ones;
	
        for (unsigned i=0;i<dim;++i)
                ones[i]=1;
        
        Tensor< 1, dim, double > trasp;
		Tensor< 1, dim, double > divSIG;
        trasp[0]=0.;
        trasp[1]=0.;
		divSIG[1]=0.;
//         const Coefficient<dim> coefficient(par.r, par.S0);
	        std::vector<Point<dim> >    quad_points (n_q_points);
        
        // cell loop
        for (; cell !=endc;++cell) {
                
                fe_values.reinit(cell);
                
                cell_dd=0;
                cell_fd=0;
                cell_ff=0;
                cell_system=0;
                
                quad_points=fe_values.get_quadrature_points();
                
                for (unsigned q_point=0;q_point<n_q_points;++q_point)
                        for (unsigned i=0;i<dofs_per_cell;++i)
                                for (unsigned j=0; j<dofs_per_cell;++j) {
                                        
                                        trasp[0]=par.r*quad_points[q_point][0];
                                        trasp[1]=quad_points[q_point][0];
// 									cerr<< "got to here\n";
										sigma_matrix[0][0]=0.5*par.sigma*par.sigma*
														quad_points[q_point][0]*quad_points[q_point][0];
// 									cout<< fe_values.jacobian(q_point)[1][1]<< "\n";
// 										sigma_matrix[1][1]=0.5*fabs(trasp[1])/fe_values.jacobian(q_point)[1][1];
			  
									divSIG[0]=par.sigma*par.sigma*quad_points[q_point][0];
			  
// 									cerr<< "but not here";
                                        cell_dd(i, j)+=fe_values.shape_grad(i, q_point)*sigma_matrix*fe_values.shape_grad(j, q_point)*fe_values.JxW(q_point);
                                        cell_fd(i, j)+=fe_values.shape_value(i, q_point)*(ones*fe_values.shape_grad(j,q_point))*fe_values.JxW(q_point);
                                        cell_ff(i, j)+=fe_values.shape_value(i, q_point)*fe_values.shape_value(j, q_point)*fe_values.JxW(q_point);
                                        cell_system(i, j)+=fe_values.JxW(q_point)*
                                        (fe_values.shape_grad(i, q_point)*sigma_matrix*fe_values.shape_grad(j, q_point)+
                                        fe_values.shape_value(i, q_point)*divSIG*fe_values.shape_grad(j, q_point)
                                        -fe_values.shape_value(i, q_point)*(trasp*fe_values.shape_grad(j,q_point))+
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
	
	VectorTools::interpolate (dof_handler, PayOff<dim>(par.T), solution);
	
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
        
        Boundary_Condition<dim> bc;
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
                                                                  ZeroFunction<dim>(),
                                                                  boundary_values);
                        VectorTools::interpolate_boundary_values (dof_handler,
                                                                  1,
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

template<int dim>
double Opzione<dim>::get_price() {
        
        if (ran==false) {
                this->run();
        }
        
        // Creo nuova grigla ( che passi da (0,0) )
//         Triangulation<dim> price;
        // Creo degli fe
//         FE_Q<dim> fe2 (1);
        // Creo un DoFHandler e lo attacco a price
//         DoFHandler<dim> dof_handler_2 (price);
        // Costruisco la griglia, in modo che passi da (0,0) e non la rifinisco
//         GridGenerator::hyper_rectangle(price, Point<dim> (0.,par.S0), Point<dim> (Smax,Amax));
        // Assegno a dof_handler_2 gli elementi finit fe2 appena creati
//         dof_handler_2.distribute_dofs(fe2);
        // Definisco questa fantomatica funzione FEFieldFunction
        Functions::FEFieldFunction<dim> fe_function (dof_handler, solution);
        // Creo il vettore che conterrà i valori interpolati
//         Vector<double> solution_vector(4);
        // Interpolo
//         VectorTools::interpolate(dof_handler_2, fe_function, solution_vector);
        // Ritorno il valore interpolato della soluzione in (0,0)
        return fe_function.value(Point<dim>(par.S0, par.S0));
}

int main() {
	Parametri par;
	par.T=1.;
	par.K=100;
	par.S0=100;
	par.r=0.0367;
	par.sigma=0.120381;
        
        // Parametri della parte salto
        par.p=0.20761;           // Parametro 1 Kou
        par.lambda=0.330966;     // Parametro 2 Kou
        par.lambda_piu=9.65997;  // Parametro 3 Kou
        par.lambda_meno=3.13868; // Parametro 4 Kou
        
        // tempo // spazio
	Opzione<2> Call(par, 100, 5);
	double prezzo=Call.run();
        
        cout<<"Prezzo "<<prezzo<<"\n";
        cout<<"Asian 2d v000\n";
	
	return 0;
}