#ifndef __option_base_hpp
#define __option_base_hpp

#include "deal_ii.hpp"
#include "matrix_with_psor.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <climits>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <string>
#include <memory>
#include <exception>

#include "BoundaryConditionsPrice.hpp"
#include "BoundaryConditionsLogPrice.hpp"
#include "FinalConditionsPrice.hpp"
#include "FinalConditionsLogPrice.hpp"
#include "OptionTypes.hpp"
#include "models.hpp"
#include "constants.hpp"
#include "OptionParameters.hpp"

using namespace dealii;

template<unsigned dim>
class OptionBase
{
protected:
        // Model and Option parameters
        ExerciseType            type;
        std::vector<Model *>    models;
        double                  rho;
        double                  r;
        double                  T;
        double                  K;
        
        enum class ModelType
        {
                BlackScholes,
                Merton,
                Kou
        };
        
        ModelType               model_type;
        
        // Triangulation and fe objects
        dealii::Triangulation<dim>      triangulation;
	dealii::FE_Q<dim>               fe;
	dealii::DoFHandler<dim>         dof_handler;
        
        // Matrices
        dealii::SparsityPattern         sparsity_pattern;
	
        dealii::SparseMatrix_PSOR<double, dim> system_matrix;
	
        dealii::SparseMatrix<double>    system_M2;
	dealii::SparseMatrix<double>    dd_matrix;
	dealii::SparseMatrix<double>    fd_matrix;
	dealii::SparseMatrix<double>    ff_matrix;
        
        // points of grid
        std::vector< Point<dim> >       grid_points;
        std::map<dealii::types::global_dof_index, dealii::Point<dim> > vertices;
	
        // Solution and rhs vectors
	dealii::Vector<double>          solution;
	dealii::Vector<double>          system_rhs;
        
        // Mesh boundaries
        dealii::Point<dim>              Smin, Smax;
        
        // Disctretization parameters
        unsigned                refs;      // space
        unsigned                time_step; // time
        double                  dt;
        double                  price;
        double                  f;
        bool                    ran;
        
        bool 			refine;
        float                   coarse_index;
        float                   refine_index;
        
        static unsigned         id;
        
        // Integral Part
        std::unique_ptr< LevyIntegralBase<dim> > levy;       
        
        // Protected methods
        virtual void setup_system();
        virtual void refine_grid();
        // Pure abstract methods
        virtual void make_grid() = 0;
        virtual void assemble_system() = 0;
        virtual void solve() = 0;
        virtual void setup_integral() = 0;
        
public:
        OptionBase()=delete;
        
        OptionBase(const OptionBase &)=delete;
        
        //! Constructor 1d
        /*!
         * Constructor 1d called by inherited classes.
         */
        OptionBase(ExerciseType type_,
                   Model * const model,
                   double r_,
                   double T_,
                   double K_,
                   unsigned refs_,
                   unsigned time_step_);
        
        //! Cosntructor 2d
        /*!
         * Constructor 2d called by inherited classes.
         */
        OptionBase(ExerciseType type_,
                   Model * const model1,
                   Model * const model2,
                   double rho_,
                   double r_,
                   double T_,
                   double K_,
                   unsigned refs_,
                   unsigned time_step_);
        
        //! Destructor
        virtual ~OptionBase()=default;
        
        OptionBase& operator=(const OptionBase &)=delete;
        
        //! 
        /*!
         * This function creates the system and solves it.
         */
        virtual void run()
        {
                make_grid();
                setup_system();
                setup_integral();
                assemble_system();
                solve();
        };
        
        //!
        /*!
         * This function is used to set the scale factor of the grid boundary.
         * \param f_    is set by default to 0.5. The user can specify it in ]0,1[.
         */
        virtual void set_scale_factor(double f_) {
                if (f_<=0. && f_>=1.) {
                        throw(std::logic_error("Error! The scale factor must be in ]0,1[.\n"));
                }
                else
                        f=f_;
        };
        
        virtual void set_refine_status(bool status, float refine_=0.2, float coarse_=0.03) {
                refine=status;
                refine_index=refine_;
                coarse_index=coarse_;
        };
        
        virtual bool get_refine_status() {
                return refine;
        };
        
        //!
        /*!
         * This function returns the price of the option
         */
        virtual inline double get_price()=0;
        
        virtual void estimate_doubling(double time, Vector<float> & errors);
        
        //TODO add output functions
        
        virtual void print_grid(std::string name) {
                name.append(".eps");
                std::ofstream out (name);
                GridOut grid_out;
                grid_out.write_eps (triangulation, out);
        };
        
        virtual void print_solution_gnuplot(std::string name) {
                DataOut<dim> data_out;
                data_out.attach_dof_handler(dof_handler);
                data_out.add_data_vector(solution, name);
                
                name.append(".gpl");
                data_out.build_patches();
                std::ofstream out(name);
                data_out.write_gnuplot(out);
        };
        
};

template<unsigned dim>
unsigned OptionBase<dim>::id=1;

// Constructor 1d
template <unsigned dim>
OptionBase<dim>::OptionBase(ExerciseType type_,
                            Model * const model,
                            double r_,
                            double T_,
                            double K_,
                            unsigned refs_,
                            unsigned time_step_)
{
        throw(std::logic_error("Error! Dimension must be 1 or 2.\n"));
}

// Constructor 1d specialized
template <>
OptionBase<1>::OptionBase(ExerciseType type_,
                          Model * const model,
                          double r_,
                          double T_,
                          double K_,
                          unsigned refs_,
                          unsigned time_step_)
:
type(type_),
r(r_),
T(T_),
K(K_),
fe (1),
dof_handler (triangulation),
refs(refs_),
time_step(time_step_),
dt(T/static_cast<double>(time_step_)),
price(0.),
f(0.5),
ran(false), 
refine(false)
//levy(NULL)
{
        ++id;
        
        if (system( NULL ))
                system("mkdir -p plot");
        else
                std::exit(-1);
        
        models.push_back(model);
        
        BlackScholesModel       *     bs(dynamic_cast<BlackScholesModel *> (model));
        KouModel                *     kou(dynamic_cast<KouModel *> (model));
        MertonModel             *     mer(dynamic_cast<MertonModel *> (model));
        
        if (bs) 
                model_type=ModelType::BlackScholes;
        
        else if (kou) {
                model_type=ModelType::Kou;
        }
        
        else if (mer)
                model_type=ModelType::Merton;
        
        else    
                throw(std::logic_error("Error! Unknown models.\n"));
        
}

// Constructor 2d
template <unsigned dim>
OptionBase<dim>::OptionBase(ExerciseType type_,
                            Model * const model1,
                            Model * const model2,
                            double rho_,
                            double r_,
                            double T_,
                            double K_,
                            unsigned refs_,
                            unsigned time_step_)
{
        throw(std::logic_error("Error! Dimension must be 1 or 2.\n"));
}

// Constructor 2d specialized
template <>
OptionBase<2>::OptionBase(ExerciseType type_,
                          Model * const model1,
                          Model * const model2,
                          double rho_,
                          double r_,
                          double T_,
                          double K_,
                          unsigned refs_,
                          unsigned time_step_)
:
type(type_),
rho(rho_),
r(r_),
T(T_),
K(K_),
fe (1),
dof_handler (triangulation),
refs(refs_),
time_step(time_step_),
dt(T/static_cast<double>(time_step_)),
price(0.),
f(0.5),
ran(false), 
refine(false)
//levy(NULL)
{
        ++id;
        
        if (system( NULL ))
                system("mkdir -p plot");
        else
                std::exit(-1);
        
        models.push_back(model1);
        models.push_back(model2);
        
        BlackScholesModel       *     bs(dynamic_cast<BlackScholesModel *> (model1));
        KouModel                *     kou(dynamic_cast<KouModel *> (model1));
        MertonModel             *     mer(dynamic_cast<MertonModel *> (model1));
        
        if (bs) {
                model_type=ModelType::BlackScholes;
                BlackScholesModel * bs2(dynamic_cast<BlackScholesModel *> (model2));
                if (!bs2)
                        throw(std::logic_error("Error! Different types of model.\n"));
        }
        else if (kou) { 
                model_type=ModelType::Kou;
                KouModel * kou2(dynamic_cast<KouModel *> (model2));
                if (!kou2)
                        throw(std::logic_error("Error! Different types of model.\n"));
        }
        else if (mer) {
                model_type=ModelType::Merton;
                MertonModel * mer2(dynamic_cast<MertonModel *> (model2));
                if (!mer2)
                        throw(std::logic_error("Error! Different types of model.\n"));
        }
        else    
                throw(std::logic_error("Error! Unknown models.\n"));
        
}


// setup system
template<unsigned dim>
void OptionBase<dim>::setup_system()
{
        
	dof_handler.distribute_dofs(fe);
        
	std::cout << "   Number of degrees of freedom: "
	<< dof_handler.n_dofs()
	<< std::endl;
        
	dealii::CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
	dealii::DoFTools::make_sparsity_pattern (dof_handler, c_sparsity);
        
	sparsity_pattern.copy_from(c_sparsity);
        
        dd_matrix.reinit(sparsity_pattern);
	fd_matrix.reinit(sparsity_pattern);
	ff_matrix.reinit(sparsity_pattern);
	system_matrix.reinit(sparsity_pattern);
	system_M2.reinit(sparsity_pattern);
        
	solution.reinit(dof_handler.n_dofs());
	system_rhs.reinit(dof_handler.n_dofs());
        
        vertices.clear();
	DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dof_handler, vertices);
        
        return;
        
}

template<unsigned dim>
void OptionBase<dim>::refine_grid()
{
	Vector<float> estimated_error_per_cell (this->triangulation.n_active_cells());
	KellyErrorEstimator<dim>::estimate (this->dof_handler, QGauss<dim-1>(3),
                                            typename FunctionMap<dim>::type(),
                                            this->solution,
                                            estimated_error_per_cell);
        
	GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                         estimated_error_per_cell,
                                                         refine_index, coarse_index);
	
	SolutionTransfer<dim> solution_trans(this->dof_handler);
	Vector<double> previous_solution;
	previous_solution = this->solution;
	this->triangulation.prepare_coarsening_and_refinement();
	solution_trans.prepare_for_coarsening_and_refinement(previous_solution);
        
	this->triangulation.execute_coarsening_and_refinement ();
	this->setup_system ();
        
	solution_trans.interpolate(previous_solution, solution);
	this->assemble_system();
}


template<unsigned dim>
void OptionBase<dim>::estimate_doubling(double time, Vector< float >& errors)
{
	using namespace dealii;
	
	using namespace std;
	
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
	//TODO need a solve_one_step here if using this
        // solve_one_step(time);
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



#endif