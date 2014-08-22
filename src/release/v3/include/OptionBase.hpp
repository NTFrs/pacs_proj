#ifndef __option_base_hpp
#define __option_base_hpp

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
#include <sys/time.h>
#include <iomanip>

#include "dealii.hpp"
#include "SparseMatrix_PSOR.hpp"
#include "BoundaryConditionsPrice.hpp"
#include "BoundaryConditionsLogPrice.hpp"
#include "FinalConditionsPrice.hpp"
#include "FinalConditionsLogPrice.hpp"
#include "OptionTypes.hpp"
#include "Models.hpp"
#include "Constants.hpp"

//! Abstract base class for the option-type classes.
/*!
 * This class stores all the protected members of the option-type class and defines the layout of these classes.
 */
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
	
        dealii::ConstraintMatrix	constraints;
        
        // Points of the grid
        std::vector< dealii::Point<dim> >       vertices;
	
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
        
        unsigned                id;
        
        // PSOR parameters
        double                  tollerance;
        unsigned                maxiter;
        double                  omega;
        
        // Integral Part
        std::unique_ptr< LevyIntegralBase<dim> > levy;
        bool                    integral_adapt;
        unsigned                order;
        unsigned                order_max;
        double                  alpha_toll;
        double                  J_toll;
        
        // Time parameters
        double                  clock_time;
        double                  real_time;
        bool                    timing;
        
        // Output parameters
        bool                    verbose;
        bool                    print;
        bool                    print_grids;
        
        // Protected methods
        virtual void setup_system();
        virtual void refine_grid();
        // Pure abstract methods
        virtual void make_grid() = 0;
        virtual void assemble_system() = 0;
        virtual void solve() = 0;
        virtual void setup_integral() = 0;
        virtual void print_solution_matlab(std::string name_) = 0;
        
        // Output methods
        virtual void print_grid(unsigned step) {
                
                std::string name("plot/Mesh-");
                name.append(std::to_string(this->id));
                name.append("-");
                name.append(std::to_string(step));
                name.append(".eps");
                
                std::ofstream out (name);
                dealii::GridOut grid_out;
                grid_out.write_eps (triangulation, out);
                
        };
        
        virtual void print_solution_gnuplot(std::string name_) {
                
                dealii::DataOut<dim> data_out;
                data_out.attach_dof_handler(dof_handler);
                data_out.add_data_vector(solution, name_);
                
                std::string name("gnuplot/");
                name.append(name_);
                name.append("-");
                name.append(std::to_string(this->id));
                name.append(".gpl");
                data_out.build_patches();
                std::ofstream out(name);
                data_out.write_gnuplot(out);
                
        };
        
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
        
        //! Constructor 2d
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
        virtual void run();
        
        //!
        /*!
         * This function is used to set the scale factor f of the grid limits, calulated in this way:
         * \f[ S_{min}=(1-f)S_0exp\left( \left(r-\frac{\sigma^2}{2}\right)T-6\sigma\sqrt{T}\right), \f]
         * \f[ S_{max}=(1+f)S_0exp\left( \left(r-\frac{\sigma^2}{2}\right)T+6\sigma\sqrt{T}\right). \f]
         * \param f_    is set by default to 0.5. The user can specify it in ]0,1[.
         */
        virtual void set_scale_factor(double f_) {
                if (f_<=0. && f_>=1.) {
                        throw(std::logic_error("Error! The scale factor must be in ]0,1[.\n"));
                }
                else
                        f=f_;
        };
        
        //!
        /*! This simple function allows to set the refinement of the grid and reset the object.
         */
        virtual void set_refs(unsigned refs_) {
                refs=refs_;
                reset();
        }
        
        //!
        /*! This function allows to set some adaptivity parameters
         * \param integral_adapt_       true for adapting, false for not adapting. True is default.
         * \param order_                If integral_adapt_ is true, this is the starting order of the integral formula. If integral_adapt_ is false, this is the order of the integral formula.
         * \param order_max_            Max number of integration nodes
         * \param alpha_toll_           Tollerance for alpha
         * \param J_toll_               Tollerance for J
         * \note J_toll is only used in LogPrice transformation. Any changes in J_toll will not affect the OptionBasePrice-type classes.
         * \note For the LogPrice trasformations, a high number for order_max or a small tollerance for J will slow down the integral calculations.
         */
        virtual void set_integral_adaptivity_params(bool integral_adapt_,
                                                    unsigned order=16,
                                                    unsigned order_max_=32,
                                                    double alpha_toll_=constants::light_toll,
                                                    double J_toll_=constants::light_toll)
        {
                this->integral_adapt=integral_adapt_;
                this->order_max=order_max_;
                this->alpha_toll=alpha_toll_;
                this->J_toll=J_toll_;
        }
        
        //!
        /*! This simple function allows to set the number of time step and reset the object.
         */
        virtual void set_timestep(unsigned time_step_) {
                time_step=time_step_;
                dt=T/static_cast<double>(time_step_);
                reset();
        }
        
        //!
        /*! This function allows to set the adaptivity of the grid. Default is false.
         * \param status        to set adaptivity, set status=true
         * \param refine_       refinement index, e.g. if refine_=0.2, during every refinement step, the program will refine the 20% of the cells.
         * \param coarse_       coarsening index, e.g. if coarse_=0.03, during every refinement step, the program will coarse the 3% of the cells.
         */
        virtual void set_refine_status(bool status, float refine_=0.2, float coarse_=0.03) {
                refine=status;
                refine_index=refine_;
                coarse_index=coarse_;
                reset();
        };
        
        //!
        /*! This function allows to clock the execution of the method run(). Default is false.
         * \note This function must be set before running the calculation, not after.
         */
        virtual void set_timing(bool timing_) {
                timing=timing_;
                reset();
        }
        
        //!
        /*! This function allows the user to set the PSOR parameters used in American Options.
         * \param tollerance_   The tollerance for stopping the PSOR iterations
         * \param maxiter       Max number of iteration, after this number the solver iteration stops, even if the tollerance has not been reached.
         * \param omega_        SOR relaxation parameter, must be in ]0, 2[.
         */
        virtual void set_PSOR_parameters(double tollerance_,
                                         unsigned maxiter_=1000,
                                         double omega_=1.)
        {
                if (type!=ExerciseType::US) {
                        throw(std::logic_error("PSOR is used only in American Options."));
                }
                else if (omega_>2. || omega_<0.) {
                        throw(std::logic_error("omega_ must be in ]0,2[."));
                }
                else if (tollerance_<0.) {
                        throw(std::logic_error("tollerance_ must be in ]0,2[.")); 
                }
                else {
                        tollerance=tollerance_;
                        maxiter=maxiter_;
                        omega=omega_;
                }
        }
        
        //!
        /*! Set the verbosity of the Option: false for nothing, true for everything. Default is true.
         */
        virtual void set_verbose(bool verbose_) {
                verbose=verbose_;
        }
        
        //!
        /*! This function tells the program to print the final condition and the solution of the problem in matlab and gnuplot format.
         */
        virtual void set_print(bool print_) {
                print=print_;
        }
        
        //!
        /*! This function tells the program to print an image of the 2d-mesh in the folder "plot".
         */
        virtual void set_print_grid(bool print_) {
                if (dim==1) {
                        throw(std::logic_error("Error! This program cannot print 1d grids.\n"));
                }
                else {
                        print_grids=print_;
                }
        }
        
        //!
        /*! This function return the number of cells in the mesh.
         */
        virtual unsigned long get_number_of_cells() {
                return  dim==1
                ?
                pow(2, refs)
                :
                pow(2, 2*refs);
        }
        
        //!
        /*! This function return the number timesteps.
         */
        virtual unsigned long get_number_of_timesteps() {
                return time_step;
        }
        
        //!
        /*! If timing is true, this function returns a pair of times in microseconds,
         *  the clock time and the real time taken by the class to solve the system
         */
        virtual std::pair<double, double> get_times() {
                if (clock_time==log(-1)) {
                        throw(std::logic_error("Error! The flag timing must be set BEFORE running the calculation.\n"));
                }
                else if (!timing) {
                        throw(std::logic_error("Error! The flag timing must be set BEFORE running the calculation.\n"));
                }
                else {
                        auto times=std::make_pair(clock_time, real_time);
                        return times;
                }
                
                
        }
        
        //!
        /*! This function allows to get a copy of the solution.
         */
        virtual void get_solution(std::vector<double> &sol){
                sol.resize(solution.size());
                for (unsigned i=0; i<solution.size(); ++i) {
                        sol[i]=solution(i);
                }
        }
        
        //!
        /*! This function allows the get a copy of the mesh.
         */
        virtual void get_mesh(std::vector< dealii::Point<dim> > &mesh) {
                mesh=vertices;
        }
        
        //!
        /*! This function allows to reset the class, in order to run again the solve method.
         */
        virtual void reset() {
                ran=false;
                triangulation.clear();
                system_matrix.clear();
                system_M2.clear();
        }
        
        //!
        /*!
         * This function returns the price of the option.
         */
        virtual double get_price()=0;
        
};

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
refine(false),
tollerance(constants::high_toll),
maxiter(1000),
omega(1.),
integral_adapt(true),
order(16),
order_max(32),
alpha_toll(constants::light_toll),
J_toll(constants::light_toll),
clock_time(log(-1)),
real_time(log(-1)),
timing(false),
verbose(true),
print(false),
print_grids(false)
{
        tools::Counter::counter()++;
        id=tools::Counter::counter();
        
        // if this is the first option instantiated, create the folders "gnuplot", "matlab" and "plot"
        if (id==1) {
                if (system(NULL)) {
                        if(system("mkdir -p plot && mkdir -p gnuplot && mkdir -p matlab"));
                }
                else {
                        std::exit(-1);
                }
        }
        
        models.push_back(model);
        
        // setting the kind of model...
        if (model->get_type()==0) 
                model_type=ModelType::BlackScholes;
        
        else if (model->get_type()==1) {
                model_type=ModelType::Kou;
        }
        
        else if (model->get_type()==2) {
                model_type=ModelType::Merton;
        }
        else {
                throw(std::logic_error("Error! Unknown models.\n"));
        }
        
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
refine(false),
tollerance(constants::high_toll),
maxiter(1000),
omega(1.),
integral_adapt(true),
order(16),
order_max(32),
alpha_toll(constants::light_toll),
J_toll(constants::light_toll),
clock_time(log(-1)),
real_time(log(-1)),
timing(false),
verbose(true),
print(false),
print_grids(false)
{
        tools::Counter::counter()++;
        id=tools::Counter::counter();
        
        // if this is the first option instantiated, create the folders "gnuplot", "matlab" and "plot"
        if (id==1) {
                if (system(NULL)) {
                        if(system("mkdir -p plot && mkdir -p gnuplot && mkdir -p matlab"));
                }
                else {
                        std::exit(-1);
                }
        }
        
        models.push_back(model1);
        models.push_back(model2);
        
        // setting the kind of model...
        if (model1->get_type()==0) {
                model_type=ModelType::BlackScholes;
                if (model2->get_type()!=0)
                        throw(std::logic_error("Error! Different types of model.\n"));
        }
        else if (model1->get_type()==1) { 
                model_type=ModelType::Kou;
                if (model2->get_type()!=1)
                        throw(std::logic_error("Error! Different types of model.\n"));
        }
        else if (model1->get_type()==2) {
                model_type=ModelType::Merton;
                if (model2->get_type()!=2)
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
        
        if (verbose) {
                std::cout << "Building system...\nNumber of degrees of freedom: "<<
                dof_handler.n_dofs()<<"\n";
        }
        
        this->constraints.clear();
        dealii::DoFTools::make_hanging_node_constraints(this->dof_handler, this->constraints);
        this->constraints.close();
        
	dealii::CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
	dealii::DoFTools::make_sparsity_pattern (dof_handler, c_sparsity, this->constraints, false);
        
	sparsity_pattern.copy_from(c_sparsity);
        
        dd_matrix.reinit(sparsity_pattern);
	fd_matrix.reinit(sparsity_pattern);
	ff_matrix.reinit(sparsity_pattern);
	system_matrix.reinit(sparsity_pattern);
	system_M2.reinit(sparsity_pattern);
        
	solution.reinit(dof_handler.n_dofs());
	system_rhs.reinit(dof_handler.n_dofs());
        
        vertices.clear();
        vertices.resize(dof_handler.n_dofs());
        dealii::DoFTools::map_dofs_to_support_points(dealii::MappingQ1<dim>(), this->dof_handler, this->vertices);
        
        return;
        
}

template<unsigned dim>
void OptionBase<dim>::refine_grid()
{
        using namespace dealii;
        
        Vector<float> estimated_error_per_cell (this->triangulation.n_active_cells());
	KellyErrorEstimator<dim>::estimate (this->dof_handler, QGauss<dim-1>(3),
                                            typename FunctionMap<dim>::type(),
                                            this->solution,
                                            estimated_error_per_cell);
        
	GridRefinement::refine_and_coarsen_fixed_number (this->triangulation,
                                                         estimated_error_per_cell,
                                                         this->refine_index, this->coarse_index);
	
	SolutionTransfer<dim> solution_trans(this->dof_handler);
	Vector<double> previous_solution;
	previous_solution = this->solution;
	this->triangulation.prepare_coarsening_and_refinement();
	solution_trans.prepare_for_coarsening_and_refinement(previous_solution);
        
	this->triangulation.execute_coarsening_and_refinement();
	this->setup_system ();
        
	solution_trans.interpolate(previous_solution, this->solution);
	this->assemble_system();
}


template<unsigned dim>
void OptionBase<dim>::run()
{
        if (ran==false) {
                clock_t clock_s=0;
                clock_t clock_e=0;
                struct timeval real_s, real_e;
                
                if (timing) {
                        gettimeofday(&real_s, NULL);
                        clock_s=clock();
                }
                
                make_grid();
                setup_system();
                setup_integral();
                assemble_system();
                solve();
                
                if (timing) {
                        gettimeofday(&real_e, NULL);
                        clock_e=clock();
                        
                        clock_time=static_cast<double> (((clock_e-clock_s)*1.e6)/CLOCKS_PER_SEC);
                        real_time=((real_e.tv_sec-real_s.tv_sec)*1.e6+real_e.tv_usec - real_s.tv_usec);
                }
                
                ran=true;
        }
        
}


#endif