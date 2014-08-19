#ifndef __american_option_price_hpp
#define __american_option_price_hpp

#include "OptionBase.hpp"

//! A class for evaluation of American Options
/*!
 * This class evaluates the price of American Option, of type Put and Basket Put
 */
template <unsigned dim>
class AmericanOptionPrice final: public OptionBasePrice<dim>
{
private:
        ExerciseType us;
        
        virtual void solve();
public:
        AmericanOptionPrice()=delete;
        
        AmericanOptionPrice(const AmericanOptionPrice &)=delete;
        
        //! 1d Constructor
        /*!
         * Constructor for American Option with payoff max(K-S_T,0).
         * \param model         Model class pointer: it takes a pointer to a class inherited by the abstract class Model in "models.hpp"
         * \param r_            Interest rate
         * \param T_            Time to Maturity
         * \param K_            Strike Price
         * \param refs_         Refinement of the grid (e.g. insert 10 for 2^10=1024 cells)
         * \param time_step_    Number of TimeStep
         * \note The objects "model" are passed through a pointer and addressed this way in the library. So, please, do not destroy them before destroying this object.
         */
        AmericanOptionPrice(Model * const model,
                            double r_,
                            double T_,
                            double K_,
                            unsigned refs_,
                            unsigned time_step_)
        :
        OptionBasePrice<dim>::OptionBasePrice(ExerciseType::US, model, r_, T_, K_, refs_, time_step_),
        us(ExerciseType::US)
        {};
        
        //! 2d Constructor
        /*!
         * Constructor for American Option with payoff max(K-S^1_T-S^2_T,0).
         * \param model1        Model class pointer: it takes a pointer to a class inherited by the abstract class Model in "models.hpp"
         * \param model2        Model class pointer: it takes a pointer to a class inherited by the abstract class Model in "models.hpp"
         * \param rho_          Correlation between Stocks
         * \param r_            Interest rate
         * \param T_            Time to Maturity
         * \param K_            Strike Price
         * \param refs_         Refinement of the grid (e.g. insert 8 for 2^(2*6)=4096 cells)
         * \param time_step_    Number of TimeStep
         * \note                model1 and model2 MUST be of the same type.
         * \note The objects "model" are passed through a pointer and addressed this way in the library. So, please, do not destroy them before destroying this object.
         */
        AmericanOptionPrice(Model * const model1,
                            Model * const model2,
                            double rho_,
                            double r_,
                            double T_,
                            double K_,
                            unsigned refs_,
                            unsigned time_step_)
        :
        OptionBasePrice<dim>::OptionBasePrice(ExerciseType::US, model1, model2, rho_, r_, T_, K_, refs_, time_step_),
        us(ExerciseType::US)
        {};
        
        AmericanOptionPrice& operator=(const AmericanOptionPrice &)=delete;
        
};

template <unsigned dim>
void AmericanOptionPrice<dim>::solve ()
{
        using namespace dealii;
        using namespace std;
        
        VectorTools::interpolate (this->dof_handler,
                                  FinalConditionPrice<dim>(this->K, OptionType::Put),
                                  this->solution);
        
	unsigned Step=this->time_step;
        
        if (dim==2 && this->print_grids) {
                this->print_grid(Step);
        }
        
        if (this->print) {
                this->print_solution_gnuplot("begin");
                this->print_solution_matlab("begin");
        }
        
        BoundaryConditionPrice<dim> bc(this->K, this->T,  this->r, OptionType::Put);
        
	for (double time=this->T-this->dt; Step>0 ;time-=this->dt, --Step) {
                
                if (this->verbose) {
                        cout<< "Step "<<Step<<"\t at time "<<time<<"\n";
                }
                
                if (this->refine && Step%20==0 && Step!=this->time_step) {
                        this->refine_grid();

                        
                        if (dim==2 && this->print_grids) {
                                this->print_grid(Step);
                        }
                }
                if (this->model_type!=OptionBase<dim>::ModelType::BlackScholes) {
                        
                        Vector<double> *J_x;
                        Vector<double> *J_y;
                        Vector<double> temp;
                        
                        this->levy->compute_J(this->solution, this->dof_handler, this->fe, this->vertices);
                        
                        if (dim==1)
                                this->levy->get_j_1(J_x);
                        else
                                this->levy->get_j_both(J_x, J_y);
                        
                        (this->system_M2).vmult(this->system_rhs,this->solution);
                        
                        temp.reinit(this->dof_handler.n_dofs());
                        
                        (this->ff_matrix).vmult(temp, *J_x);
                        
                        this->system_rhs+=temp;
                        
                        if (dim==2) {
                                temp.reinit(this->dof_handler.n_dofs());
                                
                                this->ff_matrix.vmult(temp, *J_y);
                                
                                this->system_rhs+=temp;
                        }
                        
                }
                
                else
                        this->system_M2.vmult(this->system_rhs, this->solution);
                
                bc.set_time(this->T);
                
                {
                        
                        std::map<types::global_dof_index,double> boundary_values;
                        
                        VectorTools::interpolate_boundary_values (this->dof_handler,
                                                                  0,
                                                                  bc,
                                                                  boundary_values);
                        if (dim==1)                                              
                                VectorTools::interpolate_boundary_values (this->dof_handler,1,bc,boundary_values);
                        
                        MatrixTools::apply_boundary_values (boundary_values,
                                                            (this->system_matrix),
                                                            this->solution,
                                                            this->system_rhs, false);
                        
                }
                
                Vector<double> solution_old=this->solution;
                
                bool converged=false;
                
                for (unsigned k=0; k<this->maxiter && !converged; ++k) {
                        
                        this->system_matrix.ProjectedSOR_step(this->solution,
                                                              solution_old,
                                                              this->system_rhs,
                                                              this->vertices,
                                                              FinalConditionPrice<dim>(this->K, OptionType::Put),
                                                              this->omega);
                                                              
						this->constraints.distribute(this->solution);
						this->constraints.distribute(solution_old);
                        auto temp=this->solution;
                        temp.add(-1, solution_old);
                        
                        if (temp.linfty_norm()<this->tollerance){
                                converged=true;
                        }
                        
                        else
                                solution_old=this->solution;
                        
                        
                        if (k==this->maxiter-1) {
                                cout<<"Warning: maxiter reached, with error="<<temp.linfty_norm()<<"\n";
                        }
                        
                }

                
        }
        
        if (this->print) {
                this->print_solution_gnuplot("end");
                this->print_solution_matlab("end");
        }
        
}

#endif