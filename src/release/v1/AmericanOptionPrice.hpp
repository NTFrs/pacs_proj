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
        //! 1d Constructor
        /*!
         * Constructor for American Option with payoff max(K-S_T,0).
         * \param model         Model class pointer: it takes a pointer to a class inherited by the abstract class Model in "models.hpp"
         * \param r_            Interest rate
         * \param T_            Time to Maturity
         * \param K_            Strike Price
         * \param refs_         Refinement of the grid (e.g. insert 10 for 2^10=1024 cells)
         * \param time_step_    Number of TimeStep
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
        
        //! 1d Constructor
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
        
        
};

template <unsigned dim>
void AmericanOptionPrice<dim>::solve ()
{
        using namespace dealii;
        using namespace std;
        
        VectorTools::interpolate (this->dof_handler,
                                  FinalConditionPrice<dim>(this->K, OptionType::Put),
                                  this->solution);
        
        
        {
                DataOut<dim> data_out;
                
                data_out.attach_dof_handler (this->dof_handler);
                data_out.add_data_vector (this->solution, "begin");
                
                data_out.build_patches ();
                
                std::ofstream output ("begin.gpl");
                data_out.write_gnuplot (output);
        }
        
	unsigned Step=this->time_step;
        
        BoundaryConditionPrice<dim> bc(this->K, this->T,  this->r, OptionType::Put);
        
	cout<< "time step is"<< this->time_step << endl;
	
	for (double time=this->T-this->dt;time >=0;time-=this->dt, --Step) {
                
                cout<< "Step "<< Step<<"\t at time \t"<< time<< endl;
                this->system_M2.vmult(this->system_rhs, this->solution);
                bc.set_time(this->dt);
                
                {
                        
                        std::map<types::global_dof_index,double> boundary_values;
                        
                        VectorTools::interpolate_boundary_values (this->dof_handler,
                                                                  0,
                                                                  bc,
                                                                  boundary_values);
                        
                        MatrixTools::apply_boundary_values (boundary_values,
                                                            (this->system_matrix),
                                                            this->solution,
                                                            this->system_rhs, false);
                        
                }
                
                unsigned maxiter=1000;
                double tollerance=constants::high_toll;
                
                Vector<double> solution_old=this->solution;
                
                bool converged=false;
                
                for (unsigned k=0; k<maxiter && !converged; ++k) {
                        
                        this->system_matrix.ProjectedSOR_step(this->solution,
                                                              solution_old,
                                                              this->system_rhs,
                                                              this->vertices,
                                                              this->K);
                        
                        auto temp=this->solution;
                        temp.add(-1, solution_old);
                        
                        if (temp.linfty_norm()<tollerance){
                                converged=true;
                        }
                        
                        else
                                solution_old=this->solution;
                        
                        
                        if (k==maxiter) {
                                cout<<"Warning: maxiter reached. Error="<<temp.linfty_norm()<<"\n";
                        }
                        
                }
                
        }
        
        {
                DataOut<dim> data_out;
                
                data_out.attach_dof_handler (this->dof_handler);
                data_out.add_data_vector (this->solution, "end");
                
                data_out.build_patches ();
                
                std::ofstream output ("end.gpl");
                data_out.write_gnuplot (output);
        }
        
	this->ran=true;
        
}

#endif