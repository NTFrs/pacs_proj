#ifndef __european_option_logprice_hpp
#define __european_option_logprice_hpp

#include "OptionBaseLogPrice.hpp"

//! A class for evaluation of European Options
/*!
 * This class evaluates the price of European Option, of type Put, Call, Basket Put and Basket Call.
 */
template <unsigned dim>
class EuropeanOptionLogPrice final: public OptionBaseLogPrice<dim> {
private:
        OptionType type;
        ExerciseType eu;
        
        virtual void solve();
public:
        //! 1d Constructor
        /*!
         * Constructor for European Option with payoff max(S_T-K,0) and max(K-S_T,0).
         * \param type_         Option Type, Call or Put
         * \param model         Model class pointer: it takes a pointer to a class inherited by the abstract class Model in "models.hpp"
         * \param r_            Interest rate
         * \param T_            Time to Maturity
         * \param K_            Strike Price
         * \param refs_         Refinement of the grid (e.g. insert 10 for 2^10=1024 cells)
         * \param time_step_    Number of TimeStep
         */
        EuropeanOptionLogPrice(OptionType type_,
                            Model * const model,
                            double r_,
                            double T_,
                            double K_,
                            unsigned refs_,
                            unsigned time_step_)
        :
        OptionBaseLogPrice<dim>::OptionBaseLogPrice(ExerciseType::EU, model, r_, T_, K_, refs_, time_step_),
        type(type_),
        eu(ExerciseType::EU)
        {};
        
        //! 2d Constructor
        /*!
         * Constructor for European Option with payoff max(S^1_T+S^2_T-K,0) and max(K-S^1_T-S^2_T,0).
         * \param type_         Option Type, Call or Put
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
        EuropeanOptionLogPrice(OptionType type_,
                            Model * const model1,
                            Model * const model2,
                            double rho_,
                            double r_,
                            double T_,
                            double K_,
                            unsigned refs_,
                            unsigned time_step_)
        :
        OptionBaseLogPrice<dim>::OptionBasePrice(ExerciseType::EU, model1, model2, rho_, r_, T_, K_, refs_, time_step_),
        type(type_),
        eu(ExerciseType::EU)
        {};
        
        
};

template <unsigned dim>
void EuropeanOptionLogPrice<dim>::solve ()
{
        VectorTools::interpolate (this->dof_handler,
                                  FinalCondition<dim>(this->K, this->type),
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
        
        BoundaryCondition<dim> bc(this->K, this->T,  this->r, this->type);
        
	cout<< "time step is"<< this->time_step << endl;
	
	for (double time=this->T-this->dt;time >=0;time-=this->dt, --Step) {
                
                cout<< "Step "<< Step<<"\t at time \t"<< time<< endl;
                
                //
                if (this->model_type!=OptionBase<dim>::ModelType::BlackScholes && dim==1) {
                        
                        Vector<double> J;
                        Vector<double> temp;
                        
                        (this->levy)->get_part2(J, this->solution, this->fe, this->dof_handler);
                        
                        (this->ff_matrix).vmult(this->system_rhs, J);
                        
                        temp.reinit(this->dof_handler.n_dofs());
                        
                        (this->system_M2).vmult(temp,this->solution);
                        
                        this->system_rhs+=temp;
                        
                }
                
                else
                        this->system_M2.vmult(this->system_rhs, this->solution);
                
                //
                
                bc.set_time(time);
                
                {
                        
                        std::map<types::global_dof_index,double> boundary_values;
                        
                        VectorTools::interpolate_boundary_values (this->dof_handler,
                                                                  0,
                                                                  bc,
                                                                  boundary_values);
                        
                        MatrixTools::apply_boundary_values (boundary_values,
                                                            *(this->system_matrix),
                                                            this->solution,
                                                            this->system_rhs, false);
                        
                }
                
                SparseDirectUMFPACK solver;
                solver.initialize(this->sparsity_pattern);
                solver.factorize(*(this->system_matrix));
                solver.solve(this->system_rhs);
                
                this->solution=this->system_rhs;
                
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