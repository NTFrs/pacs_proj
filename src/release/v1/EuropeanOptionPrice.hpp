#ifndef __european_option_price_hpp
#define __european_option_price_hpp

#include "OptionBasePrice.hpp"

//! A class for evaluation of European Options
/*!
 * This class evaluates the price of European Option, of type Put, Call, Basket Put and Basket Call.
 */
template <unsigned dim>
class EuropeanOptionPrice final: public OptionBasePrice<dim> {
private:
        OptionType type2;
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
        EuropeanOptionPrice(OptionType type_,
                            Model * const model,
                            double r_,
                            double T_,
                            double K_,
                            unsigned refs_,
                            unsigned time_step_)
        :
        OptionBasePrice<dim>::OptionBasePrice(ExerciseType::EU, model, r_, T_, K_, refs_, time_step_),
        type2(type_),
        eu(ExerciseType::EU)
        {};
        
        //! 1d Constructor
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
        EuropeanOptionPrice(OptionType type_,
                            Model * const model1,
                            Model * const model2,
                            double rho_,
                            double r_,
                            double T_,
                            double K_,
                            unsigned refs_,
                            unsigned time_step_)
        :
        OptionBasePrice<dim>::OptionBasePrice(ExerciseType::EU, model1, model2, rho_, r_, T_, K_, refs_, time_step_),
        type2(type_),
        eu(ExerciseType::EU)
        {};
        
        
};

template <unsigned dim>
void EuropeanOptionPrice<dim>::solve ()
{
        using namespace dealii;
        using namespace std;
        
        VectorTools::interpolate (this->dof_handler,
                                  FinalConditionPrice<dim>(this->K, this->type2),
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
        
        BoundaryConditionPrice<dim> bc(this->K, this->T,  this->r, this->type2);
        
	cout<< "time step is"<< this->time_step << endl;
	
	for (double time=this->T-this->dt;time >=0;time-=this->dt, --Step) {
                
                cout<< "Step "<< Step<<"\t at time \t"<< time<< endl;
                
                //
                if (this->model_type!=OptionBase<dim>::ModelType::BlackScholes) {
                        
                        Vector<double> *J_x;
                        Vector<double> *J_y;
                        Vector<double> temp;
                        
                        this->levy->compute_J(this->solution, this->dof_handler, this->fe);
                        
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
                
                //
                
                bc.set_time(this->dt);
                
                {
                        
                        std::map<types::global_dof_index,double> boundary_values;
                        
                        VectorTools::interpolate_boundary_values (this->dof_handler,
                                                                  0,
                                                                  bc,
                                                                  boundary_values);
                        
                        if (dim==1) {
                                VectorTools::interpolate_boundary_values (this->dof_handler,
                                                                          1,
                                                                          bc,
                                                                          boundary_values);
                        }
                        
                        MatrixTools::apply_boundary_values (boundary_values,
                                                            (this->system_matrix),
                                                            this->solution,
                                                            this->system_rhs, false);
                        
                }
                
                auto pointer=static_cast<SparseMatrix<double> *> (&(this->system_matrix));
                
                SparseDirectUMFPACK solver;
                solver.initialize(this->sparsity_pattern);
                solver.factorize(*pointer);
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
        
        {
                ofstream print;
                print.open("solution.m");
                
                if (print.is_open()) {
                        print<<"sol=[ ";
                        for (unsigned i=0; i<this->solution.size()-1; ++i) {
                                print<<this->solution(i)<<"; ";
                        }
                        print<<this->solution(this->solution.size()-1)<<" ];\n";
                }
                
                print.close();
        }
        
	this->ran=true;
        
}

#endif