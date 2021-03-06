#ifndef __american_option_logprice_hpp
#define __american_option_logprice_hpp

#include "OptionBase.hpp"

//! A class for evaluation of American Options
/*!
 * This class evaluates the price of American Option, of type Put and Basket Put
 */
template <unsigned dim>
class AmericanOptionLogPrice final: public OptionBaseLogPrice<dim>
{
private:
        ExerciseType us;
        
        virtual void setup_integral();
        virtual void solve();
public:
        AmericanOptionLogPrice()=delete;
        
        AmericanOptionLogPrice(const AmericanOptionLogPrice &)=delete;
        
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
        AmericanOptionLogPrice(Model * const model,
                               double r_,
                               double T_,
                               double K_,
                               unsigned refs_,
                               unsigned time_step_)
        :
        OptionBaseLogPrice<dim>::OptionBaseLogPrice(ExerciseType::US, model, r_, T_, K_, refs_, time_step_),
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
        AmericanOptionLogPrice(Model * const model1,
                               Model * const model2,
                               double rho_,
                               double r_,
                               double T_,
                               double K_,
                               unsigned refs_,
                               unsigned time_step_)
        :
        OptionBaseLogPrice<dim>::OptionBaseLogPrice(ExerciseType::US, model1, model2, rho_, r_, T_, K_, refs_, time_step_),
        us(ExerciseType::US)
        {};
        
        AmericanOptionLogPrice& operator=(const AmericanOptionLogPrice &)=delete;

};

template <unsigned dim>
void AmericanOptionLogPrice<dim>::setup_integral(){
        
        std::vector<double> S0(dim);
        for (unsigned d=0; d<dim; ++d) {
                S0[d]=this->models[d]->get_spot();
        }
        
        std::unique_ptr<dealii::Function<dim> > bc
        (new BoundaryConditionLogPrice<dim> (S0, this->K, this->T,  this->r, OptionType::Put));
        
        if (this->model_type==OptionBase<dim>::ModelType::Kou) {
                this->levy=std::unique_ptr<LevyIntegralBase<dim> >
                (new LevyIntegralLogPriceKou<dim>(this->Smin, this->Smax, this->models, std::move(bc)));
        }
        else if (this->model_type==OptionBase<dim>::ModelType::Merton) {
                this->levy=std::unique_ptr<LevyIntegralBase<dim> >
                (new LevyIntegralLogPriceMerton<dim>(this->Smin, this->Smax,this->models, std::move(bc)));
        }
}


template <unsigned dim>
void AmericanOptionLogPrice<dim>::solve ()
{
        using namespace dealii;
        using namespace std;
        
        std::vector<double> S0(dim);
        for (unsigned d=0; d<dim; ++d) {
                S0[d]=this->models[d]->get_spot();
        }
        
        VectorTools::interpolate (this->dof_handler,
                                  FinalConditionLogPrice<dim>(S0, this->K, OptionType::Put),
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
        
        BoundaryConditionLogPrice<dim> bc(S0, this->K, this->T,  this->r, OptionType::Put);
        
	cout<< "time step is"<< this->time_step << endl;
	
	for (double time=this->T-this->dt;time >=0;time-=this->dt, --Step) {
                
                cout<< "Step "<< Step<<"\t at time \t"<< time<< endl;
                
                if (this->refine && Step%20==0)
                        this->refine_grid();
                
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
                bc.set_time(time);
                
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
                                                              this->K,
                                                              S0);
                        
                        auto temp=this->solution;
                        temp.add(-1, solution_old);
                        
                        if (temp.linfty_norm()<tollerance){
                                converged=true;
                        }
                        
                        else
                                solution_old=this->solution;
                        
                        
                        if (k==maxiter-1) {
                                cout<<"Warning: maxiter reached, with error="<<temp.linfty_norm()<<"\n";
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