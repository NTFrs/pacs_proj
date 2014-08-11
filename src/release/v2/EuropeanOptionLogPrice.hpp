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
        OptionType type2;
        ExerciseType eu;
        
        virtual void setup_integral();
        virtual void solve();
public:
        EuropeanOptionLogPrice()=delete;
        
        EuropeanOptionLogPrice(const EuropeanOptionLogPrice &)=delete;
        
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
        type2(type_),
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
        OptionBaseLogPrice<dim>::OptionBaseLogPrice(ExerciseType::EU, model1, model2, rho_, r_, T_, K_, refs_, time_step_),
        type2(type_),
        eu(ExerciseType::EU)
        {};
        
        EuropeanOptionLogPrice& operator=(const EuropeanOptionLogPrice &)=delete;

};

template <unsigned dim>
void EuropeanOptionLogPrice<dim>::setup_integral(){
        
        std::vector<double> S0(dim);
        for (unsigned d=0; d<dim; ++d) {
                S0[d]=this->models[d]->get_spot();
        }
        
        std::unique_ptr<dealii::Function<dim> > bc
        (new BoundaryConditionLogPrice<dim> (S0, this->K, this->T,  this->r, this->type2));
        //TODO what if different model? Expansions!
        if (this->model_type==OptionBase<dim>::ModelType::Kou) {
                this->levy=std::unique_ptr<LevyIntegralBase<dim> > (new LevyIntegralLogPriceKou<dim>(this->Smin, this->Smax, this->models, std::move(bc), false));
        }
        else if (this->model_type==OptionBase<dim>::ModelType::Merton) {
                this->levy=std::unique_ptr<LevyIntegralBase<dim> > (new LevyIntegralLogPriceMerton<dim>(this->Smin, this->Smax,this->models, std::move(bc)));
        }
}

template <unsigned dim>
void EuropeanOptionLogPrice<dim>::solve ()
{
        using namespace dealii;
        using namespace std;
        
        std::vector<double> S0(dim);
        for (unsigned d=0; d<dim; ++d) {
                S0[d]=this->models[d]->get_spot();
        }
        
        VectorTools::interpolate (this->dof_handler,
                                  FinalConditionLogPrice<dim>(S0, this->K, this->type2),
                                  this->solution);
        
        if (this->print) {
                this->print_solution_gnuplot("begin");
                this->print_solution_matlab("begin");
        }
        
	unsigned Step=this->time_step;
        
        BoundaryConditionLogPrice<dim> bc(S0, this->K, this->T,  this->r, this->type2);
        
	cout<< "time step is"<< this->time_step << endl;
        
        if (dim==2 && this->print_grids) {
                this->print_grid(Step);
        }
	
	for (double time=this->T-this->dt;time >=0;time-=this->dt, --Step) {
                
                cout<< "Step "<< Step<<"\t at time \t"<< time<< endl;
                
                if (this->refine && Step%20==0 && Step!=this->time_step) {
                        this->refine_grid();
                        if (dim==2 && this->print_grids) {
                                this->print_grid(Step);
                        }
                }
                //
                if (this->model_type!=OptionBase<dim>::ModelType::BlackScholes) {
                        
                        Vector<double> *J_x;
                        Vector<double> *J_y;
                        Vector<double> temp;
                        this->levy->set_time(time);
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
        
        if (this->print) {
                this->print_solution_gnuplot("end");
                this->print_solution_matlab("end");
        }
        
	this->ran=true;
        
}

/*
 template<unsigned int>
 void EuropeanOptionLogPrice::solve_one_step(double time)
 {
 using namespace dealii;
 BoundaryConditionLogPrice<dim> bc(S0, this->K, this->T,  this->r, this->type2);
 
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
 
 */
#endif