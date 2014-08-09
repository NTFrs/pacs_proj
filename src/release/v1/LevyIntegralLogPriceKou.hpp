#ifndef __levy_integral_log_price_kou__
# define __levy_integral_log_price_kou__

# include "LevyIntegralLogPrice.hpp"
# include "Quadrature.hpp"
# include <cmath>

template<unsigned dim>
class LevyIntegralLogPriceKou: public LevyIntegralLogPrice<dim> {
	
protected:
	std::vector<Quadrature_Laguerre> leftQuads;
	std::vector<Quadrature_Laguerre> rightQuads;
	bool adapting;
	
	//! Creates quadrature nodes and weitghts of order n
	virtual void setup_quadratures(unsigned n);
	//! Reimplementation of LevyIntegralBase::compute_alpha() using Laguerre nodes
    virtual void compute_alpha();
        
	
public:
	LevyIntegralLogPriceKou()=delete;
        
        LevyIntegralLogPriceKou(const LevyIntegralLogPriceKou &)=delete;
        
		//! Only constructor of this class
		/*!
		 * Similar to constructor of base class,  adds the space for a boundary condition.
		 * \param lower_limit_ 		the left-bottom limit of the domain		
		 * \param upper_limit_ 		the rigth-upper limit of the domain
		 * \param Models_			A vector containing the needed models
		 * \param BC_ 				Pointer to the Boundary Condition. Best to use std::move(BC),  where BC is std::unique_ptr to a dinamically allocated Function\<dim\> object from Deal.II (possibly a BoundaryConditionLogPrice)
		 * \param apt				Used to set if the quadrature uses adaptive nodes (Default true) 
		 */
        LevyIntegralLogPriceKou(dealii::Point<dim> lower_limit_,
                                dealii::Point<dim> upper_limit_,
                                std::vector<Model *> & Models_,
                                std::unique_ptr<dealii::Function<dim> > BC_,
                                bool apt=true)
        :
        LevyIntegralLogPrice<dim>::LevyIntegralLogPrice(lower_limit_,
                                                        upper_limit_,
                                                        Models_,
                                                        std::move(BC_)),
        adapting(apt) {
                if (!adapting)
                        this->setup_quadratures(32);
                else
                        this->setup_quadratures(2);
	}
        
        LevyIntegralLogPriceKou& operator=(const LevyIntegralLogPriceKou &)=delete;
	
	//!Reimplementation of LevyIntegralLogPrice::compute_J using Laguerre nodes
	virtual void compute_J(dealii::Vector< double >& sol, dealii::DoFHandler<dim>& dof_handler, dealii::FE_Q<dim>& fe);
};

template<unsigned dim>
void LevyIntegralLogPriceKou<dim>::setup_quadratures(unsigned int n)
{
	leftQuads.clear();
        rightQuads.clear();
        
	for (unsigned d=0;d<dim;++d) {
                leftQuads.emplace_back(Quadrature_Laguerre(n, (this->mods[d])->get_lambda_m()));
                rightQuads.emplace_back(Quadrature_Laguerre(n, (this->mods[d])->get_lambda_p()));
        }
        
}

template<unsigned dim>
void LevyIntegralLogPriceKou<dim>::compute_alpha() {
	this->alpha=std::vector<double>(dim, 0.);
        
	if (!adapting) {
				//for each dimension it computes alpha
                for (unsigned d=0;d<dim;++d) {
						//since the exponential part is included in the weights,  we use the remaining part of the density exlicitly,  here for the positive part of the axis
                        for (unsigned i=0; i<rightQuads[d].get_order(); ++i) {
                                this->alpha[d]+=(exp((rightQuads[d].get_nodes())[i])-1)*
                                ((this->mods[d])->get_p())*((this->mods[d])->get_lambda())*
                                ((this->mods[d])->get_lambda_p())*(rightQuads[d].get_weights())[i];
                        }
						//and here for the negative part of the density's support
                        for (unsigned i=0; i<leftQuads[d].get_order(); ++i) {
                                this->alpha[d]+=(exp(-(leftQuads[d].get_nodes())[i])-1)*
                                (1-((this->mods[d])->get_p()))*((this->mods[d])->get_lambda())*
                                ((this->mods[d])->get_lambda_m())*(leftQuads[d].get_weights())[i];
                        }
                        
                }
        }
        
	else {
                unsigned order_max=64;
                
                std::vector<double> alpha_old;
                double err;
				//same but adaptive
                do  {
                        alpha_old=this->alpha;
                        this->alpha=std::vector<double>(dim, 0.);
                        
                        for (unsigned d=0;d<dim;++d) {
                                for (unsigned i=0; i<rightQuads[d].get_order(); ++i) {
                                        this->alpha[d]+=(exp((rightQuads[d].get_nodes())[i])-1)*
                                        ((this->mods[d])->get_p())*((this->mods[d])->get_lambda())*
                                        ((this->mods[d])->get_lambda_p())*(rightQuads[d].get_weights())[i];
                                }
                                
                                for (unsigned i=0; i<leftQuads[d].get_order(); ++i) {
                                        this->alpha[d]+=(exp(-(leftQuads[d].get_nodes())[i])-1)*
                                        (1-((this->mods[d])->get_p()))*((this->mods[d])->get_lambda())*
                                        ((this->mods[d])->get_lambda_m())*(leftQuads[d].get_weights())[i];
                                }
                        }
                        
                        setup_quadratures(2*leftQuads[0].get_order());
                        
                        
                        err=0.;
                        for (unsigned d=0;d<dim;++d)
                                err+=fabs(alpha_old[d]-(this->alpha[d]));
                        
                }
                while (err>constants::light_toll &&
                       rightQuads[0].get_order()<=order_max);
        }
        
}

template<unsigned dim>
void LevyIntegralLogPriceKou<dim>::compute_J(dealii::Vector< double >& sol, dealii::DoFHandler<dim>& dof_handler, dealii::FE_Q<dim>& fe)
{
    using namespace dealii;
	unsigned N(sol.size());

	//prepare the vector that will hold J1 and 2 by putting it to 0
	Vector<double> J; J.reinit(2*N);
    
    //prepare a map between dofs and point of the grid
	std::map<types::global_dof_index, Point<dim> > vertices;	DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dof_handler, vertices);
    
	//then for each dimension we repeat the following
	for (unsigned d=0;d<dim;++d) {
				//the next class is used to return the value of the solution on the specified point,  if the point is inside the domain. Otherwise returns the boundary condition.
                tools::Solution_Trimmer<dim> func(d,*(this->boundary), dof_handler, sol, this->lower_limit, this->upper_limit);
				
				//we already have the prepared quadrature nodes and weights

				//thus,  for each node on the mesh
				# pragma omp parallel for
                for (unsigned int it=0;it<N;++it)
                {
					
						//we prepare a vector that will contain d-dimensional quadrature points and one for the sol values
                        std::vector< Point<dim> > quad_points(leftQuads[d].get_order()+rightQuads[d].get_order());
                        std::vector<double> f_u(leftQuads[d].get_order()+rightQuads[d].get_order());
                        
						//and we fill it with quadrature nodes + the actual vertex
                        for (unsigned i=0; i<leftQuads[d].get_order(); ++i) {
                                quad_points[i][d]=-this->leftQuads[d].get_nodes()[i] + vertices[it][d];
                                if (dim==2) {
                                        quad_points[i][1-d]=vertices[it][1-d];
                                }
                        }
                        //but we must do it separately since the parameters on both sides are different
                        for (unsigned i=0; i<rightQuads[d].get_order(); ++i) {
                                quad_points[i+leftQuads[d].get_order()][d]=
                                this->rightQuads[d].get_nodes()[i] + vertices[it][d];
                                if (dim==2) {
                                        quad_points[i+leftQuads[d].get_order()][1-d]=vertices[it][1-d];
                                }
                        }
                        
						// we evaluate the solutions in those points
                        func.value_list(quad_points, f_u);
                        
						// And now we have everiting to integrate first on the negative side
                        for (unsigned i=0;i<leftQuads[d].get_order();++i) {
                                J[d*N+it]+=f_u[i]*(1-((this->mods[d])->get_p()))*((this->mods[d])->get_lambda())*
                                ((this->mods[d])->get_lambda_m())*(leftQuads[d].get_weights())[i];
                        }
                        // and then on the positive side
                        for (unsigned i=0;i<rightQuads[d].get_order();++i) {
                                J[d*N+it]+=f_u[i+leftQuads[d].get_order()]*((this->mods[d])->get_p())*((this->mods[d])->get_lambda())*
                                ((this->mods[d])->get_lambda_p())*(rightQuads[d].get_weights())[i];
                        }
                        
                }
                
        }
        
	//we then transfer the computed values on j1 and j2
	this->j1.reinit(N);
	for (unsigned i=0;i<this->j1.size();++i)
                this->j1[i]=J[i];
	if (dim==2) {
                this->j2.reinit(N);
                for (unsigned i=0;i<this->j2.size();++i)
                        this->j2[i]=J[i+N];
        }
                
        this->j_ran=true;
}

#endif