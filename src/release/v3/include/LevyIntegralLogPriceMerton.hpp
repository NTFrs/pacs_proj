#ifndef __levy_integral_log_price_merton_hpp
#define __levy_integral_log_price_merton_hpp

#include "LevyIntegralLogPrice.hpp"

template<unsigned dim>
class LevyIntegralLogPriceMerton: public LevyIntegralLogPrice<dim> {
        
protected:
	std::vector<Quadrature_Hermite> quadratures;
	
        //! Creates quadrature nodes and weitghts of order n
	virtual void setup_quadratures(unsigned n);
	//! Reimplementation of LevyIntegralBase::compute_alpha() using Hermite nodes
	virtual void compute_alpha();
	
	virtual double get_one_J(dealii::Point<dim> vert, tools::Solution_Trimmer<dim> & trim,  unsigned d);
	
public:
	LevyIntegralLogPriceMerton()=delete;
        
        LevyIntegralLogPriceMerton(const LevyIntegralLogPriceMerton &)=delete;
        
        //! Only constructor of this class
        /*!
         * Similar to constructor of base class,  adds the space for a boundary condition.
         * \param lower_limit_ 		the left-bottom limit of the domain		
         * \param upper_limit_ 		the rigth-upper limit of the domain
         * \param Models_			A vector containing the needed models
         * \param BC_ 				Pointer to the Boundary Condition. Best to use std::move(BC),  where BC is std::unique_ptr to a dinamically allocated Function\<dim\> object from Deal.II (possibly a BoundaryConditionLogPrice)
         * \param apt				Used to set if the quadrature uses adaptive nodes (Default true)
         */
        LevyIntegralLogPriceMerton(dealii::Point<dim> lower_limit_,
                                   dealii::Point<dim> upper_limit_,
                                   std::vector<Model *> & Models_,
                                   std::unique_ptr<dealii::Function<dim> > BC_,
                                   bool apt=true)
        :
        LevyIntegralLogPrice<dim>::LevyIntegralLogPrice(lower_limit_,
                                                        upper_limit_,
                                                        Models_,
                                                        std::move(BC_),
                                                        apt) 
        {
                if (!this->adapting)
                        this->setup_quadratures(16);
                else
                        this->setup_quadratures(2);
        }
        
        LevyIntegralLogPriceMerton& operator=(const LevyIntegralLogPriceMerton &)=delete;
        
};

template<unsigned dim>
void LevyIntegralLogPriceMerton<dim>::setup_quadratures(unsigned int n)
{
	quadratures.clear();
	using namespace std;
	for (unsigned d=0;d<dim;++d) {
                quadratures.emplace_back(Quadrature_Hermite(n, (this->mods[d])->get_nu(), (this->mods[d])->get_delta() ));
        }
}

template<unsigned dim>
void LevyIntegralLogPriceMerton<dim>::compute_alpha()
{
	this->alpha=std::vector<double>(dim, 0.);
        
        
	if (!this->adapting) {
		//for each dimension it computes alpha
                for (unsigned d=0;d<dim;++d) {
			//since the gaussian part is included in the weights,  we use the remaining part of the density exlicitly
                        for (unsigned i=0; i<quadratures[d].get_order(); ++i) {
                                this->alpha[d]+=(exp((quadratures[d].get_nodes())[i])-1)*
                                ((this->mods[d])->get_lambda())/(((this->mods[d])->get_delta())*sqrt(2*constants::pi))
                                *(quadratures[d].get_weights())[i];
                        }
                }
        }
        
	else {
                unsigned order_max=64;
                
                std::vector<double> alpha_old;
                double err;
		// same as above but adaptive
                do  {
                        alpha_old=this->alpha;
                        this->alpha=std::vector<double>(dim, 0.);
                        
                        for (unsigned d=0;d<dim;++d) {
                                for (unsigned i=0; i<quadratures[d].get_order(); ++i) {
                                        this->alpha[d]+=(exp((quadratures[d].get_nodes())[i])-1)*
                                        ((this->mods[d])->get_lambda())/(((this->mods[d])->get_delta())*sqrt(2*constants::pi))
                                        *(quadratures[d].get_weights())[i];
                                }
                                
                        }
                        
                        setup_quadratures(2*quadratures[0].get_order());
                        
                        
                        err=0.;
                        for (unsigned d=0;d<dim;++d)
                                err+=fabs(alpha_old[d]-(this->alpha[d]));
                        
                }
                while (err>this->alpha_toll &&
                       quadratures[0].get_order()<order_max);
        }
        this->order=quadratures[0].get_order();
        
}


template<unsigned dim>
double LevyIntegralLogPriceMerton<dim>::get_one_J(dealii::Point< dim > vert, tools::Solution_Trimmer< dim >& trim, unsigned int d)
{
	using namespace dealii;
	double j(0);
	//we prepare a vector that will contain d-dimensional quadrature points and one for the sol values
	std::vector< Point<dim> > quad_points(quadratures[d].get_order());
	std::vector<double> f_u(quadratures[d].get_order());
        
	//and we fill it with quadrature nodes + the actual vertex
	for (unsigned i=0; i<quad_points.size(); ++i) {
                quad_points[i][d]=this->quadratures[d].get_nodes()[i] + vert[d];
                if (dim==2) {
                        quad_points[i][1-d]=vert[1-d];
                }
        }
        
        
	// we evaluate the solutions in those points
	trim.value_list(quad_points, f_u);
        
	// And now we have everiting to integrate
	for (unsigned i=0;i<quadratures[d].get_order();++i) {
                j+=f_u[i]*((this->mods[d])->get_lambda())/(((this->mods[d])->get_delta())*sqrt(2*constants::pi))
                *(quadratures[d].get_weights())[i];
        }
        return j;
}


#endif