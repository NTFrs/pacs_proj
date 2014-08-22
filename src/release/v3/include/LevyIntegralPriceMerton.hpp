#ifndef __levy_integral_price_merton_hpp
#define __levy_integral_price_merton_hpp

#include "LevyIntegralPrice.hpp"

//! A class to treat the integral part when in Price form optimized for Merton model
/*!
 * This class derived from LevyIntegralPrice that reimplemets the computing of alpha in order to use more efficient nodes. In particular, it uses Gauss-Hermite nodes to compute the alpha part of the integral,  since they suit better the gaussian kernel of the integral.
 */

template<unsigned dim>
class LevyIntegralPriceMerton: public LevyIntegralPrice<dim> {
        
protected:
        
	std::vector<Quadrature_Hermite> quadratures;
	bool adapting;
        
	//! Creates quadrature nodes and weitghts of order n
	virtual void setup_quadratures(unsigned n);
	//! Reimplementation of LevyIntegralBase::compute_alpha() using Hermite nodes
	virtual void compute_alpha();
        
public:
	LevyIntegralPriceMerton()=delete;
        
        LevyIntegralPriceMerton(const LevyIntegralPriceMerton &)=delete;
        
	//! Only constructor of the class
	/*!
	 * Similar to constructor of base class,  adds the option to make the quadrature of alpha adaptive.
	 * \param lower_limit_ 		the left-bottom limit of the domain		
	 * \param upper_limit_ 		the rigth-upper limit of the domain
	 * \param Models_			A vector containing the needed models
	 * \param apt 				boolean indicating if the quadrature must be adaptive. Default true.
	 */
	LevyIntegralPriceMerton(dealii::Point<dim> lower_limit_,
                                dealii::Point<dim> upper_limit_,
                                std::vector<Model *> & Models_,
                                unsigned order_,
                                bool apt=true)
        :
        LevyIntegralPrice<dim>::LevyIntegralPrice(lower_limit_,
                                                  upper_limit_,
                                                  Models_,
                                                  order_),
        adapting(apt) {
                if (!adapting)
                        this->setup_quadratures(order_);
                else
                        this->setup_quadratures(2);   
        }
        
        LevyIntegralPriceMerton& operator=(const LevyIntegralPriceMerton &)=delete;
        
};

template<unsigned dim>
void LevyIntegralPriceMerton<dim>::setup_quadratures(unsigned int n)
{
	quadratures.clear();
	using namespace std;
	for (unsigned d=0;d<dim;++d) {
                quadratures.emplace_back(Quadrature_Hermite(n, (this->mods[d])->get_nu(), (this->mods[d])->get_delta() ));
        }
}


template<unsigned dim>
void LevyIntegralPriceMerton<dim>::compute_alpha(){
        
	this->alpha=std::vector<double>(dim, 0.);
        
	if (!adapting) {
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
                // same as above but adaptive
                std::vector<double> alpha_old;
                double err;
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
                       quadratures[0].get_order()<this->order_max);
        }
        
        this->order=quadratures[0].get_order();
        
}


#endif