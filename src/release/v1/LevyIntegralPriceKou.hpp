#ifndef __levy_integral_price_kou__
# define __levy_integral_price_kou__

#include "LevyIntegralPrice.hpp"
#include "Quadrature.hpp"
#include <cmath>

//! A class to treat the integral part when in Price form optimized for Kou model
/*!
 * This class derived from LevyIntegralPrice that reimplemets the computing of alpha in order to use more efficient nodes. In particular, it uses Gauss-Laguerre nodes to compute the alpha part of the integral,  since they suit better the exponential kernel of the integral.
 */

template<unsigned dim>
class LevyIntegralPriceKou: public LevyIntegralPrice<dim> {
        
protected:
        
	std::vector<Quadrature_Laguerre> leftQuads;
	std::vector<Quadrature_Laguerre> rightQuads;
	bool adapting;
	
	virtual void setup_quadratures(unsigned n);
	//! Reimplementation of compute_alpha using Laguerre nodes
	virtual void compute_alpha();
	
public:
	LevyIntegralPriceKou()=delete;
        
    LevyIntegralPriceKou(const LevyIntegralPriceKou &)=delete;
	
	//! Only constructor of the class
	/*!
     * Similar to constructor of base class,  adds the option to make the quadrature of alpha adaptive.
	 * \param lower_limit_ 		the left-bottom limit of the domain		
	 * \param upper_limit_ 		the rigth-upper limit of the domain
	 * \param Models_			A vector containing the needed models
	 * \param apt 				boolean indicating if the quadrature must be adaptive. Default true.
     */
	LevyIntegralPriceKou(dealii::Point<dim> lower_limit_,
                             dealii::Point<dim> upper_limit_,
                             std::vector<Model *> & Models_,
                             bool apt=true)
        :
        LevyIntegralPrice<dim>::LevyIntegralPrice(lower_limit_,
                                                  upper_limit_,
                                                  Models_),
        adapting(apt) {
                if (!adapting)
			this->setup_quadratures(16);
                else
			this->setup_quadratures(2);   
	}
        
        LevyIntegralPriceKou& operator=(const LevyIntegralPriceKou &)=delete;
        
};


template<unsigned dim>
void LevyIntegralPriceKou<dim>::setup_quadratures(unsigned int n)
{
	leftQuads.clear();rightQuads.clear();
        
	for (unsigned d=0;d<dim;++d) {
                leftQuads.emplace_back(Quadrature_Laguerre(n, (this->mods[d])->get_lambda_m()));
                rightQuads.emplace_back(Quadrature_Laguerre(n, (this->mods[d])->get_lambda_p()));
	}
}


template<unsigned dim>
void LevyIntegralPriceKou<dim>::compute_alpha(){
        
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
				//same but adaptive
                unsigned order_max=64;
                
                std::vector<double> alpha_old;
                double err;
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
        //TODO verbosity
//         for (unsigned i=0; i<dim; ++i) {
//                 std::cout<<this->alpha[i]<<"\t";
//         }
//         std::cout<<"\n";
        
}


# endif