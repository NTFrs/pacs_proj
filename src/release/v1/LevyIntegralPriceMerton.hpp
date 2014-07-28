#ifndef __levy_integral_price_merton__
# define __levy_integral_price_merton__

# include "LevyIntegralPrice.hpp"
# include "Quadrature.hpp"
# include <cmath>

template<unsigned dim>
class LevyIntegralPriceMerton: public LevyIntegralPrice<dim> {
        
protected:
        
	std::vector<Quadrature_Hermite> quadratures;
	bool adapting;
        
	virtual void setup_quadratures(unsigned n);
	virtual void compute_alpha();
        
public:
	LevyIntegralPriceMerton()=delete;
	LevyIntegralPriceMerton(dealii::Point<dim> lower_limit_,  dealii::Point<dim> upper_limit_,std::vector<Model *> & Models_,  bool apt=true): LevyIntegralPrice<dim>::LevyIntegralPrice(lower_limit_, upper_limit_, Models_), adapting(apt) {
                if (!adapting)
                        this->setup_quadratures(16);
                else
                        this->setup_quadratures(2);   
        }
        
};

template<unsigned dim>
void LevyIntegralPriceMerton<dim>::setup_quadratures(unsigned int n)
{
	quadratures.clear();
	using namespace std;
	for (unsigned d=0;d<dim;++d) {
                quadratures.emplace_back(Quadrature_Hermite(n, (this->Mods[d])->get_nu(), (this->Mods[d])->get_delta() ));
        }
}


template<unsigned dim>
void LevyIntegralPriceMerton<dim>::compute_alpha(){
        
	this->alpha=std::vector<double>(dim, 0.);
        
	std::cerr<<"Im calculating the merton one\n";
        
	if (!adapting) {
                for (unsigned d=0;d<dim;++d) {
                        for (unsigned i=0; i<quadratures[d].get_order(); ++i) {
                                this->alpha[d]+=(exp((quadratures[d].get_nodes())[i])-1)*
                                ((this->Mods[d])->get_lambda())/(((this->Mods[d])->get_delta())*sqrt(2*constants::pi))
                                *(quadratures[d].get_weights())[i];
                        }
                }
        }
        
	else {
                unsigned order_max=64;
                
                std::vector<double> alpha_old;
                double err;
                do  {
                        alpha_old=this->alpha;
                        this->alpha=std::vector<double>(dim, 0.);
                        
                        for (unsigned d=0;d<dim;++d) {
                                for (unsigned i=0; i<quadratures[d].get_order(); ++i) {
                                        this->alpha[d]+=(exp((quadratures[d].get_nodes())[i])-1)*
                                        ((this->Mods[d])->get_lambda())/(((this->Mods[d])->get_delta())*sqrt(2*constants::pi))
                                        *(quadratures[d].get_weights())[i];
                                }
                                
                        }
                        
                        setup_quadratures(2*quadratures[0].get_order());
                        
                        
                        err=0.;
                        for (unsigned d=0;d<dim;++d)
                                err+=fabs(alpha_old[d]-(this->alpha[d]));
                        
                }
                while (err>constants::light_toll &&
                       quadratures[0].get_order()<=order_max);
        }
        
        
}


#endif