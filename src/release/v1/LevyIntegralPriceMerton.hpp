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
	LevyIntegralPriceMerton(std::vector<Model *> & Models_,  bool apt=true): LevyIntegralPrice<dim>::LevyIntegralPrice(dealii::Point<dim>(), dealii::Point<dim>(), Models_), adapting(apt) {
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

	for (unsigned d=0;d<dim;++d) {
	 quadratures.emplace_back(Quadrature_Laguerre(n, (this->Mods[d])->()));//TODO
   }
}


template<unsigned dim>
void LevyIntegralPriceMerton<dim>::compute_alpha(){

	this->alpha=std::vector<double>(dim, 0.);



	if (!adapting) {
	 for (unsigned d=0;d<dim;++d) {
	  for (unsigned i=0; i<rightQuads[d].get_order(); ++i) {
	   this->alpha[d]+=(exp((rightQuads[d].get_nodes())[i])-1)*
	   ((this->Mods[d])->get_p())*((this->Mods[d])->get_lambda())*
	   ((this->Mods[d])->get_lambda_p())*(rightQuads[d].get_weights())[i];
	 }

	  for (unsigned i=0; i<leftQuads[d].get_order(); ++i) {
	   this->alpha[d]+=(exp(-(leftQuads[d].get_nodes())[i])-1)*
	   (1-((this->Mods[d])->get_p()))*((this->Mods[d])->get_lambda())*
	   ((this->Mods[d])->get_lambda_m())*(leftQuads[d].get_weights())[i];
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
	   for (unsigned i=0; i<rightQuads[d].get_order(); ++i) {
		this->alpha[d]+=(exp((rightQuads[d].get_nodes())[i])-1)*
		((this->Mods[d])->get_p())*((this->Mods[d])->get_lambda())*
		((this->Mods[d])->get_lambda_p())*(rightQuads[d].get_weights())[i];
	  }

	   for (unsigned i=0; i<leftQuads[d].get_order(); ++i) {
		this->alpha[d]+=(exp(-(leftQuads[d].get_nodes())[i])-1)*
		(1-((this->Mods[d])->get_p()))*((this->Mods[d])->get_lambda())*
		((this->Mods[d])->get_lambda_m())*(leftQuads[d].get_weights())[i];
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


#endif