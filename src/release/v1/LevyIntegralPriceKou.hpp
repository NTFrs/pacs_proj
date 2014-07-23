#ifndef __levy_integral_price_kou__
# define __levy_integral_price_kou__

#include "LevyIntegralPrice.hpp"
#include "Quadrature.hpp"

template<unsigned dim>
class LevyIntegralPriceKou: public LevyIntegralPrice<dim> {

protected:

	std::vector<Quadrature_Laguerre> leftQuads;
	std::vector<Quadrature_Laguerre> rightQuads;
	bool adapting;
	
	virtual void setup_quadratures(unsigned n); 
	
public:
	LevyIntegralPriceKou()=delete;
	LevyIntegralPriceKou(std::vector<Model *> & Models_,  bool apt=true): LevyIntegralPrice<dim>::LevyIntegralPrice(dealii::Point<dim>(), dealii::Point<dim>(), Models_), adapting(apt) {
		  if (!adapting)
			this->setup_quadratures(16);
		  else
			this->setup_quadratures(2);   
	}

};

template<unsigned dim>
void LevyIntegralPriceKou<dim>::setup_quadratures(unsigned int n)
{
	leftQuads.clear();rightQuads.clear();

	for (unsigned d=0;d<dim;++d) {
	  leftQuads.emplace_back(Quadrature_Laguerre(n, (this->Mods[d])->get_lambda_m()));
	  rightQuads.emplace_back(Quadrature_Laguerre(n, (this->Mods[d])->get_lambda_p()));
	}
}

/*  
template<unsigned dim>
void LevyIntegralPriceKou<dim>::compute_alpha(){

	this->alpha=std::vector<double>(dim, 0.);
	
	for (unsigned d=0;d<dim;++d) {
	
	if (!adapting) {
	 
	 for (unsigned i=0; i<rightQuads[d].get_order(); ++i) {
	  alpha[d]+=(exp((rightQuads[d].get_nodes())[i])-1)*
	  ((this->Mods[d])->get_p())*((this->Mods[d])->get_lambda())*
	  ((this->Mods[d])->get_lambda_p())*(rightQuads[d].get_nodes())[i];
	}

	 for (unsigned i=0; i<left_quad.get_order(); ++i) {
	  alpha+=(exp(-left_quad_nodes[i])-1)*
	  (1-(models[0]->get_p()))*(models[0]->get_lambda())*
	  (models[0]->get_lambda_m())*left_quad_weights[i];
	}

   }

	else {
	 unsigned order_max=64;

	 double alpha_old=0.;

	 do  {
	  alpha_old=alpha;
	  alpha=0.;

	  for (unsigned i=0; i<right_quad.get_order(); ++i) {
	   alpha+=(exp(right_quad_nodes[i])-1)*
	   (models[0]->get_p())*(models[0]->get_lambda())*
	   (models[0]->get_lambda_p())*right_quad_weights[i];
	 }

	  for (unsigned i=0; i<left_quad.get_order(); ++i) {
	   alpha+=(exp(-left_quad_nodes[i])-1)*
	   (1-(models[0]->get_p()))*(models[0]->get_lambda())*
	   (models[0]->get_lambda_m())*left_quad_weights[i];
	 }

	  setup_quadrature_rigth(2*right_quad.get_order());
	  setup_quadrature_left(2*left_quad.get_order());
	  setup_quadrature_points();

	}
	 while (abs(alpha_old-alpha)>toll &&
	  right_quad.get_order()<=order_max &&
	  left_quad.get_order()<=order_max);
   }


}
}*/

# endif