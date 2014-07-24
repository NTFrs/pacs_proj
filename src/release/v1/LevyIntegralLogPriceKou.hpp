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
	
	virtual void setup_quadratures(unsigned n);
	
public:
	LevyIntegralLogPriceKou()=delete;
	LevyIntegralLogPriceKou(dealii::Point<dim> lower_limit_,  dealii::Point<dim> upper_limit_,  std::vector<Model *> & Models_,  dealii::Function<dim> & BC_,  bool apt=true): LevyIntegralLogPrice<dim>::LevyIntegralLogPrice(lower_limit_, upper_limit_, Models_, BC), adapting(apt) {
	 if (!adapting)
	 this->setup_quadratures(16);
	 else
	 this->setup_quadratures(2);
	}
};

template<unsigned dim>
void LevyIntegralLogPriceKou::setup_quadratures(unsigned int n)
{
	leftQuads.clear();rightQuads.clear();

	for (unsigned d=0;d<dim;++d) {
	 leftQuads.emplace_back(Quadrature_Laguerre(n, (this->Mods[d])->get_lambda_m()));
	 rightQuads.emplace_back(Quadrature_Laguerre(n, (this->Mods[d])->get_lambda_p()));
   }

}


#endif