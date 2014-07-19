#ifndef __integral_levy_normal_hpp
# define __integral_levy_normal_hpp

# include "deal_ii.hpp"
# include "Quadrature.hpp"
# include "Densities.hpp"
# include "tools.hpp"

template<unsigned dim>
class LevyIntegral<dim> {

private:
	std::vector<double> alpha;
	dealii::Vector<double> J1, J2;
	
	std::vector<>
	
	bool alpha_ran;
	
	void compute_alpha();
	
	dealii::Vector<double> J1, J2;

public:
	double get_alpha_1() {if (!alpha_ran)
							compute_alpha();
							else return alpha[0];}
	
	double get_alpha_2() {if (!alpha_ran)                      // add exception if dim < 2
							 compute_alpha();
							 else return alpha[1];}
};

template<unsigned dim>
void LevyIntegral< dim >::compute_alpha()
{
for (unsigned d=0;d<dim;++d) {

}
}


#endif