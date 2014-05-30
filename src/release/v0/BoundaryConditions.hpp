#ifndef __boundary_conditions_hpp
#define __boundary_conditions_hpp

#include "deal_ii.hpp"
#include "OptionTypes.hpp"

template<unsigned dim>
class BoundaryCondition: public Function<dim>
{
public:
	BoundaryCondition(double K_, double T_,  double r_, OptionType type_)
        :
        Function<dim>(),
        K(K_),
        T(T_),
        r(r_),
        type(type_)
        {};
        
	virtual double value (const Point<dim> &p, const unsigned int component =0) const;
        
private:
	double K;
	double T;
	double r;
        OptionType type;
};

template<unsigned dim>
double BoundaryCondition<dim>::value(const Point<dim> &p, const unsigned int component) const
{
	Assert (component == 0, ExcInternalError());
        
        double point(0.);
        
        for (unsigned i=0; i<dim; ++i) {
                point+=p(i);
                
        }
        
        if (type==OptionType::Put)
                return  (K*exp(-r*(T-this->get_time()))-point>0.)?
                        (K*exp(-r*(T-this->get_time()))-point):0.;
        else
                return  (point-K*exp(-r*(T-this->get_time())))?
                        (point-K*exp(-r*(T-this->get_time()))):0.;
        
}

#endif