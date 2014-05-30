#ifndef __final_conditions_hpp
#define __final_conditions_hpp

#include "deal_ii.hpp"
#include "OptionTypes.hpp"

template<unsigned dim>
class FinalCondition : public Function<dim>
{
public:
	FinalCondition (double K_, OptionType type_)
        :
        Function<dim>(),
        K(K_),
        type(type_)
        {};
        
	virtual double value (const Point<dim>   &p,
                              // scalar function, return the first component, label 0
                              const unsigned int  component = 0) const;
private:
	double K;
        OptionType type;
};

template<unsigned dim>
double FinalCondition<dim>::value (const Point<dim>  &p,
                           const unsigned int component) const
{
	Assert (component == 0, ExcInternalError());
        
        double point(0.);
        
        for (unsigned i=0; i<dim; ++i) {
                point+=p(i);
        
        }
        
        if (type==OptionType::Put)
                return (K-point>0.)?(K-point):0.;
        else
                return (point-K>0.)?(point-K):0.;
}

#endif