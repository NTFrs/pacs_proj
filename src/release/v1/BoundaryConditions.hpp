#ifndef __boundary_conditions_hpp
#define __boundary_conditions_hpp

#include "deal_ii.hpp"
#include "OptionTypes.hpp"

//! Class of Boundary Condititions
/*!
 * This class is used to set the boundary conditions of our problems. It works in 1d and 2d and needs to know if the option is a Put or a Call.
 */
template<unsigned dim>
class BoundaryCondition: public dealii::Function<dim>
{
public:
        //! Constructor
        /*!
         * \param K_    Strike Price
         * \param T_    Time to Maturity
         * \param r_    Interest Rate
         * \param type_ Option type (Put or Call)
         */
	BoundaryCondition(double K_, double T_,  double r_, OptionType type_)
        :
        dealii::Function<dim>(),
        K(K_),
        T(T_),
        r(r_),
        type(type_)
        {};
        
        //! Function needed by dealii
	virtual double value (const dealii::Point<dim> &p, const unsigned int component =0) const;
        
private:
	double K;
	double T;
	double r;
        OptionType type;
};

template<unsigned dim>
double BoundaryCondition<dim>::value(const dealii::Point<dim> &p, const unsigned int component) const
{
	Assert (component == 0, dealii::ExcInternalError());
        
        double point(0.);
        
        for (unsigned i=0; i<dim; ++i) {
                point+=p(i);
                
        }

        if (type==OptionType::Put)
                return  (K*exp(-r*(T-this->get_time()))-point>0.)?
                        (K*exp(-r*(T-this->get_time()))-point):0.;
        else
                return  (point-K*exp(-r*(T-this->get_time()))>0.)?
                        (point-K*exp(-r*(T-this->get_time()))):0.;
        
}

#endif