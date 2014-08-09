#ifndef __boundary_conditions_logprice_hpp
#define __boundary_conditions_logprice_hpp

#include "deal_ii.hpp"
#include "OptionTypes.hpp"

//! Class of Boundary Condititions
/*!
 * This class is used to set the boundary conditions of our problems. It works in 1d and 2d and needs to know if the option is a Put or a Call. Works with the equation in LogPrice form. The function has an internal time,  that an be set through the method set_time(double), inherited from base class dealii::Function<dim>.
 */
template<unsigned dim>
class BoundaryConditionLogPrice: public dealii::Function<dim>
{
public:
        //! Constructor
        /*!
         * \param S0_   Spot Price
         * \param K_    Strike Price
         * \param T_    Time to Maturity
         * \param r_    Interest Rate
         * \param type_ Option type (Put or Call)
         */
	BoundaryConditionLogPrice(std::vector<double> & S0_, double K_, double T_,  double r_, OptionType type_)
        :
        dealii::Function<dim>(),
        S0(S0_),
        K(K_),
        T(T_),
        r(r_),
        type(type_)
        {};
        
	//! Returns the value of the function at the point p
	virtual double value (const dealii::Point<dim> &p, const unsigned int component =0) const;
        
private:
        std::vector<double> S0;
	double K;
	double T;
	double r;
        OptionType type;
};

template<unsigned dim>
double BoundaryConditionLogPrice<dim>::value(const dealii::Point<dim> &p, const unsigned int component) const
{
	Assert (component == 0, dealii::ExcInternalError());
        
        double point(0.);
        
        for (unsigned i=0; i<dim; ++i) {
                point+=S0[i]*exp(p(i));
                
        }
        
        if (type==OptionType::Put)
                return  (K*exp(-r*(T-this->get_time()))-point>0.)?
                (K*exp(-r*(T-this->get_time()))-point):0.;
        else
                return  (point-K*exp(-r*(T-this->get_time()))>0.)?
                (point-K*exp(-r*(T-this->get_time()))):0.;
        
}

#endif