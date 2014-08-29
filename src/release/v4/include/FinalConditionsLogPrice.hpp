#ifndef __final_conditions_logprice_hpp
#define __final_conditions_logprice_hpp

#include "dealii.hpp"
#include "OptionTypes.hpp"

//! Class of Final Condititions
/*!
 * This class is used to set the final conditions of our problems. It works in 1d and 2d and needs to know if the option is a Put or a Call. Works with the equation in LogPrice form.
 */
template<unsigned dim>
class FinalConditionLogPrice : public dealii::Function<dim>
{
public:
        //! Constructor
        /*!
         * \param K_    Strike Price
         * \param type_ Option type (Put or Call)
         */
        FinalConditionLogPrice (std::vector<double> & S0_, double K_, OptionType type_)
                :
                dealii::Function<dim>(),
                S0(S0_),
                K(K_),
                type(type_)
        {};

        //! Returns the value of the function at the point p
        virtual double value (const dealii::Point<dim>  & p,
                              // scalar function, return the first component, label 0
                              const unsigned int  component = 0) const;
private:
        std::vector<double> S0;
        double K;
        OptionType type;
};

template<unsigned dim>
double FinalConditionLogPrice<dim>::value (const dealii::Point<dim> & p,
                const unsigned int component) const
{
        using namespace dealii;

        Assert (component == 0, ExcInternalError());

        double point(0.);

        for (unsigned i=0; i<dim; ++i)
        {
                point+=S0[i]*exp(p(i));

        }

        if (type==OptionType::Put)
        {
                return (K-point>0.)?(K-point):0.;
        }
        else
        {
                return (point-K>0.)?(point-K):0.;
        }
}

#endif