#ifndef __densities_hpp
#define __densities_hpp

#include "deal_ii.hpp"


class DensityBase {
        
public:
	virtual double operator()(double pt) const =0;
};

//! Kou Density
/*!
 * This class is a functor to handle the Kou Density Function. It inherites from DensityBase.
 */

class Kou_Density: public DensityBase {
        
public:
	//! Constructor
	/*!
	 * \param p_            Probability of positives jumps
	 * \param lambda_       Intensity of jumps
	 * \param lambda_plus_  Intensity of positive jumps
	 * \param lambda_minus_ Intensity of negative jumps
	 */
	Kou_Density(double p_,  double lambda_, double lambda_plus_,  double lambda_minus_):  
	p(p_), 
	lambda(lambda_), 
	lambda_plus(lambda_plus_),
	lambda_minus(lambda_minus_)
	{};
	
	//! Returns the probability of positives jumps
	virtual inline double get_p() const { return p; };
	//! Returns the intensity of jumps
	virtual inline double get_lambda () const { return lambda; };
	//! Returns the intensity of positive jumps
	virtual inline double get_lambda_p () const { return lambda_plus; };
	//! Returns the intensity of negative jumps
	virtual inline double get_lambda_m () const { return lambda_minus; };
	
	virtual double operator()(double pt) const;
	
private:
	double p;
	double lambda;
	double lambda_plus;
	double lambda_minus;
};


double Kou_Density::operator()(double pt ) const
{
	if (pt>0)
                return p*lambda*lambda_plus*exp(-lambda_plus*pt);
	else
                return (1-p)*lambda*lambda_minus*exp(lambda_minus*pt);
}



//! Merton Density
/*!
 * This class handles the Merton Density Function. It inherites from the dealii's Function<dim>.
 * NOT CODED YET!
 */
class Merton_Density: public DensityBase {
public:
        Merton_Density(){
                std::cerr<<"Densità di Merton non ancora implementata!\n";
        };
        
};

#endif