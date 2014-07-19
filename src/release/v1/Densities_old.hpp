#ifndef __densities_hpp
#define __densities_hpp

#include "deal_ii.hpp"

//! Kou Density
/*!
 * This class handles the Kou Density Function. It inherites from the dealii's Function<dim>.
 */
template <unsigned dim>
class Kou_Density: public Function<dim>
{
public:
        //! Constructor
        /*!
         * \param p_            Probability of positives jumps
         * \param lambda_       Intensity of jumps
         * \param lambda_plus_  Intensity of positive jumps
         * \param lambda_minus_ Intensity of negative jumps
		 * \param ax_			On wich direction itworks
         */
	Kou_Density(unsigned ax_ , double p_,  double lambda_, double lambda_plus_,  double lambda_minus_)
        :
        Function<dim>(),  
        ax(ax_), 
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
	
        //! Function needed by dealii
        virtual double  value           (const Point<dim> &p_,  const unsigned int component=0) const;
        //! Function needed by dealii
        virtual void    value_list      (const std::vector<Point<dim> > &points,
                                         std::vector<double> &values,
                                         const unsigned int component = 0) const;
private:
		unsigned ax;
        double p;
        double lambda;
        double lambda_plus;
        double lambda_minus;
};

template<unsigned dim>
double Kou_Density<dim>::value(const Point<dim> &p_,  const unsigned int component) const
{
	Assert (component == 0, ExcInternalError());
	if (p_[ax]>0)
                return p*lambda*lambda_plus*exp(-lambda_plus*p_[ax]);
	else
                return (1-p)*lambda*lambda_minus*exp(lambda_minus*p_[ax]);
        
}



template<unsigned dim>
void Kou_Density<dim>::value_list(const std::vector<Point<dim> > &points, std::vector<double> &values, const unsigned int component) const
{
	Assert (values.size() == points.size(),
                ExcDimensionMismatch (values.size(), points.size()));
	Assert (component == 0, ExcInternalError());
        
	const unsigned int n_points=points.size();
        
	for (unsigned int i=0;i<n_points;++i)
                if (points[i][ax]>0)
                        values[i]=p*lambda*lambda_plus*exp(-lambda_plus*points[i][ax]);
                else
                        values[i]=(1-p)*lambda*lambda_minus*exp(lambda_minus*points[i][ax]);
}

//! Merton Density
/*!
 * This class handles the Merton Density Function. It inherites from the dealii's Function<dim>.
 * NOT CODED YET!
 */
template<unsigned dim>
class Merton_Density: public Function<dim> {
public:
        Merton_Density():Function<dim>(){
                cerr<<"Densità di Merton non ancora implementata!\n";
        };
        
};

#endif