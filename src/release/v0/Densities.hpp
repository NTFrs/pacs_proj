#ifndef __densities_hpp
#define __densities_hpp

#include "deal_ii.hpp"

template <unsigned dim>
class Kou_Density: public Function<dim>
{
public:
	Kou_Density(double p_,  double lambda_, double lambda_plus_,  double lambda_minus_)
        :
        Function<dim>(),  
        p(p_), 
        lambda(lambda_), 
	lambda_plus(lambda_plus_),
        lambda_minus(lambda_minus_)
        {};
        
        virtual inline double get_p() const { return p; };
        virtual inline double get_lambda () const { return lambda; };
        virtual inline double get_lambda_p () const { return lambda_plus; };
        virtual inline double get_lambda_m () const { return lambda_minus; };
	
        virtual double  value           (const Point<dim> &p_,  const unsigned int component=0) const;
        virtual void    value_list      (const std::vector<Point<dim> > &points,
                                         std::vector<double> &values,
                                         const unsigned int component = 0) const;
private:
        double p;
        double lambda;
        double lambda_plus;
        double lambda_minus;
};

template<unsigned dim>
double Kou_Density<dim>::value(const Point<dim> &p_,  const unsigned int component) const
{
	Assert (component == 0, ExcInternalError());
	if (p_[0]>0)
                return p*lambda*lambda_plus*exp(-lambda_plus*p_[0]);
	else
                return (1-p)*lambda*lambda_minus*exp(lambda_minus*p_[0]);
        
}



template<unsigned dim>
void Kou_Density<dim>::value_list(const std::vector<Point<dim> > &points, std::vector<double> &values, const unsigned int component) const
{
	Assert (values.size() == points.size(),
                ExcDimensionMismatch (values.size(), points.size()));
	Assert (component == 0, ExcInternalError());
        
	const unsigned int n_points=points.size();
        
	for (unsigned int i=0;i<n_points;++i)
                if (points[i][0]>0)
                        values[i]=p*lambda*lambda_plus*exp(-lambda_plus*points[i][0]);
                else
                        values[i]=(1-p)*lambda*lambda_minus*exp(lambda_minus*points[i][0]);
}

template<unsigned dim>
class Merton_Density: public Function<dim> {
public:
        Merton_Density():Function<dim>(){
                cerr<<"Densità di Merton non ancora implementata!\n";
        };
        
};

#endif