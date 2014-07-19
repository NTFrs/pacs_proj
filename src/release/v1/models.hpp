#ifndef __models_hpp
#define __models_hpp

#include <iostream>
#include <cmath> 

//! Abstract class Model
/*!
 * This class gives the idea of how we intend a model class.
 */
class Model
{
private:
        double S0;
        double sigma;
        
public:
        //! Sintetic Constructor
        Model()=default;
        //! Default Constructor
        /*!
         * \param S0_           Spot Price
         * \param sigma_        Volatility
         */
        Model(double S0_,
              double sigma_):
        S0(S0_),
        sigma(sigma_)
        {};
        
        virtual ~Model()=default;
        
        //! Returns the Spot Price
        virtual inline double get_spot()        const   {return S0;};
        //! Returns the Volatility
        virtual inline double get_vol()         const   {return sigma;};
        //! Returns the pointer to che class (needed in OptionBase<dim> constructors)
        virtual inline Model* get_pointer()             {return this;};

        virtual inline double get_lambda()      const   { return 0.; };
        
        virtual inline double get_p()           const   { return 0.; };
        virtual inline double get_lambda_p()    const   { return 0.; };
        virtual inline double get_lambda_m()    const   { return 0.; };

        virtual inline double get_nu()          const   =0;
        virtual inline double get_delta()       const   =0;
        
        virtual double density(double pt) const =0;
};

//! Black&Scholes Model
/*!
 * This is the simplest and the most used model. With this model, the object OptionBase<dim> will solve the the Black&Scholes' PDE, without the integral part.
 */
class BlackScholesModel: public Model
{
private:
        virtual inline double get_p()           const   { return 0.; };
        virtual inline double get_lambda()      const   { return 0.; };
        virtual inline double get_lambda_p()    const   { return 0.; };
        virtual inline double get_lambda_m()    const   { return 0.; };
        
        virtual inline double get_nu()          const   { return 0.; };
        virtual inline double get_delta()       const   { return 0.; };
        
public:
        //! Sintetic Constructor
        BlackScholesModel()=default;
        
        //! Default Constructor
        /*!
         * \param S0_           Spot Price
         * \param sigma_        Volatility
         */
        BlackScholesModel(double S0_,
                          double sigma_):
        Model(S0_, sigma_)
        {};
        virtual double density(double pt) const {return 0;};
};

//! Kou Model
/*!
 * This class describes the Kou Model, which describes the classic geometric brownian motion with jumps. The pdf of jumps is modeled as an exponential.
 */
class KouModel: public Model
{
private:
        
        virtual inline double get_nu()          const   { return 0.; };
        virtual inline double get_delta()       const   { return 0.; };
        
        double p;
        double lambda;
        double lambda_plus;
        double lambda_minus;
        
public:
        //! Sintetic Constructor
        KouModel()=default;
        
        //! Default Constructor
        /*!
         * \param S0_           Spot Price
         * \param sigma_        Volatility
         * \param p_            Probability of positives jumps
         * \param lambda_       Intensity of jumps
         * \param lambda_plus_  Intensity of positive jumps
         * \param lambda_minus_ Intensity of negative jumps
         */
        KouModel(double S0_,
                 double sigma_,
                 double p_,
                 double lambda_,
                 double lambda_plus_,
                 double lambda_minus_)
        :
        Model(S0_, sigma_),
        p(p_),
        lambda(lambda_),
        lambda_plus(lambda_plus_),
        lambda_minus(lambda_minus_)
        {};
        
        virtual inline double get_p()           const   {return p;}
        virtual inline double get_lambda()      const   {return lambda;}
        virtual inline double get_lambda_p()    const   {return lambda_plus;}
        virtual inline double get_lambda_m()    const   {return lambda_minus;}
        
        virtual double density(double pt) const ;
        
};

double KouModel::density(double pt) const
{
	if (pt>0)
	return p*lambda*lambda_plus*exp(-lambda_plus*pt);
	else
	return (1-p)*lambda*lambda_minus*exp(lambda_minus*pt);
}


//! Merton Model
/*!
 * This class describes the Merton Model, which describes the classic geometric brownian motion with jumps. The pdf of jumps is modeled as a gaussian function.
 */
class MertonModel: public Model
{
private:
        
        virtual inline double get_p()           const   { return 0.; };
        virtual inline double get_lambda_p()    const   { return 0.; };
        virtual inline double get_lambda_m()    const   { return 0.; };
        
        double nu;
        double delta;
        double lambda;

public:
        MertonModel()=default;
        
        MertonModel(double S0_,
                    double sigma_,
                    double nu_,
                    double delta_,
                    double lambda_)
        :
        Model(S0_, sigma_),
        nu(nu_),
        delta(delta_),
        lambda(lambda_)
        {};
        
        virtual inline double get_nu()          const   { return nu; };
        virtual inline double get_delta()       const   { return delta; };
        virtual inline double get_lambda()      const   { return lambda; }
        
};

#endif