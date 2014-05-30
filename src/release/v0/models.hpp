#ifndef __models_hpp
#define __models_hpp

#include <iostream>

class BlackScholesModel
{
private:
        double S0;
        double sigma;
        
public:
        BlackScholesModel(double S0_,
                          double sigma_):
        S0(S0_),
        sigma(sigma_)
        {};
        
        virtual inline double get_spot() {return S0;}
        virtual inline double get_vol()  {return sigma;}        

        virtual std::ostream& operator<<(std::ostream& OS){return OS;};
};

class KouModel: public BlackScholesModel
{
private:
        double p;
        double lambda;
        double lambda_plus;
        double lambda_minus;
        
public:
        KouModel(double S0_,
                 double sigma_,
                 double p_,
                 double lambda_,
                 double lambda_plus_,
                 double lambda_minus_)
        :
        BlackScholesModel(S0_, sigma_),
        p(p_),
        lambda(lambda_),
        lambda_plus(lambda_plus_),
        lambda_minus(lambda_minus_)
        {};
        
        virtual inline double get_p()           {return p;}
        virtual inline double get_lambda()      {return lambda;}
        virtual inline double get_lambda_p()    {return lambda_plus;}
        virtual inline double get_lambda_m()    {return lambda_minus;}
        
        virtual std::ostream& operator<<(std::ostream& OS){return OS;};
};

class MertonModel: public BlackScholesModel
{
private:
        double nu;
        double delta;
        double C;

public:
        MertonModel(double S0_,
                    double sigma_,
                    double nu_,
                    double delta_,
                    double C_)
        :
        BlackScholesModel(S0_, sigma_),
        nu(nu_),
        delta(delta_),
        C(C_)
        {};
        
        virtual std::ostream& operator<<(std::ostream& OS){return OS;};
};

#endif