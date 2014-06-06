#ifndef __models_hpp
#define __models_hpp

#include <iostream>

class Model
{
private:
        double S0;
        double sigma;
        
public:
        Model()=default;
        Model(double S0_,
              double sigma_):
        S0(S0_),
        sigma(sigma_)
        {};
        
        virtual ~Model()=default;
        
        virtual inline double get_spot()        const   {return S0;};
        virtual inline double get_vol()         const   {return sigma;};
        virtual inline Model* get_pointer()             {return this;};

        virtual std::ostream& operator<<(std::ostream& OS)=0;
};

class BlackScholesModel: public Model
{
public:
        BlackScholesModel()=default;
        
        BlackScholesModel(double S0_,
                          double sigma_):
        Model(S0_, sigma_)
        {};
        
        virtual std::ostream& operator<<(std::ostream& OS){return OS;};
};

class KouModel: public Model
{
private:
        double p;
        double lambda;
        double lambda_plus;
        double lambda_minus;
        
public:
        KouModel()=default;
        
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
        
        virtual inline double get_p()           {return p;}
        virtual inline double get_lambda()      {return lambda;}
        virtual inline double get_lambda_p()    {return lambda_plus;}
        virtual inline double get_lambda_m()    {return lambda_minus;}
        
        virtual std::ostream& operator<<(std::ostream& OS){return OS;};
};

class MertonModel: public Model
{
private:
        double nu;
        double delta;
        double C;

public:
        MertonModel()=default;
        
        MertonModel(double S0_,
                    double sigma_,
                    double nu_,
                    double delta_,
                    double C_)
        :
        Model(S0_, sigma_),
        nu(nu_),
        delta(delta_),
        C(C_)
        {};
        
        virtual std::ostream& operator<<(std::ostream& OS){return OS;};
};

#endif