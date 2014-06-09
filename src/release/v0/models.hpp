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

        virtual inline double get_lambda()      const   { return 0.; };
        
        virtual inline double get_p()           const   { return 0.; };
        virtual inline double get_lambda_p()    const   { return 0.; };
        virtual inline double get_lambda_m()    const   { return 0.; };

        virtual inline double get_nu()          const   =0;
        virtual inline double get_delta()       const   =0;
        
};

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
        BlackScholesModel()=default;
        
        BlackScholesModel(double S0_,
                          double sigma_):
        Model(S0_, sigma_)
        {};
        
};

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
        
        virtual inline double get_p()           const   {return p;}
        virtual inline double get_lambda()      const   {return lambda;}
        virtual inline double get_lambda_p()    const   {return lambda_plus;}
        virtual inline double get_lambda_m()    const   {return lambda_minus;}
        
};

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