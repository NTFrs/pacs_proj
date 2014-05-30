#include <iostream>
#include "Levy.hpp"

using namespace std;

int main(){
        
        BlackScholesModel model1(95., 0.120381);
        BlackScholesModel model2(110., 0.2);
        
        EuropeanOption<1> foo(OptionType::Call, model1, 0.0367, 1., 90., 10, 100);
        foo.run();
        
        EuropeanOption<1> goofy(OptionType::Put, model1, 0.0367, 1., 90., 10, 100);
        goofy.run();
        
        AmericanOption<1> minnie(model1, 0.0367, 1., 90., 10, 100);
        minnie.run();
        
        cout<<foo.get_price()<<"\n";
        cout<<goofy.get_price()<<"\n";
        cout<<minnie.get_price()<<"\n";
        
        cout<<"TARGET (blsprice MATLAB)\n"
            <<"PDE 1d Call (10, 100): 9.6652.\n"
            <<"PDE 1d Put  (10, 100): 1.4221.\n"
            <<"PDE 1d PutAm(10, 100): 1.5644.\n";
        
}