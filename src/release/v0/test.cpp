#include <iostream>
#include "pide.hpp"

using namespace std;

int main(){
        
        Model<ModelType::BlackScholes> model0(95., 0.120381);
        
        BlackScholesModel model(95., 0.120381);
        
        Option<1, ModelType::BlackScholes> foo(model, 0.0367, 1., 90., 10, 100);
        
        foo.run();
        
        cout<<foo.get_price()<<"\n";
}