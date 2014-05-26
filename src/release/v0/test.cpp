#include <iostream>
#include "pide.hpp"

using namespace std;

int main(){
        BlackScholesModel model(95., 0.120381);
        Option<1> foo(model, 0.0367, 1., 90., 12, 100);
        foo.run();
        cout<<foo.get_price()<<"\n";
}