#include <iostream>
#include "pide.hpp"

using namespace std;

int main(){
        BlackScholesModel model(95., 0.12);
        KouModel model2(95., 0.12, 1., 2., 3., 4.);
        Option<1> foo(model2, 0.0367, 1., 10, 10);
}