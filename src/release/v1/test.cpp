#include <iostream>
//#include "Levy.hpp"
#include "Factory.hpp"

using namespace std;
using namespace dealii;

int main(){
	
        using namespace dealii;
        
        BlackScholesModel model(95., 0.120381);
        
        BlackScholesModel model1(80., 0.1256);
        BlackScholesModel model2(120., 0.2);
        
        KouModel model3(95, 0.120381, 0.20761, 0.330966, 9.65997, 3.13868);
        KouModel model4(80, 0.1256, 0.20761, 0.330966, 9.65997, 3.13868);
        KouModel model5(120, 0.2, 0.20761, 0.330966, 9.65997, 3.13868);
        
        //EuropeanOptionPrice<1> foo(OptionType::Put, model.get_pointer(), 0.0367, 1., 90., 12, 250);
        
        //EuropeanOptionLogPrice<1> foo2
        //(OptionType::Put, model.get_pointer(), 0.0367, 1., 90., 12, 250);
        
        //EuropeanOptionPrice<1> duffy(OptionType::Call, model3.get_pointer(), 0.0367, 1., 90., 8, 100);
        //EuropeanOptionLogPrice<1> duffy2(OptionType::Call, model3.get_pointer(), 0.0367, 1., 90., 8, 100);
        
        
        AmericanOptionPrice<1> minnie(model.get_pointer(), 0.0367, 1., 90., 12, 100);
        
        AmericanOptionLogPrice<1> minnie2(model.get_pointer(), 0.0367, 1., 90., 12, 100);
        /*
         EuropeanOptionPrice<2> mickey(OptionType::Call, model1.get_pointer(), model2.get_pointer(),
         -0.2, 0.1, 1., 200., 7, 100);
         
         EuropeanOptionLogPrice<2> mickey2(OptionType::Call, model1.get_pointer(), model2.get_pointer(),
         -0.2, 0.1, 1., 200., 7, 100);
         
         EuropeanOptionPrice<2> goofy(OptionType::Call, model5.get_pointer(), model4.get_pointer(),
         -0.2, 0.1, 1., 200., 6, 25);
         
         EuropeanOptionLogPrice<2> goofy2(OptionType::Call, model5.get_pointer(), model4.get_pointer(),
         -0.2, 0.1, 1., 200., 6, 25);
         
         */
        //foo.run();
        //foo2.run();
        //duffy.run();
        //duffy2.run();
        minnie.run();
        minnie2.run();
        //mickey.run();
        //mickey2.run();
        //goofy.run();
        //goofy2.run();
        
        //cout<<foo.get_price()<<"\n";
        //cout<<foo2.get_price()<<"\n";
        //cout<<duffy.get_price()<<"\n";
        //cout<<duffy2.get_price()<<"\n";
        cout<<minnie.get_price()<<"\n";
        cout<<minnie2.get_price()<<"\n";
        //cout<<mickey.get_price()<<"\n";
        //cout<<mickey2.get_price()<<"\n";
        //cout<<goofy.get_price()<<"\n";
        //cout<<goofy2.get_price()<<"\n";
        /*
        auto x=OptionFactory::instance()->create(ExerciseType::EU,
                                                 OptionType::Put,
                                                 Transformation::Price,
                                                 model.get_pointer(),
                                                 0.0367, 1., 90., 12, 250);
        
        x->run();
        cout<<x->get_price()<<"\n";
        */
        cout<<"TARGET (Premia)\n"
        <<"PDE  1d Call  (10, 100): 9.62216.\n"
        <<"PDE  1d Put   (10, 100): 1.43609.\n"
        <<"PDE  1d PutAm (10, 100): 1.56720.\n"
        <<"PDE  2d Call  (7,  100): 21.3480.\n"
        <<"PIDE 1d Call  (8,  100): 12.3683.\n"
        <<"PIDE 2d Call  MC       : 24.8000.\n";
        
        return 0;
        
}