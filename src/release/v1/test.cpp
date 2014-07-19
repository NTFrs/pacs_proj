#include <iostream>
#include "Levy.hpp"

// #include <vector>
// #include "models.hpp"

using namespace std;

int main(){
	
		/*
		KouModel Mod(100,0.2, 0.5, 3, 2, 4);
		Model * ModPtr=&Mod;
		std::vector<Model *> Mods;
		Mods.push_back(ModPtr);
		
		std::cout<< Mod.density(0.5)<< std::endl;
		std::cout<< (*ModPtr).density(0.5)<< std::endl;
        std::cout<< (*Mods[0]).density(0.5)<< std::endl;
		*/
 
        BlackScholesModel model(95., 0.120381);
        
        BlackScholesModel model1(80., 0.1256);
        BlackScholesModel model2(120., 0.2);
        
        KouModel model3(95, 0.120381, 0.20761, 0.330966, 9.6599, 3.13868);
        
        EuropeanOptionPrice<1> foo(OptionType::Call, model.get_pointer(), 0.0367, 1., 90., 10, 100);
        
        EuropeanOptionLogPrice<1> foo2
        (OptionType::Call, model.get_pointer(), 0.0367, 1., 90., 10, 100);
        
        EuropeanOptionPrice<1> goofy(OptionType::Put, model.get_pointer(), 0.0367, 1., 90., 10, 100);
        
        AmericanOptionPrice<1> minnie(model.get_pointer(), 0.0367, 1., 90., 10, 100);
        
        EuropeanOptionPrice<2> mickey(OptionType::Call, model1.get_pointer(), model2.get_pointer(),
                                 -0.2, 0.1, 1., 200., 7, 100);
        
        EuropeanOptionPrice<1> duffy(OptionType::Call, model3.get_pointer(), 0.0367, 1., 90., 8, 100);
        
        foo.run();
        foo2.run();
        goofy.run();
        minnie.run();
        mickey.run();
        duffy.run();
        
        cout<<foo.get_price()<<"\n";
        cout<<foo2.get_price()<<"\n";
        cout<<goofy.get_price()<<"\n";
        cout<<minnie.get_price()<<"\n";
        cout<<mickey.get_price()<<"\n";
        cout<<duffy.get_price()<<"\n";
        
        
        cout<<"TARGET (Premia)\n"
            <<"PDE  1d Call  (10, 100): 9.62216.\n"
            <<"PDE  1d Put   (10, 100): 1.43609.\n"
            <<"PDE  1d PutAm (10, 100): 1.56720.\n"
            <<"PDE  2d Call  (7,  100): 21.3480.\n"
            <<"PIDE 1d Call  (8,  100): 12.3683.\n";
        
        return 0;
        
}