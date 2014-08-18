#include <iostream>
#include "Factory.hpp"

int main(){
	
        using namespace dealii;
        using namespace std;
        
        cout<<"*** European Put 1d.\n";
        
        // We create here a BlackScholesModel object with spot price and volatility
        BlackScholesModel model(95., 0.120381);
        
        {
                // We create an object of type Option, a European Put with the Price transformation
                // with the model created above.
                auto foo=Factory::instance()->create(ExerciseType::EU,
                                                     OptionType::Put,
                                                     Transformation::Price,
                                                     model.get_pointer(),
                                                     0.0367, 1., 90., 12, 250);
                
                foo->set_print(true);
                foo->set_timing(true);
                foo->run();
                
                auto times=foo->get_times();
                
                cout<<"The price of the option is "<<foo->get_price()<<", evaluated in "<<times.second/1.e6<<".\n";
        }
        
        cout<<"Press return to continue...\n";
        //cin.get();
        
        {
                // We create an object of type Option, a European Put with the LogPrice transformation
                // with the model created above.
                auto goofy=Factory::instance()->create(ExerciseType::EU,
                                                       OptionType::Put,
                                                       Transformation::LogPrice,
                                                       model.get_pointer(),
                                                       0.0367, 1., 90., 12, 250);
                
                goofy->set_print(true);
                goofy->set_timing(true);
                goofy->run();
                
                auto times=goofy->get_times();
                
                cout<<"The price of the option is "<<goofy->get_price()<<", evaluated in "<<times.second/1.e6<<"s.\n";
        }
        
        cout<<"Press return to continue...\n";
        //cin.get();
        
        cout<<"*** European Call 2d.\n";
        
        // We create here two BlackScholesModel objects with spot price and volatility
        BlackScholesModel model1(80., 0.1256);
        BlackScholesModel model2(120., 0.2);
        
        {
                auto duffy=Factory::instance()->create(ExerciseType::EU,
                                                       OptionType::Call,
                                                       Transformation::Price,
                                                       model1.get_pointer(),
                                                       model2.get_pointer(),
                                                       -0.2, 0.1, 1., 200., 7, 100);
                
                duffy->set_print_grid(true);
                duffy->set_print(true);
                duffy->set_timing(true);
                duffy->run();
                
                auto times=duffy->get_times();
                
                cout<<"The price of the option is "<<duffy->get_price()<<", evaluated in "<<times.second/1.e6<<"s.\n";
        }
        
        cout<<"Press return to continue...\n";
        //cin.get();
        
        {
                auto mickey=Factory::instance()->create(ExerciseType::EU,
                                                        OptionType::Call,
                                                        Transformation::LogPrice,
                                                        model1.get_pointer(),
                                                        model2.get_pointer(),
                                                        -0.2, 0.1, 1., 200., 7, 100);
                
                mickey->set_print_grid(true);
                mickey->set_print(true);
                mickey->set_timing(true);
                mickey->run();
                
                auto times=mickey->get_times();
                
                cout<<"The price of the option is "<<mickey->get_price()<<", evaluated in "<<times.second/1.e6<<"s.\n";
        }
        
        cout<<"Press return to continue...\n";
        //cin.get();
        
        cout<<"*** Do you want to perform some convergence tests? (y/n) ";
        string s;
        cin>>s;
        
        if (s=="y") {
                cout<<"*** Convergence test for 1d Call, Black&Scholes\n";
                
                {
                        const unsigned dim=5;
                        
                        array<unsigned, dim>      ref={8, 9, 10, 11, 12};
                        array<unsigned, dim>      steps={25, 50, 100, 250, 500};
                        
                        array<double, dim>        p, l, times_p, times_l;
                        
                        auto minnie=Factory::instance()->create(ExerciseType::EU,
                                                                OptionType::Call,
                                                                Transformation::Price,
                                                                model.get_pointer(),
                                                                0.0367, 1., 90., 1, 25);
                        
                        auto daisy=Factory::instance()->create(ExerciseType::EU,
                                                               OptionType::Call,
                                                               Transformation::LogPrice,
                                                               model.get_pointer(),
                                                               0.0367, 1., 90., 1, 25);
                        
                        minnie->set_verbose(false);
                        minnie->set_timing(true);
                        
                        daisy->set_verbose(false);
                        daisy->set_timing(true);
                        
                        for (unsigned i=0; i<dim; ++i) {
                                
                                minnie->reset();
                                daisy->reset();
                                
                                minnie->set_refs(ref[i]);
                                daisy->set_refs(ref[i]);
                                
                                minnie->set_timestep(steps[i]);
                                daisy->set_timestep(steps[i]);
                                
                                cout<<"Evaluating options with grid size="<<minnie->get_number_of_cells()
                                <<" and timesteps="<<minnie->get_number_of_timesteps()<<"\n";
                                
                                minnie->run();
                                daisy->run();
                                
                                p[i]=minnie->get_price();
                                l[i]=daisy->get_price();
                                
                                auto time_p=minnie->get_times();
                                auto time_l=daisy->get_times();
                                
                                times_p[i]=time_p.second;
                                times_l[i]=time_l.second;
                                
                        }
                        
                        cout<<"\t";
                        for (unsigned i=0; i<dim; ++i) {
                                cout<<pow(2, ref[i])<<"/"<<steps[i]<<"\t\t";
                        }
                        cout<<"\np:\t";
                        for (unsigned i=0; i<dim; ++i) {
                                cout<<p[i]<<"("<<times_p[i]/1.e6<<"s)\t";
                        }
                        cout<<"\nl:\t";
                        for (unsigned i=0; i<dim; ++i) {
                                cout<<l[i]<<"("<<times_l[i]/1.e6<<"s)\t";
                        }
                        cout<<"\n";
                        
                }
                
                cout<<"*** Convergence test for 2d Put, Black&Scholes\n";
                
                {
                        const unsigned dim=5;
                        
                        array<unsigned, dim>      ref={4, 5, 6, 7, 8};
                        array<unsigned, dim>      steps={25, 50, 100, 250, 500};
                        
                        array<double, dim>        p, l, times_p, times_l;
                        
                        auto minnie=Factory::instance()->create(ExerciseType::EU,
                                                                OptionType::Put,
                                                                Transformation::Price,
                                                                model1.get_pointer(),
                                                                model2.get_pointer(),
                                                                -0.2, 0.1, 1., 200., 1, 25);
                        
                        auto daisy=Factory::instance()->create(ExerciseType::EU,
                                                               OptionType::Put,
                                                               Transformation::LogPrice,
                                                               model1.get_pointer(),
                                                               model2.get_pointer(),
                                                               -0.2, 0.1, 1., 200., 1, 25);
                        
                        minnie->set_verbose(false);
                        minnie->set_timing(true);
                        
                        daisy->set_verbose(false);
                        daisy->set_timing(true);
                        
                        for (unsigned i=0; i<dim; ++i) {
                                
                                minnie->reset();
                                daisy->reset();
                                
                                minnie->set_refs(ref[i]);
                                daisy->set_refs(ref[i]);
                                
                                minnie->set_timestep(steps[i]);
                                daisy->set_timestep(steps[i]);
                                
                                cout<<"Evaluating options with grid size="<<minnie->get_number_of_cells()
                                <<" and timesteps="<<minnie->get_number_of_timesteps()<<"\n";
                                
                                minnie->run();
                                daisy->run();
                                
                                p[i]=minnie->get_price();
                                l[i]=daisy->get_price();
                                
                                auto time_p=minnie->get_times();
                                auto time_l=daisy->get_times();
                                
                                times_p[i]=time_p.second;
                                times_l[i]=time_l.second;
                                
                        }
                        
                        cout<<"\t";
                        for (unsigned i=0; i<dim; ++i) {
                                cout<<pow(2, ref[i])<<"/"<<steps[i]<<"\t\t";
                        }
                        cout<<"\np:\t";
                        for (unsigned i=0; i<dim; ++i) {
                                cout<<p[i]<<"("<<times_p[i]/1.e6<<"s)\t";
                        }
                        cout<<"\nl:\t";
                        for (unsigned i=0; i<dim; ++i) {
                                cout<<l[i]<<"("<<times_l[i]/1.e6<<"s)\t";
                        }
                        cout<<"\n";
                        
                }

        }
        else if (s!="n") {
                throw(logic_error("Something went wrong..."));
        }
        
        return 0;
}
