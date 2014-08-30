#include <iostream>
#include "Levy.hpp"

int main()
{

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


                // We evaluate here the fisrt and the second derivatives of the solution
                vector< dealii::Point<1> >      grid;
                vector<double>                  solution;

                foo->get_mesh(grid);
                foo->get_solution(solution);

                vector<double> first_d(grid.size()-2);
                vector<double> second_d(grid.size()-2);

                double dx=grid[2][0]-grid[1][0];

                for (unsigned i=0; i<grid.size()-2; ++i)
                {
                        first_d[i]=(solution[i+2]-solution[i])/(2*dx);
                        second_d[i]=(solution[i+2]+solution[i]-2*solution[i+1])/(dx*dx);
                }

                ofstream stream;
                stream.open("derivatives.m");

                if (stream.is_open())
                {
                        stream<<"mesh=[ ";
                        for (unsigned i=0; i<grid.size()-3; ++i)
                        {
                                stream<<grid[i+1][0]<<"; ";
                        }
                        stream<<grid[grid.size()-2]<<" ];\n";
                        stream<<"first_der=[ ";
                        for (unsigned i=0; i<grid.size()-3; ++i)
                        {
                                stream<<first_d[i]<<"; ";
                        }
                        stream<<first_d[grid.size()-2]<<" ];\n";
                        stream<<"second_der=[ ";
                        for (unsigned i=0; i<grid.size()-3; ++i)
                        {
                                stream<<second_d[i]<<"; ";
                        }
                        stream<<second_d[grid.size()-2]<<" ];\n";
                }
                else
                {
                        throw(ios_base::failure("Unable to open the file."));
                }

                stream.close();
                //

                cout<<"The price of the option is "<<foo->get_price()<<", evaluated in "<<times.second/1.e6<<".\n";
        }

        cout<<"Press return to continue...\n";
        cin.get();

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
        cin.get();

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
        cin.get();

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
        cin.get();

        cout<<"*** Do you want to perform some convergence tests? (y/n) ";
        string s;
        cin>>s;

        if (s=="y" || s=="yes" || s=="Yes" || s=="YES" || s=="Y")
        {
                cout<<"*** Convergence test for 1d Call, Black&Scholes\n";

                {
                        const unsigned dim=5;

                        array<unsigned, dim>      ref= {8, 9, 10, 11, 12};
                        array<unsigned, dim>      steps= {25, 50, 100, 250, 500};

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

                        for (unsigned i=0; i<dim; ++i)
                        {

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
                        for (unsigned i=0; i<dim; ++i)
                        {
                                cout<<pow(2, ref[i])<<"/"<<steps[i]<<"\t\t";
                        }
                        cout<<"\np:\t";
                        for (unsigned i=0; i<dim; ++i)
                        {
                                cout<<p[i]<<"("<<times_p[i]/1.e6<<"s)\t";
                        }
                        cout<<"\nl:\t";
                        for (unsigned i=0; i<dim; ++i)
                        {
                                cout<<l[i]<<"("<<times_l[i]/1.e6<<"s)\t";
                        }
                        cout<<"\n";

                }

                cout<<"*** Convergence test for 2d Put, Black&Scholes\n";

                {
                        const unsigned dim=5;

                        array<unsigned, dim>      ref= {4, 5, 6, 7, 8};
                        array<unsigned, dim>      steps= {25, 50, 100, 250, 500};

                        array<double, dim>        p, l, times_p, times_l;

                        auto clarabell=Factory::instance()->create(ExerciseType::EU,
                                        OptionType::Put,
                                        Transformation::Price,
                                        model1.get_pointer(),
                                        model2.get_pointer(),
                                        -0.2, 0.1, 1., 200., 1, 25);

                        auto amelia=Factory::instance()->create(ExerciseType::EU,
                                                                OptionType::Put,
                                                                Transformation::LogPrice,
                                                                model1.get_pointer(),
                                                                model2.get_pointer(),
                                                                -0.2, 0.1, 1., 200., 1, 25);

                        clarabell->set_verbose(false);
                        clarabell->set_timing(true);

                        amelia->set_verbose(false);
                        amelia->set_timing(true);

                        for (unsigned i=0; i<dim; ++i)
                        {

                                clarabell->reset();
                                amelia->reset();

                                clarabell->set_refs(ref[i]);
                                amelia->set_refs(ref[i]);

                                clarabell->set_timestep(steps[i]);
                                amelia->set_timestep(steps[i]);

                                cout<<"Evaluating options with grid size="<<clarabell->get_number_of_cells()
                                    <<" and timesteps="<<clarabell->get_number_of_timesteps()<<"\n";

                                clarabell->run();
                                amelia->run();

                                p[i]=clarabell->get_price();
                                l[i]=amelia->get_price();

                                auto time_p=clarabell->get_times();
                                auto time_l=amelia->get_times();

                                times_p[i]=time_p.second;
                                times_l[i]=time_l.second;

                        }

                        cout<<"\t";
                        for (unsigned i=0; i<dim; ++i)
                        {
                                cout<<pow(2, ref[i])<<"/"<<steps[i]<<"\t\t";
                        }
                        cout<<"\np:\t";
                        for (unsigned i=0; i<dim; ++i)
                        {
                                cout<<p[i]<<"("<<times_p[i]/1.e6<<"s)\t";
                        }
                        cout<<"\nl:\t";
                        for (unsigned i=0; i<dim; ++i)
                        {
                                cout<<l[i]<<"("<<times_l[i]/1.e6<<"s)\t";
                        }
                        cout<<"\n";

                }

        }
        else if (s!="n" && s!="N" && s!="no" && s!="NO" && s!="No")
        {
                throw(logic_error("Something went wrong..."));
        }

        return 0;
}
