#include <iostream>
//#include "Levy.hpp"
#include "Factory.hpp"

int main(){
	
        using namespace dealii;
        using namespace std;
        
        BlackScholesModel model(95., 0.120381);
        
        BlackScholesModel model1(80., 0.1256);
        BlackScholesModel model2(120., 0.2);
        
        KouModel model3(95, 0.120381, 0.20761, 0.330966, 9.65997, 3.13868);

        KouModel model4(80, 0.1256, 0.20761, 0.330966, 9.65997, 3.13868);
        KouModel model5(120, 0.2, 0.20761, 0.330966, 9.65997, 3.13868);
        
        MertonModel model6(80., 0.2, -0.390078, 0.338796, 0.174814);
        MertonModel model7(120., 0.2, -0.390078, 0.338796, 0.174814);
        
        OptionType tipo(OptionType::Call);
        
        EuropeanOptionPrice<1> a(tipo, model6.get_pointer(), 0.0367, 1., 90., 10, 100);
        EuropeanOptionLogPrice<1> b(tipo, model6.get_pointer(), 0.0367, 1., 90., 10, 100);
        
        EuropeanOptionPrice<2> c
        (tipo, model6.get_pointer(), model7.get_pointer(),
         -0.2, 0.1, 1., 200., 6, 100);
        
        EuropeanOptionLogPrice<2> d
        (tipo, model6.get_pointer(), model7.get_pointer(),
         -0.2, 0.1, 1., 200., 6, 100);
        
        a.set_scale_factor(0.5);
        b.set_scale_factor(0.5);
        c.set_scale_factor(0.5);
        d.set_scale_factor(0.5);
        
        a.set_timing(true);
        b.set_timing(true);
        c.set_timing(true);
        d.set_timing(true);
        
        a.set_print(true);
        b.set_print(true);
        c.set_print(true);
        d.set_print(true);

        c.set_print_grid(true);
        d.set_print_grid(true);
        
// 		a.set_refine_status(true, 0.2, 0);
// 		b.set_refine_status(true, 0.2, 0);
// 		c.set_refine_status(true, 0.03, 0.);
// 		d.set_refine_status(true, 0.2, 0.);
        
//         a.run();
//         b.run();
//         c.run();
        d.run();
        

        
//         cout<<"1d Price "<< a.get_price()<<"\n";
// 		cout<<"1d LogPrice "<< b.get_price()<<"\n";
// 		cout<<"2d Price "<< c.get_price()<<"\n";
		cout<<"2d LogPrice "<< d.get_price()<<"\n";

		/*        
        auto x=a.get_times();
        cout<<x.first/1.e6<<" "<<x.second/1.e6<<"\n";
        auto y=b.get_times();
        cout<<y.first/1.e6<<" "<<y.second/1.e6<<"\n";
        auto z=c.get_times();
        cout<<z.first/1.e6<<" "<<z.second/1.e6<<"\n";
        auto w=d.get_times();
        cout<<w.first/1.e6<<" "<<w.second/1.e6<<"\n";
        */

	    cout<<"My targets with scale factor 0.5\n"
		<<"PDE  1d Call price model (10, 100): 9.66063\n"
		<<"PDE  1d Call logprice model  (10, 100): 9.6606\n"
		<<"PDE  2d Call price model1 model2 (6, 100): 21.6607\n"
		<<"PDE  2d Call logprice model1 model2 (6,  100): 21.6065\n"
		<<"And\n"
		<< "Targets for levy with scale factor 0.5\n"
		<< "1d Kou Price model 3 12.4258\n"
		<< "1d Kou LogPrice model 3 12.4264\n"
		<< "2d Kou Price models 3 4 25.1415\n"
		<< " 2d Kou LogPrice models 3 4 25.1219\n"
		<< "And\n"
		<< "1d Merton Price model 6 5.05387\n"
		<< "1d Merton LogPrice model 6 5.05362\n"
		<< "2d Merton Price model 6 7 25.5609\n"
		<< "2d Merton LogPrice model 6 7 \n";
        
        return 0;
        
}