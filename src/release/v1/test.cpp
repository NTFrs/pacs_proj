#include <iostream>
//#include "Levy.hpp"
#include "Factory.hpp"

#include <ctime>

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
        
        MertonModel model6(80., 0.2, -0.390078, 0.338796, 0.174814);
        MertonModel model7(120., 0.2, -0.390078, 0.338796, 0.174814);
        /*
        EuropeanOptionPrice<1> a(OptionType::Call, model6.get_pointer(), 0.0367, 1., 100., 8, 100);
        EuropeanOptionLogPrice<1> b(OptionType::Call, model6.get_pointer(), 0.0367, 1., 100., 8, 100);
        */
        /*
        EuropeanOptionPrice<2> c
        (OptionType::Call, model4.get_pointer(), model5.get_pointer(),
         -0.2, 0.1, 1., 200., 6, 100);
        
        EuropeanOptionLogPrice<2> d
        (OptionType::Call, model4.get_pointer(), model5.get_pointer(),
         -0.2, 0.1, 1., 200., 6, 100);
		*/
// 		a.set_refine_status(true);
// 		b.set_refine_status(true);
// 		c.set_refine_status(true);		
// 		d.set_refine_status(true);
		
//         a.run();
//         b.run();
//         c.run();
//         d.run();
		
//         cout<<a.get_price()<<"\n";
//         cout<<b.get_price()<<"\n";
//         cout<<c.get_price()<<"\n";
//         cout<<d.get_price()<<"\n";
        
	const int top=7-3+1;
	double T[2][top], result[2][top], real_T[2][top];

	clock_t inizio,fine;
	struct timeval start, end;

	for (int i=0; i<top; i++) {
        
        {
			EuropeanOptionPrice<2> c
			(OptionType::Call, model4.get_pointer(), model5.get_pointer(),
	   -0.2, 0.1, 1., 200., i+3, 50);
	   
			gettimeofday(&start, NULL);
			inizio=clock();
			c.run();
			gettimeofday(&end, NULL);
			fine=clock();

			result[0][i]=c.get_price();

			T[0][i]=static_cast<double> (((fine-inizio)*1.e6)/CLOCKS_PER_SEC);
			real_T[0][i]=((end.tv_sec  - start.tv_sec) * 1000000u + 
	   end.tv_usec - start.tv_usec);
	   
        }
        
        {
		 EuropeanOptionLogPrice<2> d
	 (OptionType::Call, model4.get_pointer(), model5.get_pointer(),
	  -0.2, 0.1, 1., 200., i+3, 50);
        
			gettimeofday(&start, NULL);
			inizio=clock();
			d.run();
			gettimeofday(&end, NULL);
			fine=clock();

			result[1][i]=d.get_price();

			T[1][i]=static_cast<double> (((fine-inizio)*1.e6)/CLOCKS_PER_SEC);
			real_T[1][i]=((end.tv_sec  - start.tv_sec) * 1000000u + 
	   end.tv_usec - start.tv_usec);
			
        }
        
        
		 }
	ofstream out("TimesOldWay"); 
	out<<"Results for 100 time iterations:\n";
	for (int i=0; i<top; ++i) {
	 out<<"Price Grid\t"<<pow(2,i+3)<<"\tPrice\t"<<result[0][i]<<"\tclocktime\t"<<
	 T[0][i]/1e6<<"s\trealtime\t"<<real_T[0][i]/1e6<<"s\n";
	 out<<"LogPrice Grid\t"<<pow(2,i+3)<<"\tPrice\t"<<result[1][i]<<"\tclocktime\t"<<
	 T[1][i]/1e6<<"s\trealtime\t"<<real_T[1][i]/1e6<<"s\n";
   }
		 
        
        
        /*
        EuropeanOptionPrice<1> foo
        (OptionType::Put, model.get_pointer(), 0.0367, 1., 90., 12, 250);
        EuropeanOptionLogPrice<1> foo2
        (OptionType::Put, model.get_pointer(), 0.0367, 1., 90., 12, 250);
        
        EuropeanOptionPrice<1> duffy
        (OptionType::Call, model3.get_pointer(), 0.0367, 1., 90., 8, 100);
        EuropeanOptionLogPrice<1> duffy2
        (OptionType::Call, model3.get_pointer(), 0.0367, 1., 90., 8, 100);
        
        AmericanOptionPrice<1> minnie
        (model.get_pointer(), 0.0367, 1., 90., 12, 100);
        AmericanOptionLogPrice<1> minnie2
        (model.get_pointer(), 0.0367, 1., 90., 12, 100);
        
        EuropeanOptionPrice<2> mickey
        (OptionType::Call, model1.get_pointer(), model2.get_pointer(),
         -0.2, 0.1, 1., 200., 7, 100);
        
        EuropeanOptionLogPrice<2> mickey2
        (OptionType::Call, model1.get_pointer(), model2.get_pointer(),
         -0.2, 0.1, 1., 200., 7, 100);
        
        EuropeanOptionPrice<2> goofy
        (OptionType::Call, model5.get_pointer(), model4.get_pointer(),
         -0.2, 0.1, 1., 200., 6, 25);
        
        EuropeanOptionLogPrice<2> goofy2
        (OptionType::Call, model5.get_pointer(), model4.get_pointer(),
         -0.2, 0.1, 1., 200., 6, 25);
        
        foo.run();
        foo2.run();
        duffy.run();
        duffy2.run();
        minnie.run();
        minnie2.run();
        mickey.run();
        mickey2.run();
        goofy.run();
        goofy2.run();
        
        cout<<foo.get_price()<<"\n";
        cout<<foo2.get_price()<<"\n";
        cout<<duffy.get_price()<<"\n";
        cout<<duffy2.get_price()<<"\n";
        cout<<minnie.get_price()<<"\n";
        cout<<minnie2.get_price()<<"\n";
        cout<<mickey.get_price()<<"\n";
        cout<<mickey2.get_price()<<"\n";
        cout<<goofy.get_price()<<"\n";
        cout<<goofy2.get_price()<<"\n";
         */
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


void timestamp ( )

//****************************************************************************80
//
//  Purpose:
//
//    TIMESTAMP prints the current YMDHMS date as a time stamp.
//
//  Example:
//
//    31 May 2001 09:45:54 AM
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    08 July 2009
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    None
//
{
# define TIME_SIZE 40

	static char time_buffer[TIME_SIZE];
	const struct std::tm *tm_ptr;
	size_t len;
	std::time_t now;

	now = std::time ( NULL );
	tm_ptr = std::localtime ( &now );

	len = std::strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm_ptr );

	std::cout << time_buffer << "\n";

	return;
# undef TIME_SIZE
}