#ifndef __integral_levy_hpp
#define __integral_levy_hpp

#include "deal_ii.hpp"
#include "QuadratureRule.hpp"
#include "Densities.hpp"

template <unsigned dim>
class LevyIntegral {
protected:
        dealii::Function<dim> * density;
        Point<dim>              Smin, Smax;
        double                  toll;
public:
        LevyIntegral(dealii::Function<dim> * density_,
                     Point<dim> Smin_,
                     Point<dim> Smax_,
                     double toll_=constants::light_toll)
        :
        density(density_),
        Smin(Smin_),
        Smax(Smax_),
        toll(toll_)
        {};
        
        virtual double  get_part1();
        virtual void    get_part2(dealii::Vector<double> &J,
                                  dealii::Vector<double> const &solution,
                                  dealii::FE_Q<dim> const &fe,
                                  dealii::DoFHandler<dim> const &dof_handler);
};

// classe Kou -> calcola alpha, calcola J

template <unsigned dim>
class KouIntegral: public LevyIntegral<dim> {
        
};

// classe Merton -> calcola alpha, calcola J

template <unsigned dim>
class MertonIntegral: public LevyIntegral<dim> {
        
};

template <unsigned dim>
double LevyIntegral<dim>::get_part1() {
        
        using namespace dealii;
        
        double alpha(0.);
        
	Point<dim> Bmin(0.), Bmax(Smax);
	double step(0.5);
	
	while ((*density).value(Bmin)>toll)
                Bmin[0]+=-step;
	
	while ((*density).value(Bmax)>toll)
                Bmin[0]+=step;
        
	Triangulation<dim> integral_grid;
	FE_Q<dim> fe(1);
	DoFHandler<dim> dof_handler(integral_grid);
	
	GridGenerator::hyper_cube<dim>(integral_grid, Bmin[0], Bmax[0]);
	integral_grid.refine_global(15);
	
	dof_handler.distribute_dofs(fe);
	
	QGauss<dim> quadrature_formula(8);
	FEValues<dim> fe_values(fe, quadrature_formula,  update_quadrature_points | update_JxW_values);
        
	typename DoFHandler<dim>::active_cell_iterator
	cell=dof_handler.begin_active(),
	endc=dof_handler.end();
        
	const unsigned int n_q_points(quadrature_formula.size());
	
	for (; cell !=endc;++cell) {
                fe_values.reinit(cell);
                std::vector< Point<dim> > quad_points(fe_values.get_quadrature_points());
                for (unsigned q_point=0;q_point<n_q_points;++q_point)
                        alpha+=fe_values.JxW(q_point)*(exp(quad_points[q_point][0])-1.)*(*density).value(quad_points[q_point]);
                
	}
	cout<< "alpha is "<< alpha<<std::endl;
        
	return alpha;
        
}

template <unsigned dim>
void LevyIntegral<dim>::get_part2(dealii::Vector<double> &J,
                                  dealii::Vector<double> const &solution,
                                  dealii::FE_Q<dim> const &fe,
                                  dealii::DoFHandler<dim> const &dof_handler) {
        
        using namespace dealii;
        
        J.reinit(solution.size());
	unsigned N(solution.size());
        
        std::vector<Point<dim> > grid_points(N, 0.);
	
	QGauss<dim> quadrature_formula(5);
        
	FEValues<dim> fe_values(fe, quadrature_formula,  update_quadrature_points | update_values | update_JxW_values);
	
	const unsigned n_q_points(quadrature_formula.size());
	
	typename DoFHandler<dim>::active_cell_iterator
        cell=dof_handler.begin_active(),
        endc=dof_handler.end();
	
	vector<double> sol_cell(n_q_points);
	
	vector< Point <dim> > quad_points(n_q_points);
	Point<dim> logz(0.);
        
	for(unsigned int iter=0;iter<N;++iter) {
                
                cell=dof_handler.begin_active();
                for (; cell!=endc;++cell) {
                        
                        fe_values.reinit(cell);
                        quad_points=fe_values.get_quadrature_points();
                        fe_values.get_function_values(solution, sol_cell);
                        double a, b, c, d;
                        
                        for (unsigned q_point=0;q_point<n_q_points;++q_point) {
                                logz(0)=log(quad_points[q_point](0)/grid_points[iter](0));
                                a=fe_values.JxW(q_point);
                                b=sol_cell[q_point];
                                c=(*density).value(logz);
                                d=quad_points[q_point](0);
                                J[iter]+=a*b*c/d;
                        }
                        
                }
        }
        
        return;
        
}

#endif