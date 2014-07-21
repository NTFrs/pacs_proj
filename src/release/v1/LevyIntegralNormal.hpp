#ifndef __integral_levy_normal_hpp
# define __integral_levy_normal_hpp

# include "deal_ii.hpp"
# include "Quadrature.hpp"
# include "Densities.hpp"
# include "tools.hpp"
#include "models.hpp"

//TODO
//constructor
//inherit rest 

template<unsigned dim>
class LevyIntegral<dim> {

protected:
	std::vector<double> alpha;
	dealii::Vector<double> J1, J2;
	dealii::Point<dim> Bmin, Bmax, Smin, Smax;
	
	std::vector<Model *> Mods;
	
	bool alpha_ran;
	
	virtual void compute_alpha();
	virtual void compute_J(/* */) =0;

public:
	
	LevyIntegral()=delete;
	
	virtual double get_alpha_1() const {if (!alpha_ran)
							compute_alpha();
							return alpha[0];}
	
	virtual double get_alpha_2() const {if (!alpha_ran)                      // add exception if dim < 2
							 compute_alpha();
							 return alpha[1];}

	virtual void get_alpha(std::vector<double> & alp) const {if (!alpha_ran)
														  compute_alpha();
															alp=alpha;
														 return;}

	virtual ~LevyIntegral() {for (unsiged d=0;d<dim;++d) Mods[d]=nullptr;}
};

template<unsigned dim>
void LevyIntegral< dim >::compute_alpha()
{
for (unsigned d=0;d<dim;++d) {
	Bmin[d]=0.;Bmax[d]=Smax[d];
	while ((*Mods[d]).density(Bmin[d])>tol)
		Bmin[d]+=-0.5;
	while ((*Mods[d]).density(Bmax[d])>tol)
		Bmax[d]+=0.5;
	
	
	 Triangulation<1> integral_grid;
	 FE_Q<1> fe2(1);
	 DoFHandler<1> dof_handler2(integral_grid);

	 GridGenerator::hyper_cube<1>(integral_grid, Bmin[d], Bmax[d]);
	 integral_grid.refine_global(15);

	 dof_handler2.distribute_dofs(fe2);

	 QGauss<1> quadrature_formula(8);
	 FEValues<1> fe_values(fe2, quadrature_formula,  update_quadrature_points |update_JxW_values);

	 typename DoFHandler<1>::active_cell_iterator
	 cell=dof_handler2.begin_active(),
	 endc=dof_handler2.end();

	 const unsigned int n_q_points(quadrature_formula.size());

	 for (; cell !=endc;++cell) {
	  fe_values.reinit(cell);
	  std::vector< Point<1> > quad_points_1D(fe_values.get_quadrature_points());

	  for (unsigned q_point=0;q_point<n_q_points;++q_point) {
	   Point<dim> p;
	   p[d]=quad_points_1D[q_point][d];
	   alpha[d]+=fe_values.JxW(q_point)*(exp(p[d])-1.)*(*Mods[d]).density(p[d]);
	 }
	}

	 std::cout<< "alpha "<< d << " is "<< alpha[d]<< std::endl;
 }
	
}

template<unsigned dim>
class LevyIntegralPrice: public: LevyIntegral< dim > {

protected:
	//TODO add exception
	virtual void compute_J(/**/) {std::cerr<<"Not defined for this dimension"<< std::endl;}

public:
	LevyIntegralPrice()=delete
};

template<>
LevyIntegralPrice<1>::LevyIntegralPrice(Point<1> Smin_,  Point<1> Smax_,  Model * Mod_)

#endif