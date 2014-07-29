#ifndef __integral_levy_base_hpp
# define __integral_levy_base_hpp

# include "deal_ii.hpp"
# include "Quadrature.hpp"
//# include "Densities.hpp"
# include "tools.hpp"
# include "models.hpp"
# include "constants.hpp"
# include "cmath"

//TODO
//constructor
//inherit rest 

//! Abstract class 
/*!
 * This class gives the base for the classes calculating the integral part of the equation. It implements the function that calculates the fist part for a generic model.
 */

template<unsigned dim>
class LevyIntegralBase{
        
protected:
	
	std::vector<double> alpha;
	dealii::Vector<double> J1, J2;
	
	
	dealii::Point<dim> Bmin, Bmax, lower_limit, upper_limit;
	
	//should be const? 
	std::vector<Model *> Mods;
	
	bool alpha_ran;
	bool j_ran;
	
	virtual void compute_alpha();
	virtual void compute_Bounds();
	
public:
        
	LevyIntegralBase()=delete;
	LevyIntegralBase(dealii::Point<dim> lower_limit_,  dealii::Point<dim> upper_limit_,  std::vector<Model *> & Models_): lower_limit(lower_limit_), upper_limit(upper_limit_), Mods(Models_),  alpha_ran(false) , j_ran(false) {if (Models_.size() !=dim)
                std::cerr<< "Wrong dimension! Number of models is different from option dimension\n";
                this->compute_Bounds();};
	
	//would be nice to make it protected,  but need to pass arguments
	virtual void compute_J(dealii::Vector<double> & sol, dealii::DoFHandler<dim> & dof_handler, dealii::FE_Q<dim> & fe) =0;
	
	virtual double get_alpha_1() {
                if (!alpha_ran)
                        this->compute_alpha();
                return alpha[0];
        }
	
	virtual double get_alpha_2() {
                if (!alpha_ran)                      // add exception if dim < 2
                        this->compute_alpha();
                return alpha[1];
        }
        
	virtual void get_alpha(std::vector<double> & alp) {
                if (!alpha_ran)
                        this->compute_alpha();
                alp=alpha;
                return;
        }
	
	//add exception
	virtual void get_j_1(dealii::Vector<double> * &J_x) {
                if (!j_ran)
                        std::cerr<< "Run J first!"<< std::endl;
                else {
                        J_x=&J1;
                        j_ran=false;
                }
                return;
        }
	virtual void get_j_both(dealii::Vector<double> * &J_x, dealii::Vector<double> * &J_y) {
                if (!j_ran)
                        std::cerr<< "Run J first!"<< std::endl;
                else{
                        J_x=&J1;
                        J_y=&J2;
                        j_ran=false;
                }
                return;
        }
	
	virtual ~LevyIntegralBase() {for (unsigned d=0;d<dim;++d) Mods[d]=nullptr;}
};

template<unsigned dim>
void LevyIntegralBase<dim>::compute_Bounds() {
	for (unsigned d=0;d<dim;++d) {
                //may misbehave with merton
                Bmin[d]=std::min(0., lower_limit(d));
                Bmax[d]=upper_limit(d);
                while ((*Mods[d]).density(Bmin[d])>constants::light_toll)
                        Bmin[d]+=-0.5;
                while ((*Mods[d]).density(Bmax[d])>constants::light_toll)
                        Bmax[d]+=0.5;
                
        }
}

template<unsigned dim>
void LevyIntegralBase<dim>::compute_alpha()
{
	using namespace dealii;
	alpha=std::vector<double>(dim, 0.);
        for (unsigned d=0;d<dim;++d) {
                
                
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
                                p[d]=quad_points_1D[q_point][0];
                                alpha[d]+=fe_values.JxW(q_point)*(exp(p[d])-1.)*(*Mods[d]).density(p[d]);
                        }
                }
        }
	alpha_ran=true;
}





#endif