#ifndef __integral_levy_hpp
#define __integral_levy_hpp

#include "deal_ii.hpp"
#include "Quadrature.hpp"
#include "Densities.hpp"

template <unsigned dim>
class LevyIntegral {
};

template <>
class LevyIntegral<1> {
protected:
        dealii::Function<1> * density;
        Point<1>              Smin, Smax;
        double                  toll;
public:
        LevyIntegral():density(NULL){};
        LevyIntegral(dealii::Function<1> * density_,
                     Point<1> Smin_,
                     Point<1> Smax_,
                     double toll_=constants::light_toll)
        :
        density(density_),
        Smin(Smin_),
        Smax(Smax_),
        toll(toll_)
        {};
        
        virtual ~LevyIntegral() {
                delete density;
        };
        
        virtual double  get_part1();
        virtual void    get_part2(dealii::Vector<double> &J,
                                  dealii::Vector<double> const &solution,
                                  dealii::FE_Q<1> const &fe,
                                  dealii::DoFHandler<1> const &dof_handler);
};

// WRONG! Just to compile it!
template <>
class LevyIntegral<2> {
protected:
        dealii::Function<1> * density;
        Point<2>              Smin, Smax;
        double                toll;
public:
        LevyIntegral():density(NULL){};
        
        LevyIntegral(dealii::Function<1> * density_,
                     Point<2> Smin_,
                     Point<2> Smax_,
                     double toll_=constants::light_toll){};
        
        virtual ~LevyIntegral() {
                delete density;
        };
        
        virtual double  get_part1(){ return 0.; };
        virtual void    get_part2(dealii::Vector<double> &J,
                                  dealii::Vector<double> const &solution,
                                  dealii::FE_Q<2> const &fe,
                                  dealii::DoFHandler<2> const &dof_handler){};
        
};

// classe Kou -> calcola alpha, calcola J

template <unsigned dim>
class KouIntegral: public LevyIntegral<dim> {
        
};

template <>
class KouIntegral<1>: public LevyIntegral<1> {
private:
        
        std::vector<KouModel *> models;
        
        bool adapting;
        
        Quadrature_Laguerre right_quad, left_quad;
        
        std::vector<double> right_quad_nodes, left_quad_nodes;
        std::vector<double> right_quad_weights, left_quad_weights;
        
        std::vector<Point<1> > quadrature_points;
        
        void setup_quadrature_rigth(unsigned n);
        void setup_quadrature_left(unsigned n);
        void setup_quadrature_points();
        
public:
        
        // 1d constructor
        KouIntegral(KouModel * const p,
                    dealii::Point<1> Smin_,
                    dealii::Point<1> Smax_,
                    bool adapting_=true,
                    double toll_=constants::light_toll);
        
        virtual double get_part1();
        
};

template <>
class KouIntegral<2>: public LevyIntegral<2> {
public:
        KouIntegral(KouModel * const p,
                    dealii::Point<2> Smin_,
                    dealii::Point<2> Smax_,
                    bool adapting_=true,
                    double toll_=constants::light_toll){};
        
};

// classe Merton -> calcola alpha, calcola J

template <unsigned dim>
class MertonIntegral: public LevyIntegral<dim> {
        
};

// LevyIntegral -> implementazione
// template <>
double LevyIntegral<1>::get_part1() {
        
        using namespace dealii;
        
        double alpha(0.);
        
	Point<1> Bmin(0.), Bmax(Smax);
	double step(0.5);
	
	while ((*density).value(Bmin)>toll)
                Bmin[0]+=-step;
	
	while ((*density).value(Bmax)>toll)
                Bmin[0]+=step;
        
	Triangulation<1> integral_grid;
	FE_Q<1> fe(1);
	DoFHandler<1> dof_handler(integral_grid);
	
	GridGenerator::hyper_cube<1>(integral_grid, Bmin[0], Bmax[0]);
	integral_grid.refine_global(15);
	
	dof_handler.distribute_dofs(fe);
	
	QGauss<1> quadrature_formula(8);
	FEValues<1> fe_values(fe, quadrature_formula,  update_quadrature_points | update_JxW_values);
        
	typename DoFHandler<1>::active_cell_iterator
	cell=dof_handler.begin_active(),
	endc=dof_handler.end();
        
	const unsigned int n_q_points(quadrature_formula.size());
	
	for (; cell !=endc;++cell) {
                fe_values.reinit(cell);
                std::vector< Point<1> > quad_points(fe_values.get_quadrature_points());
                for (unsigned q_point=0;q_point<n_q_points;++q_point)
                        alpha+=fe_values.JxW(q_point)*(exp(quad_points[q_point][0])-1.)*(*density).value(quad_points[q_point]);
                
	}
	cout<< "alpha is "<< alpha<<std::endl;
        
	return alpha;
        
}

// template <>
void LevyIntegral<1>::get_part2(dealii::Vector<double> &J,
                                dealii::Vector<double> const &solution,
                                dealii::FE_Q<1> const &fe,
                                dealii::DoFHandler<1> const &dof_handler) {
        
        using namespace dealii;
        
        J.reinit(solution.size());
	
	QGauss<1> quadrature_formula(5);
	FEValues<1> fe_values(fe, quadrature_formula,  update_quadrature_points | update_values | update_JxW_values);
	
	const unsigned int n_q_points(quadrature_formula.size());
	
	typename DoFHandler<1>::active_cell_iterator
        cell=dof_handler.begin_active(),
        endc=dof_handler.end();
	
	vector<double> sol_cell(n_q_points);
	const unsigned int   dofs_per_cell = fe.dofs_per_cell;
	
	vector< Point <1> > quad_points(n_q_points);
	Point<1> logz(0.);
	vector<bool> used(solution.size(), false);
	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        
        typename DoFHandler<1>::active_cell_iterator outer_cell=dof_handler.begin_active();
        
        for (;outer_cell !=endc;++outer_cell){
                
                outer_cell->get_dof_indices(local_dof_indices);
                
                for (unsigned int j=0;j<dofs_per_cell;++j) {
                        
                        unsigned iter=local_dof_indices[j];
                        
                        if (used[iter]==false) {
				used[iter]=true;
				Point<1> actual_vertex=outer_cell->vertex(j);
				
                                cell=dof_handler.begin_active();
                                
                                for (; cell!=endc;++cell) {
                                        fe_values.reinit(cell);
                                        quad_points=fe_values.get_quadrature_points();
                                        fe_values.get_function_values(solution, sol_cell);                        
                                        for (unsigned q_point=0;q_point<n_q_points;++q_point) {
                                                logz(0)=log(quad_points[q_point](0)/actual_vertex(0));
                                                J[iter]+=fe_values.JxW(q_point)*sol_cell[q_point]*(*density).value(logz)/quad_points[q_point](0);
                                        }
                                }
                        }
                        
                }
	}
        
        return;
}

// KouIntegral implementation

// template <>
void KouIntegral<1>::setup_quadrature_rigth(unsigned n){
        right_quad=Quadrature_Laguerre(n, models[0]->get_lambda_p());
        right_quad_nodes=right_quad.get_nodes();
        right_quad_weights=right_quad.get_weights();
        return;
}

// template <>
void KouIntegral<1>::setup_quadrature_left(unsigned n){
        left_quad =Quadrature_Laguerre(n, models[0]->get_lambda_m());
        left_quad_nodes=left_quad.get_nodes();
        left_quad_weights=left_quad.get_weights();
        return;
}

// template <>
void KouIntegral<1>::setup_quadrature_points(){
        quadrature_points=std::vector<Point<1> > (left_quad.get_order()+right_quad.get_order());
        // Building one vector for nodes and weights
        for (unsigned i=0; i<left_quad.get_order(); ++i) {
                quadrature_points[i]=(Point<1>(-left_quad_nodes[i]));
        }
        for (unsigned i=0; i<right_quad.get_order(); ++i) {
                quadrature_points[i+left_quad.get_order()]=(Point<1>(right_quad_nodes[i]));
        }
        return;
}

// 1d constructor
// template <>
KouIntegral<1>::KouIntegral(KouModel * const p,
                            dealii::Point<1> Smin_,
                            dealii::Point<1> Smax_,
                            bool adapting_,
                            double toll_)
:
LevyIntegral<1>(new Kou_Density<1>(p->get_p(),
                                   p->get_lambda(),
                                   p->get_lambda_p(),
                                   p->get_lambda_m()),
                Smin_,
                Smax_,
                toll_),
adapting(adapting_)
{
        models.push_back(p);
        
        // Building Quadrature_Laguerre objects
        if (!adapting) {
                setup_quadrature_rigth(16);
                setup_quadrature_left(16);
        }
        else {
                setup_quadrature_rigth(2);
                setup_quadrature_left(2);
        }
        setup_quadrature_points();
        
}
/*
// 2d constructor (to do)
template <unsigned dim>
KouIntegral<dim>::KouIntegral(KouModel * const p,
                            dealii::Point<dim> Smin_,
                            dealii::Point<dim> Smax_,
                            bool adapting_,
                            double toll_)
{}

template <unsigned dim>
double KouIntegral<dim>::get_part1(){ return 0; }

template <unsigned dim>
void KouIntegral<dim>::setup_quadrature_rigth(unsigned n){}

template <unsigned dim>
void KouIntegral<dim>::setup_quadrature_left(unsigned n){}

template <unsigned dim>
void KouIntegral<dim>::setup_quadrature_points(){}
*/
// template <>
double KouIntegral<1>::get_part1()
{
        double alpha=0.;
        
        if (!adapting) {
                for (unsigned i=0; i<right_quad.get_order(); ++i) {
                        alpha+=(exp(right_quad_nodes[i])-1)*
                        (models[0]->get_p())*(models[0]->get_lambda())*
                        (models[0]->get_lambda_p())*right_quad_weights[i];
                }
                
                for (unsigned i=0; i<left_quad.get_order(); ++i) {
                        alpha+=(exp(-left_quad_nodes[i])-1)*
                        (1-(models[0]->get_p()))*(models[0]->get_lambda())*
                        (models[0]->get_lambda_m())*left_quad_weights[i];
                }
                
        }
        
        else {
                unsigned order_max=64;
                
                double alpha_old=0.;
                
                do  {
                        alpha_old=alpha;
                        alpha=0.;
                        
                        for (unsigned i=0; i<right_quad.get_order(); ++i) {
                                alpha+=(exp(right_quad_nodes[i])-1)*
                                (models[0]->get_p())*(models[0]->get_lambda())*
                                (models[0]->get_lambda_p())*right_quad_weights[i];
                        }
                        
                        for (unsigned i=0; i<left_quad.get_order(); ++i) {
                                alpha+=(exp(-left_quad_nodes[i])-1)*
                                (1-(models[0]->get_p()))*(models[0]->get_lambda())*
                                (models[0]->get_lambda_m())*left_quad_weights[i];
                        }
                        
                        setup_quadrature_rigth(2*right_quad.get_order());
                        setup_quadrature_left(2*left_quad.get_order());
                        setup_quadrature_points();
                        
                }
                while (abs(alpha_old-alpha)>toll &&
                       right_quad.get_order()<=order_max &&
                       left_quad.get_order()<=order_max);
        }
        
        return alpha;
        
}

#endif