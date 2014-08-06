#ifndef __tools_hpp
#define __tools_hpp

#include "deal_ii.hpp"
#include "models.hpp"
//! This namespace contains tools that are used in the library
/*!
 * Here are gathered the functions and auxiliary classes that should not be members the classes in the library.
 */
namespace tools {
        
        
        //! An iterator used to cycle on verices of a triangulation
        /*!
         * This class allows to cycle on the vertices of the triangulation, and allows to obtain both the point and the global index of that point
         */
        template<int dim>
        class Vertex_Iterator {
                
        private:
                typename dealii::DoFHandler<dim>::active_cell_iterator _cells;
                std::vector<bool> _used;
                unsigned _vert_index;
                unsigned _ndofs_cell;
                unsigned _counted;
                
        public:
                //! Default constructor deleted
                Vertex_Iterator()=delete;
                //! Constructor from a DoFHandler
                /*!
                 * Construct the iterator from a DoFHandler connected to a triangulation. This creates a Vertex_Iterator and initializes it at the first vertex of the first cell of the DoFHandler.
                 */
                Vertex_Iterator(dealii::DoFHandler<dim> const & dof): _used(dof.n_dofs(), false), _vert_index(0),  _counted(1) {
                        _ndofs_cell=dof.get_fe().dofs_per_cell;
                        _cells=dof.begin_active();
                }
                
                //! Returns the global index of the current vertex
                dealii::types::global_dof_index get_global_index();
                
                //! Returns True if all vertices have been visited
                bool at_end();
                
                //! Comparison operator
                bool operator== (Vertex_Iterator<dim> const & rhs);
                //! Not equal operator
                bool operator!= (Vertex_Iterator<dim> const & rhs);
                
                //!Forward advance operator
                /*!
                 * Advance the iterator to a new vertex. If all vertices have been visited, it (throw exception or do nothing? ).
                 */
                Vertex_Iterator & operator++ ();
                //!Dereferencing operator
                /*!
                 * Derefences the interator returning the point corresponding to the vertex
                 */
                dealii::Point<dim> & operator* () { return _cells->vertex(_vert_index);}
                
        };
        
        
        template<int dim>
        dealii::types::global_dof_index Vertex_Iterator<dim>::get_global_index() {
                std::vector<dealii::types::global_dof_index> local_ind(_ndofs_cell);
                _cells->get_dof_indices(local_ind);
                return local_ind[_vert_index];
        }
        
        template<int dim>
        Vertex_Iterator<dim> & Vertex_Iterator<dim>::operator++() {
                if (_counted<_used.size())
                {
                        do{
                                if (_vert_index<_ndofs_cell-1)
                                        _vert_index++;
                                else {
                                        ++_cells;
                                        _vert_index=0;
                                }
                                
                        } while (_used[get_global_index()]);
                        _used[get_global_index()]=true;
                        
                }
                _counted++;
                
                return * this;
        }
        
        template<int dim>
        bool Vertex_Iterator<dim>::operator== (Vertex_Iterator<dim> const & rhs) {
                return (_cells==rhs._cells && _vert_index==rhs._vert_index);
        }
        
        
        template<int dim>
        bool Vertex_Iterator<dim>::operator != (Vertex_Iterator<dim> const & rhs) {
                return (_cells !=rhs._cells or _vert_index !=rhs._vert_index);
        }
        
        template<int dim>
        bool Vertex_Iterator<dim>::at_end() {
                return _counted>_used.size();
        }
        
        
        //TODO delete default constructor
        //! A small class that implements the extension of a function when out of domain
        /*!
         * This class has a two goals. On one side, uses the boundary conditions to give a value to the function outside its domain. Secondly,  it uses DealII tools to obtain the value of the solution if the point where it is needed is not one of the degrees of freedom.
         */
        template<int dim>
        class Solution_Trimmer: public dealii::Function<dim>
        {
        protected:
                unsigned int _ax;
                //check that this causes no memory leaks while keeping hereditariety
                dealii::Function<dim> & _BC;  
                dealii::DoFHandler<dim> const & _dof;
                dealii::Vector<double> const & _sol;
                dealii::Point<dim> _l_lim, _r_lim;
                dealii::Functions::FEFieldFunction<dim> _fe_func;
                
        public:
                Solution_Trimmer(unsigned int ax,   dealii::Function<dim> & BC, dealii::DoFHandler<dim> const & dof, dealii::Vector<double> const & sol,  dealii::Point<dim> const & xmin, dealii::Point<dim> const & xmax): _ax(ax), _BC(BC),  _dof(dof), _sol(sol), _l_lim(xmin), _r_lim(xmax) , _fe_func(_dof, _sol){};
                
                virtual double value(const dealii::Point<dim> &p,  const unsigned int component=0) const;
                virtual void value_list(const std::vector<dealii::Point<dim> > &points,
                                        std::vector<double> &values,
                                        const unsigned int component = 0) const;
        };
        
        template<int dim>
        double Solution_Trimmer<dim>::value(const dealii::Point<dim> &p,  const unsigned int component) const
        {
                using namespace dealii;
                Assert (component == 0, ExcInternalError());
                
                if (p[_ax]<_l_lim[_ax])
                        return _BC.value(p);
                if (p[_ax]>_r_lim[_ax])
                        return _BC.value(p);
                return _fe_func.value(p);  
                
        }
        
        template<int dim>
        void Solution_Trimmer<dim>::value_list(const std::vector<dealii::Point<dim> > &points, std::vector<double> &values, const unsigned int component) const
        {
                using namespace dealii;
                Assert (values.size() == points.size(),
                        ExcDimensionMismatch (values.size(), points.size()));
                Assert (component == 0, ExcInternalError());
                
                const unsigned int n_points=points.size();
                
                for (unsigned int i=0;i<n_points;++i)
                {
                        if (points[i][_ax]<_l_lim[_ax]) {
                                values[i]=_BC.value(points[i]);
                        }
                        else if (points[i][_ax]>_r_lim[_ax]) {
                                values[i]=_BC.value(points[i]);
                        }
                        else
                                values[i]=_fe_func.value(points[i]);
                }
        }
       
        ////////////the next functions are helpers used in the assembling of mass matrix//////////////////
        
		//! Fills the tensor trasp_ with the coefficients for Price transformation
		/*!
         * Creates the value of the tensor t needed when calculating \f$\int_{Q_k} \phi_i \mathbf{t}^T \nabla\phi_j\f$ on the cell \f$Q_k\f$. Works with price transformation
         */
        template<int dim>
		void make_trasp(dealii::Tensor< 1 , dim, double > & trasp_, std::vector<Model *> & Models_,  double r_, double rho_,  std::vector<double> alpha_,  dealii::Point<dim> const & qpt_) {
			//TODO add exception
			std::cerr<< "Not defined in this dimension\n";
		}

		//! Specialization of make_trasp<dim>() for 1 dimension
		template<>
		void make_trasp<1>(dealii::Tensor< 1 , 1, double > & trasp_, std::vector<Model *> & Models_,  double r_, double rho_, std::vector<double> alpha_, dealii::Point<1> const & qpt_) {
		  trasp_[0]=(r_-(Models_[0])->get_vol()*(Models_[0])->get_vol()-alpha_[0])*qpt_(0);
		}
		

		//! Specialization of make_trasp<dim>() for 2 dimensions
		template<>
		void make_trasp<2>(dealii::Tensor< 1 , 2, double > & trasp_, std::vector<Model *> & Models_,  double r_, double rho_, std::vector<double> alpha_,  dealii::Point<2> const & qpt_) {
		  trasp_[0]=(r_-alpha_[0]-(Models_[0])->get_vol()*(Models_[0])->get_vol()-0.5*rho_*(Models_[0])->get_vol()*(Models_[1])->get_vol())*qpt_(0);
		  trasp_[1]=(r_-alpha_[1]-(Models_[1])->get_vol()*(Models_[1])->get_vol()-0.5*rho_*(Models_[0])->get_vol()*(Models_[1])->get_vol())*qpt_(1);
		}
		//!Fills the tensor diff_ with the coefficients for Price transformation
		/*!
		 * Creates the values of the tensor D needed when calculating \f$\int_{Q_k} \nabla\phi_i D \nabla\phi_j\f$ on the cell \f$Q_k\f$. Works with price transformation
		 */
		template<unsigned dim>
		void make_diff(dealii::Tensor< 2 , dim, double > & diff_,std::vector<Model *> & Models_, double rho_,  dealii::Point<dim> const & qpt_ ) {
			  //TODO add exception
			  std::cerr<< "Not defined in this dimension\n";
		}
		
		//! Specialization of make_diff<dim>() for 1 dimension
		template<>
		void make_diff<1>(dealii::Tensor<2, 1, double> & diff_, std::vector<Model *> & Models_, double rho_,  dealii::Point<1> const & qpt_ ) {
			diff_[0][0]=0.5*(Models_[0])->get_vol()*(Models_[0])->get_vol()*qpt_(0)*qpt_(0);
		}

		//! Specialization of make_diff<dim>() for 2 dimensions
		template<>
		void make_diff<2>(dealii::Tensor< 2 , 2, double > & diff_,std::vector<Model *> & Models_, double rho_,  dealii::Point<2> const & qpt_ ) {
		  diff_[0][0]=0.5*(Models_[0])->get_vol()*(Models_[0])->get_vol()*qpt_(0)*qpt_(0);
		  diff_[1][1]=0.5*(Models_[1])->get_vol()*(Models_[1])->get_vol()*qpt_(1)*qpt_(1);
		  diff_[0][1]=0.5*rho_*(Models_[0])->get_vol()*(Models_[1])->get_vol()*qpt_(0)*qpt_(1);
		  diff_[1][0]=diff_[0][1];
		}

}
#endif