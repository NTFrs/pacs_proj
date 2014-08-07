#ifndef __tools_hpp
#define __tools_hpp

#include "deal_ii.hpp"
#include "models.hpp"
//! This namespace contains tools that are used in the library
/*!
 * Here are gathered the functions and auxiliary classes that should not be members the classes in the library.
 */
namespace tools {
        
        
        //TODO delete default constructor
        //! A small class that implements the extension of a function when out of domain
        /*!
         * This class (functor) has a two goals. On one side, uses the boundary conditions to give a value to the function outside its domain. Secondly,  it uses DealII tools to obtain the value of the solution if the point where it is needed is not one of the degrees of freedom.
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
				Solution_Trimmer()=delete;
				//! Only constructor of the class
				/*!
                 * Constructs the functor using the parameter passed. It thus needs a solution vector and the dof_handler associated,  the limits of the rectangular domain,  the function to be imposed when the point is out of domain and the ax (x,y..) on wich it works
				 * \param ax		On wich cartesian ax should the function work
				 * \param BC		Function to be imposed when out of the domain 
				 * \param dof 		DealII DoFHandler associated to the solution
				 * \param sol 		DealII Vector containing the solution
				 * \param xmin		DealII Point that marks the bottom left point of the domain
				 * \param xmax		DealII Point that marks the upper right point of the domain
                 */
                Solution_Trimmer(unsigned int ax,   dealii::Function<dim> & BC, dealii::DoFHandler<dim> const & dof, dealii::Vector<double> const & sol,  dealii::Point<dim> const & xmin, dealii::Point<dim> const & xmax): _ax(ax), _BC(BC),  _dof(dof), _sol(sol), _l_lim(xmin), _r_lim(xmax) , _fe_func(_dof, _sol){};
                
                //! Returns the value of the solution if inside the domain,  the boundary condition value otherwise
                /*!
                 * \param p			DealII Point where the solution should be evaluated
                 */
                virtual double value(const dealii::Point<dim> &p,  const unsigned int component=0) const;
                //! Returns a vector of trimmed solutions associated to the vector of points in input
                /*!
                 * Same as value but acts on a vector of points
				 * \param points	standard vector of DealII Points where the solution is to be evaluated
				 * \param values	standard vector where the values of the solution are stored upon exit
                 */
                virtual void value_list(const std::vector<dealii::Point<dim> > &points,
                                        std::vector<double> &values,
                                        const unsigned int component = 0) const;
        };
        
        template<int dim>
        double Solution_Trimmer<dim>::value(const dealii::Point<dim> &p,  const unsigned int component) const
        {
                using namespace dealii;
                Assert (component == 0, ExcInternalError());
                //if on the left applies BC
                if (p[_ax]<_l_lim[_ax])
                        return _BC.value(p);
                //same if on the right
                if (p[_ax]>_r_lim[_ax])
                        return _BC.value(p);
                //and if internal,  uses DealII function map to find the correct value
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
                //does the same that value does but on multiple points
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
		 * \param trasp_	DealII first order tensor. Output of the function
		 * \param Models_	std::vector of pointer to models used in the option
		 * \param r_		interest rate
		 * \param rho_		correlation between underlying assets (dummy in 1 dimension)
		 * \param alpha_	vector containing the alpha part of the integrals
		 * \param qpt_		point \f$ S_j \f$ where it should be calculated
         */
        template<int dim>
		void make_trasp(dealii::Tensor< 1 , dim, double > & trasp_, std::vector<Model *> & Models_,  double r_, double rho_,  std::vector<double> alpha_,  dealii::Point<dim> const & qpt_) {
			//TODO add exception
			std::cerr<< "Not defined in this dimension\n";
		}

		//! Specialization of make_trasp<dim>() for 1 dimension
		/*!
         * Returns the following one dimensional vector on the point \f$ S_j \f$
		 * \f[\begin{bmatrix}(r-\sigma^2-\alpha)S_j \end{bmatrix}\f]
         */
		template<>
		void make_trasp<1>(dealii::Tensor< 1 , 1, double > & trasp_, std::vector<Model *> & Models_,  double r_, double rho_, std::vector<double> alpha_, dealii::Point<1> const & qpt_) {
		  trasp_[0]=(r_-(Models_[0])->get_vol()*(Models_[0])->get_vol()-alpha_[0])*qpt_(0);
		}
		

		//! Specialization of make_trasp<dim>() for 2 dimensions
		/*!
		 * Returns the following vector on the point \f$ (S_1, S_2)_j \f$
		 * \f[\begin{bmatrix}\left(r-\sigma^2_1-\alpha_1 -\frac{1}{2}\rho\sigma_1\sigma_2\right)S_{1, j}\\
		 * \left(r-\sigma^2_2-\alpha_2 -\frac{1}{2}\rho\sigma_1\sigma_2\right)S_{2, j}\end{bmatrix}\f]
		 */
		template<>
		void make_trasp<2>(dealii::Tensor< 1 , 2, double > & trasp_, std::vector<Model *> & Models_,  double r_, double rho_, std::vector<double> alpha_,  dealii::Point<2> const & qpt_) {
		  trasp_[0]=(r_-alpha_[0]-(Models_[0])->get_vol()*(Models_[0])->get_vol()-0.5*rho_*(Models_[0])->get_vol()*(Models_[1])->get_vol())*qpt_(0);
		  trasp_[1]=(r_-alpha_[1]-(Models_[1])->get_vol()*(Models_[1])->get_vol()-0.5*rho_*(Models_[0])->get_vol()*(Models_[1])->get_vol())*qpt_(1);
		}
		//!Fills the tensor diff_ with the coefficients for Price transformation
		/*!
		 * Creates the values of the tensor D needed when calculating \f$\int_{Q_k} \nabla\phi_i D \nabla\phi_j\f$ on the cell \f$Q_k\f$. Works with price transformation,  but can be used with logprice by using a dummy point with the value one in each coordinate 
		 * \param diff_		DealII second order tensor. Output of the function
		 * \param Models_	std::vector of pointer to models used in the option
		 * \param rho_		correlation between underlying assets (dummy in 1 dimension)
		 * \param qpt_		point where it should be calculated
		 */
		template<unsigned dim>
		void make_diff(dealii::Tensor< 2 , dim, double > & diff_,std::vector<Model *> & Models_, double rho_,  dealii::Point<dim> const & qpt_ ) {
			  //TODO add exception
			  std::cerr<< "Not defined in this dimension\n";
		}
		
		//! Specialization of make_diff<dim>() for 1 dimension
		/*!
         * Returns the following matrix on the point \f$ S_j \f$
		 * \f[
		 * \begin{bmatrix}\frac{1}{2}\sigma^2 S_j\end{bmatrix} 
		 * \f]
         */
		template<>
		void make_diff<1>(dealii::Tensor<2, 1, double> & diff_, std::vector<Model *> & Models_, double rho_,  dealii::Point<1> const & qpt_ ) {
			diff_[0][0]=0.5*(Models_[0])->get_vol()*(Models_[0])->get_vol()*qpt_(0)*qpt_(0);
		}

		//! Specialization of make_diff<dim>() for 2 dimensions
		/*!
         * Returns the following matrix on the point \f$ (S_1, S_2)_j \f$
		 *\f[
		 \begin{bmatrix}
		 \frac{1}{2}\sigma_1^2 S_{1, j} & \frac{1}{2}\sigma_1\sigma_2 S_{1,j}S_{2,j} \\
		 \frac{1}{2}\sigma_1\sigma_2 S_{1,j}S_{2,j} & \frac{1}{2}\sigma_2^2 S_{2, j}
		 \end{bmatrix}
		 \f] 
         */
		template<>
		void make_diff<2>(dealii::Tensor< 2 , 2, double > & diff_,std::vector<Model *> & Models_, double rho_,  dealii::Point<2> const & qpt_ ) {
		  diff_[0][0]=0.5*(Models_[0])->get_vol()*(Models_[0])->get_vol()*qpt_(0)*qpt_(0);
		  diff_[1][1]=0.5*(Models_[1])->get_vol()*(Models_[1])->get_vol()*qpt_(1)*qpt_(1);
		  diff_[0][1]=0.5*rho_*(Models_[0])->get_vol()*(Models_[1])->get_vol()*qpt_(0)*qpt_(1);
		  diff_[1][0]=diff_[0][1];
		}

}
#endif