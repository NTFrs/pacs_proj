#ifndef __deal2__sparse_matrix_withProjectedSOR_h
#define __deal2__sparse_matrix_withProjectedSOR_h

#include "dealii.hpp"

DEAL_II_NAMESPACE_OPEN

//! Sparse Matrix, with PSOR.
/*! This class is a simple decorator for \code {.cpp} dealii::SparseMatrix \endcode
 * It inheritances \code {.cpp}
 * dealii::reinit
 * dealii::add
 * \endcode
 * methods used in the program and adds two overloaded methods, called \code {.cpp} ProjectedSOR_step \endcode used for solving the obstacle problem. For all the information about the base class, we suggest you to take a look in the documentation of dealii <a href="http://www.dealii.org/developer/doxygen/deal.II/classSparseMatrix.html">classSparseMatrix</a>.
 */

template <typename number, unsigned dim>
class SparseMatrix_PSOR : public virtual SparseMatrix<number> {
public:
        // Inheriting needed typedef
        using typename SparseMatrix<number>::size_type;
        
        //!
        /*! Constructor; initializes the matrix to be empty, without any structure, i.e. the matrix is not usable at all. This constructor is therefore only useful for matrices which are members of a class. All other matrices should be created at a point in the data flow where all necessary information is available.
         */
        SparseMatrix_PSOR():SparseMatrix<number>(){};
        
        //!
        /*! Copy constructor. This constructor is only allowed to be called if the matrix to be copied is empty.
         */
        SparseMatrix_PSOR(const SparseMatrix<number> &A):SparseMatrix<number>(A){};
        
        //!
        /*! Constructor. Takes the given matrix sparsity structure to represent the sparsity pattern of this matrix.
         */
        explicit SparseMatrix_PSOR(const SparsityPattern &sparsity):SparseMatrix<number>(sparsity){};
        
        //!
        /*! Copy constructor: initialize the matrix with the identity matrix. This constructor will throw an exception if the sizes of the sparsity pattern and the identity matrix do not coincide, or if the sparsity pattern does not provide for nonzero entries on the entire diagonal.
         */
        SparseMatrix_PSOR(const SparsityPattern &sparsity, const IdentityMatrix  &id):
        SparseMatrix<number>(sparsity, id){};
        
        //!
        /*! Non-virtual method inherited by SparseMatrix
         */
        using SparseMatrix<number>::reinit;
        //!
        /*! Non-virtual method inherited by SparseMatrix
         */
        using SparseMatrix<number>::add;
        
        
        //! ProjectedSOR_step generic
        /*! This method performs a PSOR step, just like any other iterative method of SparseMatrix. This one tough solves the obstacle problem of the American Option.
         *  \param v            solution
         *  \param v_old        solution at the previous step
         *  \param b            right hand side
         *  \param vertices     vector of the mesh points
         *  \param payoff       Function that exprimes the payoff
         *  \param om           SOR parameter
         */
        void ProjectedSOR_step (Vector<number> &v,
                                const Vector<number> &v_old,
                                const Vector<number> &b,
                                const std::vector< dealii::Point<dim> > &vertices,
                                Function<dim> const & payoff,
                                const number        om = 1.);
        
};

template <typename number, unsigned dim>
void
dealii::SparseMatrix_PSOR<number, dim>::ProjectedSOR_step (Vector<number> &v,
                                                           const Vector<number> &v_old,
                                                           const Vector<number> &b,
                                                           const std::vector< dealii::Point<dim> > &vertices,                                                            
                                                           Function<dim> const & payoff,
                                                           const number        om)

{
        AssertDimension (SparseMatrix<number>::m(), SparseMatrix<number>::n());
        Assert (SparseMatrix<number>::m() == v.size(), ExcDimensionMismatch(SparseMatrix<number>::m(),v.size()));
        Assert (SparseMatrix<number>::m() == v_old.size(), ExcDimensionMismatch(SparseMatrix<number>::m(),v_old.size()));
        Assert (SparseMatrix<number>::m() == b.size(), ExcDimensionMismatch(SparseMatrix<number>::m(),b.size()));
        
#pragma omp parallel for
        for (size_type row=1; row<SparseMatrix<number>::m()-1; ++row) {
                
                SparseMatrixIterators::Iterator< number, true > col=this->begin(row);
                SparseMatrixIterators::Iterator< number, true > colend=this->begin(row+1);
                
                SparseMatrixIterators::Accessor< number, true > row_iterator(*col);
                
                number z=b(row);
                
                for ( ;  col<colend; ++col){
                        
                        row_iterator=*col;
                        
                        if (row_iterator.column()<row)
                                z-=row_iterator.value()*v(row_iterator.column());
                        
                        if (row_iterator.column()>row)
                                z-=row_iterator.value()*v_old(row_iterator.column());
                        
                }
                
                
                v(row)=(payoff.value(vertices[row])>v_old(row)+om*(z/(this->diag_element(row))-v_old(row)))
                ?
                (payoff.value(vertices[row]))
                :
                v_old(row)+om*(z/(this->diag_element(row))-v_old(row));
                
        }
}

DEAL_II_NAMESPACE_CLOSE

#endif 