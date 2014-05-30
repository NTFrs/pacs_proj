#ifndef __deal2__sparse_matrix_withProjectedSOR_h
#define __deal2__sparse_matrix_withProjectedSOR_h

#include <deal.II/base/config.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/identity_matrix.h>
#include <deal.II/lac/exceptions.h>
#include <deal.II/lac/vector.h>

#include <deal.II/lac/sparse_matrix.h>

DEAL_II_NAMESPACE_OPEN

template <typename number, unsigned dim>
class SparseMatrix_withProjectedSOR : public virtual SparseMatrix<number> {
public:
        // Inheriting needed typedef
        using typename SparseMatrix<number>::size_type;
        
        
        
        // Adding a new SOR_Step
        void ProjectedSOR_step (Vector<number> &v,                          // Solution
                                const Vector<number> &v_old,                // Solution step before
                                const Vector<number> &b,                    // right hand side
                                const std::vector< Point<dim> > &grid_points,   // mesh points
                                const number        K,                          // Strike
                                const number        om = 1.);                   // SOR parameter
        
private:
        SmartPointer<const SparsityPattern,SparseMatrix<number> > cols;
        
        number *val;
        
        std::size_t max_len;

};

template <typename number, unsigned dim>
void
dealii::SparseMatrix_withProjectedSOR<number, dim>::ProjectedSOR_step (Vector<number> &v,
                                         const Vector<number> &v_old,
                                         const Vector<number> &b,
                                         const std::vector< Point<dim> > &grid_points,
                                         const number        K,
                                         const number        om)

{
        AssertDimension (SparseMatrix<number>::m(), SparseMatrix<number>::n());
        Assert (SparseMatrix<number>::m() == v.size(), ExcDimensionMismatch(SparseMatrix<number>::m(),v.size()));
        Assert (SparseMatrix<number>::m() == v_old.size(), ExcDimensionMismatch(SparseMatrix<number>::m(),v_old.size()));
        Assert (SparseMatrix<number>::m() == b.size(), ExcDimensionMismatch(SparseMatrix<number>::m(),b.size()));
        
#pragma omp parallel for
        for (size_type row=1; row<SparseMatrix<number>::m()-1; ++row) {
                
                SparseMatrixIterators::Iterator< number, true > col (this, row, 0);
                SparseMatrixIterators::Iterator< number, true > colend (this, row+1, 0);
                
                SparseMatrixIterators::Accessor< number, true > row_iterator(*col);
                
                number z=b(row);
                
                for ( ;  col<colend; ++col){
                        
                        row_iterator=*col;
                        
                        if (row_iterator.column()<row)
                                z-=row_iterator.value()*v(row_iterator.column());
                        
                        if (row_iterator.column()>row)
                                z-=row_iterator.value()*v_old(row_iterator.column());
                        
                }
                
                v(row)=(K-grid_points[row][0]>v_old(row)+om*(z/(this->diag_element(row))-v_old(row)))
                        ?
                        (K-grid_points[row][0])
                        :
                        v_old(row)+om*(z/(this->diag_element(row))-v_old(row));
                
        }
}

DEAL_II_NAMESPACE_CLOSE

#endif 