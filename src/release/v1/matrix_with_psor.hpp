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
class SparseMatrix_PSOR : public virtual SparseMatrix<number> {
public:
        // Inheriting needed typedef
        using typename SparseMatrix<number>::size_type;
        
        // Constructors
        SparseMatrix_PSOR():SparseMatrix<number>(){};
        
        SparseMatrix_PSOR(const SparseMatrix<number> &A):SparseMatrix<number>(A){};
        
        explicit SparseMatrix_PSOR(const SparsityPattern &sparsity):SparseMatrix<number>(sparsity){};
        
        SparseMatrix_PSOR(const SparsityPattern &sparsity, const IdentityMatrix  &id):
        SparseMatrix<number>(sparsity, id){};
        
        // SparseMatrix methods
        using SparseMatrix<number>::reinit;
        using SparseMatrix<number>::add;
        
        // Adding a new SOR_Step for Price
        void ProjectedSOR_step (Vector<number> &v,                          // Solution
                                const Vector<number> &v_old,                // Solution step before
                                const Vector<number> &b,                    // right hand side
                                std::map<dealii::types::global_dof_index,dealii::Point<dim> > &vertices,                                                            // mesh points
                                const number        K,                          // Strike
                                const number        om = 1.);                   // SOR parameter
        
        // Adding a new SOR_Step for LogPrice
        void ProjectedSOR_step (Vector<number> &v,                          // Solution
                                const Vector<number> &v_old,                // Solution step before
                                const Vector<number> &b,                    // right hand side
                                std::map<dealii::types::global_dof_index,dealii::Point<dim> > &vertices,                                                            // mesh points
                                const number        K,                          // Strike
                                const std::vector<number> &S0,                  // Spot
                                const number        om = 1.);                   // SOR parameter
        
};

template <typename number, unsigned dim>
void
dealii::SparseMatrix_PSOR<number, dim>::ProjectedSOR_step (Vector<number> &v,
                                                           const Vector<number> &v_old,
                                                           const Vector<number> &b,
                                                           std::map<dealii::types::global_dof_index,dealii::Point<dim> > &vertices,                                                            
                                                           const number        K,
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
                
                double point=0.;
                
                for (unsigned d=0; d<dim; ++d) {
                        point+=vertices[row][d];
                }
                
                v(row)=(K-point>v_old(row)+om*(z/(this->diag_element(row))-v_old(row)))
                ?
                (K-point)
                :
                v_old(row)+om*(z/(this->diag_element(row))-v_old(row));
                
        }
}

template <typename number, unsigned dim>
void
dealii::SparseMatrix_PSOR<number, dim>::ProjectedSOR_step (Vector<number> &v,
                                                           const Vector<number> &v_old,
                                                           const Vector<number> &b,
                                                           std::map<dealii::types::global_dof_index,dealii::Point<dim> > &vertices,                                                            
                                                           const number        K,
                                                           const std::vector<number> &S0,
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
                
                double point=0.;
                
                for (unsigned d=0; d<dim; ++d) {
                        point+=S0[d]*exp(vertices[row][d]);
                }
                
                v(row)=(K-point>v_old(row)+om*(z/(this->diag_element(row))-v_old(row)))
                ?
                (K-point)
                :
                v_old(row)+om*(z/(this->diag_element(row))-v_old(row));
                
        }
}

DEAL_II_NAMESPACE_CLOSE

#endif 