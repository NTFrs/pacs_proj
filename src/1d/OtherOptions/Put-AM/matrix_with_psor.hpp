#ifndef __deal2__sparse_matrix_withProjectedSOR_h
#define __deal2__sparse_matrix_withProjectedSOR_h

#include <deal.II/base/config.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/identity_matrix.h>
#include <deal.II/lac/exceptions.h>
#include <deal.II/lac/vector.h>


DEAL_II_NAMESPACE_OPEN

using namespace dealii;

template <typename number>
class SparseMatrix;

#include <deal.II/lac/sparse_matrix.h>

template <typename number>
class SparseMatrix_withProjectedSOR : public SparseMatrix<number> {
public:
        // Inheriting needed typedefs
        using typename SparseMatrix<number>::size_type;
        
        using typename SparseMatrix<number>::value_type;
        
        using typename SparseMatrix<number>::real_type;
        
        using typename SparseMatrix<number>::const_iterator;
        
        using typename SparseMatrix<number>::iterator;
        
        SparseMatrix_withProjectedSOR(){};
        
        template <typename somenumber>
        void ProjectedSOR_step (Vector<somenumber> &v,
                                const Vector<somenumber> &v_old,
                                const Vector<somenumber> &b,
                                const number        om = 1.)
        
        {
                
                AssertDimension (SparseMatrix<number>::m(), SparseMatrix<number>::n());
                Assert (SparseMatrix<number>::m() == v.size(), ExcDimensionMismatch(SparseMatrix<number>::m(),v.size()));
                Assert (SparseMatrix<number>::m() == v_old.size(), ExcDimensionMismatch(SparseMatrix<number>::m(),v_old.size()));
                Assert (SparseMatrix<number>::m() == b.size(), ExcDimensionMismatch(SparseMatrix<number>::m(),b.size()));
                
                for (unsigned row=1; row<SparseMatrix<number>::m()-1; ++row) {
                        
                        SparseMatrixIterators::Iterator< number, true > col (this, row, 0);
                        SparseMatrixIterators::Iterator< number, true > colend (this, row+1, 0);
                        
                        SparseMatrixIterators::Accessor< somenumber, true > row_iterator(*col);
                        
                        somenumber z=b(row);
                        
                        somenumber diag_value=0.;
                        
                        for ( ;  col<colend; col++){
                                
                                row_iterator=*col;
                                
                                if (row_iterator.column()<row)
                                        z-=row_iterator.value()*v(row_iterator.column());
                                        
                                if (row_iterator.column()>row)
                                        z-=row_iterator.value()*v_old(row_iterator.column());
                                
                                else
                                        diag_value=row_iterator.value();
                                                
                        }
                        
                        v(row)=v_old(row)+om*(z/diag_value-v_old(row));
                        
                }
                /*
                for (size_type row=0; row<SparseMatrix<number>::m(); ++row)
                {
                        somenumber s = b(row);
                        for (size_type j=cols->rowstart[row]; j<cols->rowstart[row+1]; ++j)
                        {
                                s -= val[j] * v(cols->colnums[j]);
                        }
                        Assert(val[cols->rowstart[row]]!= 0., ExcDivideByZero());
                        v(row) += s * om / val[cols->rowstart[row]];
                }*/
        }
        
protected:
        void prepare_add();
        
        void prepare_set();
        
private:
        
        SmartPointer<const SparsityPattern, SparseMatrix<number> > cols;
        
        number *val;
        
        std::size_t max_len;
        
        // make all other sparse matrices friends
        template <typename somenumber> friend class SparseMatrix;
        template <typename somenumber> friend class SparseLUDecomposition;
        template <typename> friend class SparseILU;
        
        template <typename> friend class BlockMatrixBase;
        
        template <typename, bool> friend class SparseMatrixIterators::Iterator;
        template <typename, bool> friend class SparseMatrixIterators::Accessor;
        
        template <typename number2> friend class SparseMatrixIterators::Accessor<number2, false>::Reference;
        
        friend class SparsityPattern;
};

/*
template <typename number>
template <typename somenumber>
void
SparseMatrix_withProjectedSOR<number>::ProjectedSOR_step (Vector<somenumber> &v,
                                                          const Vector<somenumber> &b,
                                                          const number        om) const
{
        Assert (cols != 0, ExcNotInitialized());
        Assert (val != 0, ExcNotInitialized());
        AssertDimension (SparseMatrix<number>::m(), SparseMatrix<number>::n());
        Assert (SparseMatrix<number>::m() == v.size(), ExcDimensionMismatch(SparseMatrix<number>::m(),v.size()));
        Assert (SparseMatrix<number>::m() == b.size(), ExcDimensionMismatch(SparseMatrix<number>::m(),b.size()));
        
        for (size_type row=0; row<SparseMatrix<number>::m(); ++row)
        {
                somenumber s = b(row);
                for (size_type j=cols->rowstart[row]; j<cols->rowstart[row+1]; ++j)
                {
                        s -= val[j] * v(cols->colnums[j]);
                }
                Assert(val[cols->rowstart[row]]!= 0., ExcDivideByZero());
                v(row) += s * om / val[cols->rowstart[row]];
        }
}
*/

DEAL_II_NAMESPACE_CLOSE

#endif 