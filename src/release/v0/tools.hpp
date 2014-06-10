#ifndef __tools_hpp
# define __tools_hpp

namespace tools{


//!An iterator used to cycle on verices of a triangulation
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
}
#endif