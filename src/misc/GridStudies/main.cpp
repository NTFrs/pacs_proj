#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>


#include <fstream>
#include <iostream>

using namespace dealii;
using namespace std;
int main() {
	
	Triangulation<2> tria1;
	Point<2> p1, p2;
	p1[0]=-1;
	p1[1]=-1;
	p2[0]=1;
	p2[1]=1;
	GridGenerator::hyper_rectangle(tria1, p1, p2);
	FE_Q<2> fe(1);
	DoFHandler<2> dof_handler(tria1);
// 	tria1.refine_global(3);
	dof_handler.distribute_dofs(fe);

	for (unsigned int i=0;i<3;++i) {
	 {
	
	typename DoFHandler<2>::active_cell_iterator cell=dof_handler.begin_active(),  endc=dof_handler.end();
	cout<< "Faces when mesh is composed by "<< tria1.n_active_cells()<< endl;
	for (;cell!=endc;++cell) {
	cout<< "in cell N "<< cell<< "\n";
	for (unsigned int face=0;
	 face<GeometryInfo<2>::faces_per_cell;++face) {
	  cout<< "face N "<< face<< " we are at point "<< cell->face(face)->center()<< endl;
	  }
	 }
	 tria1.refine_global(1);
	 dof_handler.distribute_dofs(fe);
	 }
	 
	 }
// 	Triangulation<1> tria2;
// 	Point<1> p3, p4;
// 	p3[0]=-4;
// 	p4[0]=4;
// 	GridGenerator::hyper_rectangle(tria2, p3, p4);
/*	
	{
	 std::ofstream out("grid2.eps");
	 GridOut grid_out;
	 grid_out.write_eps(tria2, out);
	}
	*/
	return 0;
	
}