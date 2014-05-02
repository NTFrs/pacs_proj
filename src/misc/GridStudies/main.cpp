#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <fstream>

using namespace dealii;

int main() {
	
	Triangulation<2> tria1;
	Point<2> p1, p2;
	p1[0]=-2;
	p1[1]=-2;
	p2[0]=3;
	p2[1]=3;
	GridGenerator::hyper_rectangle(tria1, p1, p2);
// 	tria1.refine_global(3);
	{
	std::ofstream out("grid-1.eps");
	GridOut grid_out;
	grid_out.write_eps(tria1, out);
	}
 
	Triangulation<1> tria2;
	Point<1> p3, p4;
	p3[0]=-4;
	p4[0]=4;
	GridGenerator::hyper_rectangle(tria2, p3, p4);
/*	
	{
	 std::ofstream out("grid2.eps");
	 GridOut grid_out;
	 grid_out.write_eps(tria2, out);
	}
	*/
	return 0;
	
}