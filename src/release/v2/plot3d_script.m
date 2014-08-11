tri=delaunay(grid_x,grid_y);
[r,c] = size(tri);
h = trisurf(tri, grid_x, grid_y, sol);
axis vis3d;

lighting phong
colorbar EastOutside
