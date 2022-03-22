import numpy as np
import open3d as o3d

a = input("the name of .ply  ") + ".ply"
print(a)
pcd = o3d.io.read_point_cloud(a)
print(pcd)
print(np.asarray(pcd.points))

o3d.visualization.draw_geometries([pcd])
