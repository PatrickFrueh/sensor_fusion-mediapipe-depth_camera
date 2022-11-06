# Remove unwanted Areas around spheres with
# a specified center and desired radius

import numpy as np
import open3d

# Read point cloud from PLY-file
pcd1 = open3d.io.read_point_cloud("point_cloud-path.ply")
points = np.asarray(pcd1.points)

# Sphere center and radius
center_coordinates = np.array([0, 0, 0.4])
radius = 1.2

# Calculate distances to center, set new points
distances = np.linalg.norm(points - center_coordinates, axis=1)
pcd1.points = open3d.utility.Vector3dVector(points[distances <= radius])

# Write point cloud out
print("Writing adjusted point cloud to file...")
open3d.io.write_point_cloud("point_cloud-noise_removed.ply", pcd1)
print("Done removing noise.")
