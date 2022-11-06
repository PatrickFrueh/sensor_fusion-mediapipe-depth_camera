import open3d as o3d
import numpy as np
import copy
import cv2


rotM_21 = np.array(
    [
        [0.99928165, 0.0362255, 0.01113128],
        [-0.0234766, 0.82230798, -0.5685582],
        [-0.02974964, 0.56788845, 0.82256773],
    ]
)
transV_21 = np.array([[0.05436921], [0.30287669], [0.24625487]])


def draw_registration_result(source, target, transformation):
    # o3d.visualization.Visualizer.create_window('point clouds visualization', 1280, 720)
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries(
        [source_temp, target_temp], width=1280, height=720
    )


def apply_noise(pcd, mu, sigma):
    noisy_pcd = copy.deepcopy(pcd)
    points = np.asarray(noisy_pcd.points)
    points += np.random.normal(mu, sigma, size=points.shape)
    noisy_pcd.points = o3d.utility.Vector3dVector(points)
    return noisy_pcd


if __name__ == "__main__":
    source = o3d.io.read_point_cloud("cropped_2.ply")
    target = o3d.io.read_point_cloud("cropped_1.ply")
    source.estimate_normals()
    target.estimate_normals()
    threshold = 0.05

    # Initial guess (using e.g. estimateAffine3D, user_interaction)
    trans_init = np.array(
        [
            [0.99965696, -0.01945417, -0.01753568, 0.05130496],
            [0.00665568, 0.83624558, -0.54831472, -0.29418819],
            [0.02533114, 0.54800991, 0.8360882, -0.24016013],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    draw_registration_result(source, target, trans_init)

    print("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, trans_init
    )
    print("Evaluation of initial guess:", evaluation)

    # Point-to-point ICP
    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    # ICPConvergenceCriteria(max_iteration=2000)
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    print("")
    draw_registration_result(source, target, reg_p2p.transformation)

    # Adjusted convergence criteria
    print("Apply point-to-point ICP: convergence criteria set to 2000")
    reg_p2p_CC = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000),
    )
    print(reg_p2p_CC)
    print("Transformation is:")
    print(reg_p2p_CC.transformation)
    print("")
    draw_registration_result(source, target, reg_p2p_CC.transformation)

    # Robust point-to-plane ICP
    mu, sigma = 0, 0.1  # mean and standard deviation
    source_noisy = apply_noise(source, mu, sigma)
    print("Robust point-to-plane ICP, threshold={}:".format(threshold))
    loss = o3d.pipelines.registration.TukeyLoss(k=sigma)
    print("Using robust loss:", loss)
    p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source_noisy, target, threshold, trans_init, p2l
    )
    print(reg_p2l)
    print("Transformation is:")
    print(reg_p2l.transformation)
    draw_registration_result(source, target, reg_p2l.transformation)

    # Point-to-plane ICP
    print("Apply point-to-plane ICP")
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    print(reg_p2l)
    print("Transformation is:")
    print(reg_p2l.transformation)
    print("")
    draw_registration_result(source, target, reg_p2l.transformation)
