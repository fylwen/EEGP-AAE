# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# Implementation of the pose error functions described in:
# Hodan et al., "On Evaluation of 6D Object Pose Estimation", ECCVW 2016

import math
import numpy as np
from scipy import spatial
from . import renderer, misc, visibility

def vsd(R_est, t_est, R_gt, t_gt, model, depth_test, K, delta, tau,
        cost_type='tlinear'):
    """
    Visible Surface Discrepancy.

    :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
    :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :param depth_test: Depth image of the test scene.
    :param K: Camera matrix.
    :param delta: Tolerance used for estimation of the visibility masks.
    :param tau: Misalignment tolerance.
    :param cost_type: Pixel-wise matching cost:
        'tlinear' - Used in the original definition of VSD in:
            Hodan et al., On Evaluation of 6D Object Pose Estimation, ECCVW 2016
        'step' - Used for SIXD Challenge 2017. It is easier to interpret.
    :return: Error of pose_est w.r.t. pose_gt.
    """

    im_size = (depth_test.shape[1], depth_test.shape[0])

    # Render depth images of the model in the estimated and the ground truth pose
    depth_est = renderer.render(model, im_size, K, R_est, t_est, clip_near=100,
                                clip_far=10000, mode='depth')

    depth_gt = renderer.render(model, im_size, K, R_gt, t_gt, clip_near=100,
                               clip_far=10000, mode='depth')

    # Convert depth images to distance images
    dist_test = misc.depth_im_to_dist_im(depth_test, K)
    dist_gt = misc.depth_im_to_dist_im(depth_gt, K)
    dist_est = misc.depth_im_to_dist_im(depth_est, K)

    # Visibility mask of the model in the ground truth pose
    visib_gt = visibility.estimate_visib_mask_gt(dist_test, dist_gt, delta)

    # Visibility mask of the model in the estimated pose
    visib_est = visibility.estimate_visib_mask_est(dist_test, dist_est, visib_gt, delta)

    # Intersection and union of the visibility masks
    visib_inter = np.logical_and(visib_gt, visib_est)
    visib_union = np.logical_or(visib_gt, visib_est)

    # Pixel-wise matching cost
    costs = np.abs(dist_gt[visib_inter] - dist_est[visib_inter])
    if cost_type == 'step':
        costs = costs >= tau
    elif cost_type == 'tlinear': # Truncated linear
        costs *= (1.0 / tau)
        costs[costs > 1.0] = 1.0
    else:
        print('Error: Unknown pixel matching cost.')
        exit(-1)

    # costs_vis = np.ones(dist_gt.shape)
    # costs_vis[visib_inter] = costs
    # import matplotlib.pyplot as plt
    # plt.matshow(costs_vis)
    # plt.colorbar()
    # plt.show()

    # Visible Surface Discrepancy
    visib_union_count = visib_union.sum()
    visib_comp_count = visib_union_count - visib_inter.sum()
    if visib_union_count > 0:
        e = (costs.sum() + visib_comp_count) / float(visib_union_count)
    else:
        e = 1.0
    return e

def vsd_visib(R_est, t_est, R_gt, t_gt, model, depth_test, K, delta, tau,
        cost_type='tlinear',bgr_test=None):
    """
    Visible Surface Discrepancy.

    :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
    :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :param depth_test: Depth image of the test scene.
    :param K: Camera matrix.
    :param delta: Tolerance used for estimation of the visibility masks.
    :param tau: Misalignment tolerance.
    :param cost_type: Pixel-wise matching cost:
        'tlinear' - Used in the original definition of VSD in:
            Hodan et al., On Evaluation of 6D Object Pose Estimation, ECCVW 2016
        'step' - Used for SIXD Challenge 2017. It is easier to interpret.
    :return: Error of pose_est w.r.t. pose_gt.
    """
    im_size = (depth_test.shape[1], depth_test.shape[0])

    # Render depth images of the model in the estimated and the ground truth pose
    depth_est = renderer.render(model, im_size, K, R_est, t_est, clip_near=100,
                                clip_far=10000, mode='depth')

    depth_gt = renderer.render(model, im_size, K, R_gt, t_gt, clip_near=100,
                               clip_far=10000, mode='depth')

    # Convert depth images to distance images
    dist_test = misc.depth_im_to_dist_im(depth_test, K)
    dist_gt = misc.depth_im_to_dist_im(depth_gt, K)
    dist_est = misc.depth_im_to_dist_im(depth_est, K)

    # Visibility mask of the model in the ground truth pose
    visib_gt = visibility.estimate_visib_mask_gt(dist_test, dist_gt, delta)

    # Visibility mask of the model in the estimated pose
    visib_est = visibility.estimate_visib_mask_est(dist_test, dist_est, visib_gt, delta)

    # Intersection and union of the visibility masks
    visib_inter = np.logical_and(visib_gt, visib_est)
    visib_union = np.logical_or(visib_gt, visib_est)

    # Pixel-wise matching cost
    costs = np.abs(dist_gt[visib_inter] - dist_est[visib_inter])
    if not (bgr_test is None):
        costs_vis=np.zeros(dist_gt.shape)
        costs_vis[visib_union]=1.0
        costs_vis[visib_inter]=costs/np.float(tau)
        costs_vis[costs_vis>1.0]=1.0

    if cost_type == 'step':
        costs = costs >= tau
    elif cost_type == 'tlinear': # Truncated linear
        costs *= (1.0 / tau)
        costs[costs > 1.0] = 1.0
    else:
        print('Error: Unknown pixel matching cost.')
        exit(-1)

    # costs_vis = np.ones(dist_gt.shape)
    # costs_vis[visib_inter] = costs
    # import matplotlib.pyplot as plt
    # plt.matshow(costs_vis)
    # plt.colorbar()
    # plt.show()

    # Visible Surface Discrepancy
    visib_union_count = visib_union.sum()
    visib_comp_count = visib_union_count - visib_inter.sum()
    if visib_union_count > 0:
        e = (costs.sum() + visib_comp_count) / float(visib_union_count)
    else:
        e = 1.0

    # ==============================
    # Compute Portion of visibility mask
    mask_gt=(depth_gt>0)
    num_pixels_render_gt=np.count_nonzero(mask_gt)
    num_pixels_visib_gt=np.count_nonzero(visib_gt)
    visib_portion=float(num_pixels_visib_gt)/num_pixels_render_gt

    #==========show color=============
    if not (bgr_test is None):
        import cv2
        gray_test = cv2.cvtColor(np.uint8(bgr_test), cv2.COLOR_BGR2GRAY)[:,:,np.newaxis]
        gray_test=np.concatenate((gray_test,gray_test,gray_test),axis=-1)
        gray_test=(0.5*gray_test).astype(np.uint8)


        mask_est3=(depth_est>0)[:,:,np.newaxis]
        render_est3=np.concatenate((mask_est3,mask_est3,mask_est3),axis=-1)
        visib_est3=np.concatenate((visib_est[:,:,np.newaxis],visib_est[:,:,np.newaxis],visib_est[:,:,np.newaxis]),axis=-1)
        render_gt3=np.concatenate((mask_gt[:,:,np.newaxis],mask_gt[:,:,np.newaxis],mask_gt[:,:,np.newaxis]),axis=-1)
        visib_gt3=np.concatenate((visib_gt[:,:,np.newaxis],visib_gt[:,:,np.newaxis],visib_gt[:,:,np.newaxis]),axis=-1)
        visib_inter3=np.concatenate((visib_inter[:,:,np.newaxis],visib_inter[:,:,np.newaxis],visib_inter[:,:,np.newaxis]),axis=-1)
        visib_union3=np.concatenate((visib_union[:,:,np.newaxis],visib_union[:,:,np.newaxis],visib_union[:,:,np.newaxis]),axis=-1)

        show_render_est=np.where(render_est3,bgr_test,gray_test)
        show_visib_est=np.where(visib_est3,bgr_test,gray_test)
        show_render_gt=np.where(render_gt3,bgr_test,gray_test)
        show_visib_gt=np.where(visib_gt3,bgr_test,gray_test)
        show_inter=np.where(visib_inter3,bgr_test,gray_test)
        show_union=np.where(visib_union3,bgr_test,gray_test)

        costs_vis=(costs_vis*255).astype(np.uint8)

        cv2.imshow('show_visib_est',show_visib_est)
        cv2.imshow('show_render_est',show_render_est)
        cv2.imshow('show_visib_gt:{:.2f}'.format(visib_portion),show_visib_gt)
        cv2.imshow('show_render_gt',show_render_gt)
        cv2.imshow('show_visib_gt',show_visib_gt)
        cv2.imshow('show_inter',show_inter)
        cv2.imshow('show_union',show_union)
        cv2.imshow('show_costs_vis:{:.2f}'.format(e),costs_vis)
        cv2.waitKey()
        cv2.destroyAllWindows()




    return e,visib_portion

def cou(R_est, t_est, R_gt, t_gt, model, im_size, K):
    """
    Complement over Union, i.e. the inverse of the Intersection over Union used
    in the PASCAL VOC challenge - by Everingham et al. (IJCV 2010).

    :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
    :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :param im_size: Test image size.
    :param K: Camera matrix.
    :return: Error of pose_est w.r.t. pose_gt.
    """

    # Render depth images of the model in the estimated and the ground truth pose
    d_est = renderer.render(model, im_size, K, R_est, t_est, clip_near=100,
                            clip_far=10000, mode='depth')

    d_gt = renderer.render(model, im_size, K, R_gt, t_gt, clip_near=100,
                           clip_far=10000, mode='depth')

    # Masks of the rendered model and their intersection and union
    mask_est = d_est > 0
    mask_gt = d_gt > 0
    inter = np.logical_and(mask_gt, mask_est)
    union = np.logical_or(mask_gt, mask_est)

    union_count = float(union.sum())
    if union_count > 0:
        e = 1.0 - inter.sum() / union_count
    else:
        e = 1.0
    return e

def add(R_est, t_est, R_gt, t_gt, model):
    """
    Average Distance of Model Points for objects with no indistinguishable views
    - by Hinterstoisser et al. (ACCV 2012).

    :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
    :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :return: Error of pose_est w.r.t. pose_gt.
    """
    pts_est = misc.transform_pts_Rt(model['pts'], R_est, t_est)
    pts_gt = misc.transform_pts_Rt(model['pts'], R_gt, t_gt)
    e = np.linalg.norm(pts_est - pts_gt, axis=1).mean()
    return e

def adi(R_est, t_est, R_gt, t_gt, model):
    """
    Average Distance of Model Points for objects with indistinguishable views
    - by Hinterstoisser et al. (ACCV 2012).

    :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
    :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :return: Error of pose_est w.r.t. pose_gt.
    """
    pts_est = misc.transform_pts_Rt(model['pts'], R_est, t_est)
    pts_gt = misc.transform_pts_Rt(model['pts'], R_gt, t_gt)

    # Calculate distances to the nearest neighbors from pts_gt to pts_est
    nn_index = spatial.cKDTree(pts_est)
    nn_dists, _ = nn_index.query(pts_gt, k=1)

    e = nn_dists.mean()
    return e

def re(R_est, R_gt):
    """
    Rotational Error.

    :param R_est: Rotational element of the estimated pose (3x1 vector).
    :param R_gt: Rotational element of the ground truth pose (3x1 vector).
    :return: Error of t_est w.r.t. t_gt.
    """
    assert(R_est.shape == R_gt.shape == (3, 3))
    error_cos = 0.5 * (np.trace(R_est.dot(np.linalg.inv(R_gt))) - 1.0)
    error_cos = min(1.0, max(-1.0, error_cos)) # Avoid invalid values due to numerical errors
    error = math.acos(error_cos)
    error = 180.0 * error / np.pi # [rad] -> [deg]
    return error

def te(t_est, t_gt):
    """
    Translational Error.

    :param t_est: Translation element of the estimated pose (3x1 vector).
    :param t_gt: Translation element of the ground truth pose (3x1 vector).
    :return: Error of t_est w.r.t. t_gt.
    """
    assert(t_est.size == t_gt.size == 3)
    error = np.linalg.norm(t_gt - t_est)
    return error
