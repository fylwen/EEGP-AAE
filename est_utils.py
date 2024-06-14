import numpy as np
import open3d as o3d
import copy
import data_utils
from pysixd_stuff.pysixd import pose_error
import cv2

from pysixd_stuff.pysixd import misc, renderer_vt


verbose=False

##############################################################################################################################
def depth_refinement(depth_crop, model_reconst, R_est, t_est, K_test, test_render_dims, max_mean_dist_factor=2.0):
    rgb_est, depth_est = renderer_vt.render_phong(model_reconst, test_render_dims, np.array(K_test).reshape((3, 3)),
                                          R_est, t_est.reshape((3, 1)), clip_near=10.,
                                          clip_far=10000., mode='rgb+depth', random_light=False)

    if verbose:
        print('Depth Refinement-')
        print('Estimated R|t',R_est, t_est)
        cv2.imshow('Original',depth_crop)
        cv2.waitKey()

    depth_est_ys, depth_est_xs=np.where(depth_est>0)
    mean_depth_est=np.sum(depth_est)/len(depth_est_ys)
    max_mean_dist=np.max(np.fabs(depth_est[depth_est_ys,depth_est_xs]-mean_depth_est))

    depth_crop_ys, depth_crop_xs=np.where(depth_crop>0)
    mean_depth_real=np.sum(depth_crop)/len(depth_crop_ys)

    depth_crop=np.where(np.fabs(depth_crop-mean_depth_real)<max_mean_dist_factor*max_mean_dist,depth_crop,0)
    if verbose:
        cv2.imshow('Filtered',depth_crop)
        cv2.waitKey()
        cv2.destroyAllWindows()
    if depth_crop.any():
        depth_crop_ys, depth_crop_xs=np.where(depth_crop>0)
        mean_depth_real = np.sum(depth_crop) / len(depth_crop_ys)
    else:
        print('Use original')

    print('mean:est {:f}//real{:f}'.format(mean_depth_est, mean_depth_real),'z: original {:f}//refine {:f}'.format(t_est[2],t_est[2]+mean_depth_real-mean_depth_est))
    print('max_mean_dist', max_mean_dist)

    rel_z=mean_depth_real-mean_depth_est
    return rel_z+t_est[2], max_mean_dist



############################################################################
def transform_pts_Rt_o3d(model, Rt):
    model_copy = copy.deepcopy(model)
    return model_copy.transform(Rt)


def rotation_error_icp(depth_img, model_o3d, bbox_est,  R_est, t_est, K_test, width, height,
                       max_mean_dist=100, max_mean_dist_factor=2.0, regist_error_threshold=2.0, fitness_threshold=0.8):
    ori_Rt_est=np.eye(4)
    ori_Rt_est[:3,:3]=R_est
    ori_Rt_est[:3, 3]=t_est
    ori_Rt_est_inv=np.linalg.inv(ori_Rt_est)

    model_ori= copy.deepcopy(model_o3d)
    if verbose:
        print('Rotation Error ICP-')
        print('double check max_mean_dist', max_mean_dist)
        print('Input R|t',R_est,t_est)
        print('check inverse transformation R:', np.dot(ori_Rt_est_inv[:3,:3],R_est).astype(np.float64))
        print('check inverse transformation t:', np.dot(ori_Rt_est_inv[:3,:3],-t_est)-ori_Rt_est_inv[:3,3])

    mask=np.zeros(depth_img.shape,dtype=np.int32)
    bbox_est[0]=max(0,int(bbox_est[0]))
    bbox_est[1]=max(0,int(bbox_est[1]))
    bbox_est[2]=min(width-1,int(bbox_est[0]+bbox_est[2]))
    bbox_est[3]=min(height-1,int(bbox_est[1]+bbox_est[3]))
    mask[bbox_est[1]:bbox_est[3],bbox_est[0]:bbox_est[2]]=1
    depth_img=np.where(mask,depth_img,0)
    if verbose:
        cv2.imshow('depth_img_pre',depth_img)
        cv2.waitKey()

    depth_crop_ys, depth_crop_xs=np.where(depth_img>0)
    mean_depth_real=np.sum(depth_img)/len(depth_crop_ys)

    depth_crop_ys_refine,_=np.where(np.fabs(depth_img-mean_depth_real)<max_mean_dist*max_mean_dist_factor)
    if len(depth_crop_ys_refine)>=1/4.*len(depth_crop_ys):
        print('refine depth map')
        depth_img=np.where(np.fabs(depth_img-mean_depth_real)<max_mean_dist*max_mean_dist_factor,depth_img,0)


    real_depth_pts = misc.rgbd_to_point_cloud(K_test, depth_img)[0]

    real_depth_pts_o3d = o3d.geometry.PointCloud()
    real_depth_pts_o3d.points=o3d.utility.Vector3dVector(real_depth_pts)
    real_depth_pts_o3d=o3d.geometry.voxel_down_sample(real_depth_pts_o3d,voxel_size=0.05)

    o3d.geometry.estimate_normals(real_depth_pts_o3d,o3d.geometry.KDTreeSearchParamHybrid(radius=max_mean_dist/10.,max_nn=30))
    o3d.geometry.orient_normals_towards_camera_location(real_depth_pts_o3d,np.array([0.,0.,0.]))


    if verbose:
        model_ori.paint_uniform_color([1, 0.7, 0])
        transformed_real_depth_pts_o3d = transform_pts_Rt_o3d(real_depth_pts_o3d, ori_Rt_est_inv)
        transformed_real_depth_pts_o3d.paint_uniform_color([0, 0.6, 0.9])
        o3d.visualization.draw_geometries([model_ori,transformed_real_depth_pts_o3d], width=480, height=480, left=300, top=300, window_name='Before-model coord')

    reg = o3d.registration.registration_icp(source=real_depth_pts_o3d, target=model_ori, max_correspondence_distance= regist_error_threshold, init=ori_Rt_est_inv,
                                            estimation_method=o3d.registration.TransformationEstimationPointToPlane(),
                                            criteria=o3d.registration.ICPConvergenceCriteria(max_iteration=100,relative_fitness=1e-6,relative_rmse=1e-6))

    refine_Rt_est=np.linalg.inv(reg.transformation)
    refine_R_est = refine_Rt_est[:3,:3]
    refine_t_est = refine_Rt_est[:3, 3]


    regist_fit = reg.fitness
    print('Fitness',regist_fit)

    if verbose:
        print('Double check input R|t',R_est,t_est)
        cv2.imshow('depth_img_after',depth_img)
        cv2.waitKey()
        cv2.destroyAllWindows()

        print('After check inverse transformation R:', np.dot(reg.transformation[:3,:3],refine_R_est).astype(np.float64))
        print('After check inverse transformation t:', np.dot(reg.transformation[:3,:3],-refine_t_est)-reg.transformation[:3,3])
        model_est = transform_pts_Rt_o3d(model_o3d,refine_Rt_est)
        model_est.paint_uniform_color([1, 0.7, 0])
        real_depth_pts_o3d.paint_uniform_color([0, 0.6, 0.9])
        o3d.visualization.draw_geometries([model_est,real_depth_pts_o3d], width=480, height=480, left=300, top=300, window_name='After-scan coord')

    if regist_fit < fitness_threshold:
        R_refined = R_est
        t_refined = t_est
        return R_refined, t_refined, regist_fit

    print('After Refine t', refine_t_est)
    return refine_R_est,refine_t_est, regist_fit


def nearest_rotation(_query, sess, x, pose_retrieved):
    if _query.dtype == 'uint8':
        _query = _query / 255.
    if _query.ndim == 3:
        _query = np.expand_dims(_query, 0)

    info_lookup = sess.run([pose_retrieved], feed_dict={x: _query})

    return info_lookup


def est_tra_w_tz(_mm_tz,Radius_render_train,K_test,center_obj_x_test,center_obj_y_test,
                 K_train,center_obj_x_train,center_obj_y_train):
    center_obj_mm_tx = center_obj_x_test * _mm_tz / K_test[0, 0] \
                       - center_obj_x_train * Radius_render_train / K_train[0, 0]
    center_obj_mm_ty = center_obj_y_test * _mm_tz / K_test[1, 1] \
                       - center_obj_y_train * Radius_render_train / K_train[1, 1]
    return np.array([center_obj_mm_tx, center_obj_mm_ty, _mm_tz]).reshape((3, 1))


def rectify_rot(init_rot, est_tra):
    d_alpha_x = - np.arctan(est_tra[0] / est_tra[2])
    d_alpha_y = - np.arctan(est_tra[1] / est_tra[2])
    R_corr_x = np.array([[1, 0, 0],
                         [0, np.cos(d_alpha_y), -np.sin(d_alpha_y)],
                         [0, np.sin(d_alpha_y), np.cos(d_alpha_y)]])
    R_corr_y = np.array([[np.cos(d_alpha_x)[0], 0, -np.sin(d_alpha_x)[0]],
                         [0, 1, 0],
                         [np.sin(d_alpha_x), 0, np.cos(d_alpha_x)]])
    return np.dot(R_corr_y, np.dot(R_corr_x, init_rot))

################################################################################################################
def pose_estimation_img_detection(sess, x, pose_retrieved, img_bgr, img_depth, info_dets, info_scene,
                                  codebook_rotation_matrix, codebook_obj_bbs, model_id, image_size,
                                  model_reconst=None, model_o3d=None, use_icp='none', est_err=False,
                                  is_vis=True, path_vis_prefix='./output',dataset_name='hinterstoisser'):

    H, W, _ = img_bgr.shape
    test_imgs = img_bgr[np.newaxis, :]
    _info_dets = []
    _info_dets.append(info_dets)

    # Generate crops on img_bgr according to the detection result
    test_depths = None
    if not img_depth is None:
        test_depths = img_depth[np.newaxis, :]
    test_img_crops, test_depth_crops, bbs, _, _ = data_utils.generate_scene_crops(test_imgs, test_depths, _info_dets,
                                                                                  W_AE=image_size, H_AE=image_size)
    list_info_pose_est = []
    # For each ROI with specific label in the image
    for id_box in range(0, len(test_img_crops[0])):
        cur_info_pose_est = info_dets[id_box].copy()
        query_bgr = test_img_crops[0][id_box]

        query_edge = cv2.Canny(query_bgr, 50, 150)
        if query_edge.ndim==2:
            query_edge=np.expand_dims(query_edge,2)

        if x.shape[-1] == 1:
            query = query_edge
        elif x.shape[-1] == 3:
            query = query_bgr
        elif x.shape[-1] == 4:
            query = np.concatenate((query_bgr, query_edge), axis=-1)

        info_lookup = nearest_rotation(query, sess, x, pose_retrieved)
        cur_info_pose_est['summary_ests'] = []

        # Store and Visualize the Most-Fit pose template retrieved by AAE
        if use_icp=='GT':
            print('w_gt2d')
            cur_info_pose_est['score'] = cur_info_pose_est['visib_portion']

        idx = info_lookup[0]['encoding_indices'][0]
        cur_info_pose_est['est_rot_cb'] = codebook_rotation_matrix[idx]

        K_train =np.array([572.41140, 0, 325.26110, 0, 573.57043, 242.04899, 0, 0, 1]).reshape((3, 3)) if dataset_name=='hinterstoisser' \
            else np.array([1075.65, 0, 720 / 2, 0, 1073.90, 540 / 2, 0, 0, 1]).reshape(3, 3)
        Radius_render_train = 700

        K_test = np.array(info_scene['cam_K']).reshape((3, 3))
        K00_ratio = K_test[0, 0] / K_train[0, 0]
        K11_ratio = K_test[1, 1] / K_train[1, 1]
        mean_K_ratio = np.mean([K00_ratio, K11_ratio])

        render_bb = codebook_obj_bbs[idx].squeeze()
        est_bb = info_dets[id_box]['obj_bb']
        diag_bb_ratio = np.linalg.norm(np.float32(render_bb[2:])) / np.linalg.norm(np.float32(est_bb[2:]))

        mm_tz = diag_bb_ratio * mean_K_ratio * Radius_render_train

        # object center in image plane (bb center =/= object center)
        center_obj_x_train = render_bb[0] + render_bb[2] / 2. - K_train[0, 2]
        center_obj_y_train = render_bb[1] + render_bb[3] / 2. - K_train[1, 2]

        center_obj_x_test = est_bb[0] + est_bb[2] / 2 - K_test[0, 2]
        center_obj_y_test = est_bb[1] + est_bb[3] / 2 - K_test[1, 2]


        cur_info_pose_est['est_tra'] = est_tra_w_tz(mm_tz,Radius_render_train,K_test,center_obj_x_test,center_obj_y_test,
                 K_train,center_obj_x_train,center_obj_y_train)
        cur_info_pose_est['est_rot']=rectify_rot(cur_info_pose_est['est_rot_cb'],cur_info_pose_est['est_tra'])

        max_mean_dist_factor=2.0
        if use_icp in ['refine_z','refine_all']:
            mm_tz,max_mean_dist=depth_refinement(test_depth_crops[0][id_box], model_reconst,cur_info_pose_est['est_rot'],
                                    cur_info_pose_est['est_tra'].flatten(), K_test, (W, H), max_mean_dist_factor=max_mean_dist_factor)#5.0)

            cur_info_pose_est['est_tra'] = est_tra_w_tz(mm_tz,Radius_render_train,K_test,center_obj_x_test,center_obj_y_test,
                 K_train,center_obj_x_train,center_obj_y_train)

            cur_info_pose_est['est_rot']=rectify_rot(cur_info_pose_est['est_rot_cb'],cur_info_pose_est['est_tra'])

        if use_icp=='refine_all':
            cur_info_pose_est['est_rot'], cur_info_pose_est['est_tra'], re_icp_fitness= rotation_error_icp(img_depth, model_o3d,  info_dets[id_box]['obj_bb'],
                                                                                           cur_info_pose_est['est_rot'].copy(),
                                                                                           cur_info_pose_est['est_tra'].copy().flatten(),
                                                                                           K_test.copy(),width=W, height=H, max_mean_dist=max_mean_dist,
                                                                                           max_mean_dist_factor=max_mean_dist_factor,
                                                                                           regist_error_threshold=3.0, fitness_threshold=0.7)


        cur_info_pose_est['R'] = cur_info_pose_est['est_rot']
        cur_info_pose_est['t'] = cur_info_pose_est['est_tra']
        if not est_err:
            cur_info_pose_est['vsd_correct'], cur_info_pose_est['is_visib'] = False, False
            cur_info_pose_est['vsd_err'], cur_info_pose_est['visib_portion'] = 0., 0.
            cur_info_pose_est['re_err'] = 0.
            cur_info_pose_est['adi_err'] = 0.0
        else:
            vsd_delta = 15
            vsd_tau = 20
            vsd_theta = 0.3
            vsd_cost = 'step'
            visib_portion = 0.1
            err = pose_error.vsd(R_est=cur_info_pose_est['est_rot'],
                                                t_est=cur_info_pose_est['est_tra'],
                                                R_gt=cur_info_pose_est['cam_R_m2c'],
                                                t_gt=cur_info_pose_est['cam_t_m2c'], model=model_reconst.copy(),
                                                depth_test=img_depth, K=info_scene['cam_K'], delta=vsd_delta,
                                                tau=vsd_tau, cost_type=vsd_cost)

            cur_info_pose_est['vsd_err'] = err
            cur_info_pose_est['vsd_correct'] = (err <= vsd_theta)

            cur_info_pose_est['is_visib'] = (cur_info_pose_est['visib_portion'] > visib_portion)
            cur_info_pose_est['re_err'] = 0.0#pose_error.re(cur_info_pose_est['est_rot'], cur_info_pose_est['cam_R_m2c'])

            cur_info_pose_est['adi_err'] = 0.0 # pose_error.add(cur_info_pose_est['est_rot'].astype(np.float32), cur_info_pose_est['est_tra'].astype(np.float32),
            # cur_info_pose_est['cam_R_m2c'].astype(np.float32), cur_info_pose_est['cam_t_m2c'].astype(np.float32),model_reconst)


            print('visib', cur_info_pose_est['visib_portion'], cur_info_pose_est['is_visib'])
            print('vsd  ', cur_info_pose_est['vsd_err'], cur_info_pose_est['vsd_correct'])


        if is_vis:#            
            ttag = ''
            if dataset_name=='hinterstoisser':
                mpath_cb_imgs = '../Edge-Network/embedding92232sline/{:02d}/imgs/{:05d}.png'
                mpath_ce_imgs = '../Edge-Network/embedding92232sline/{:02d}/in_edges2/{:05d}.png'###################################################
            else:
                mpath_cb_imgs = '../Edge-Network/embedding92232s/{:02d}/imgs/{:05d}.png'
                mpath_ce_imgs = '../Edge-Network/embedding92232s/{:02d}/in_edges2/{:05d}.png'###################################################
            query_bgr = cv2.resize(query_bgr, (128, 128))
            cv2.imwrite(path_vis_prefix + '_{:02d}_{:d}_{:s}BC.png'.format(model_id, id_box, ttag), query_bgr)
            query_depth = test_depth_crops[0][id_box].copy()
            query_depth = query_depth/np.float(np.max(query_depth))*200
            cv2.imwrite(path_vis_prefix + '_{:02d}_{:d}_{:s}BD.png'.format(model_id, id_box, ttag),query_depth.astype(np.uint8))
            cv2.imwrite(path_vis_prefix+'_{:02d}_{:d}_{:s}BE.png'.format(model_id,id_box,ttag),query_edge)
            bgr_est = cv2.imread(mpath_cb_imgs.format(model_id, idx))
            cv2.imwrite(path_vis_prefix + '_{:02d}_{:d}_{:s}ESTC.png'
                        .format(model_id, id_box, ttag, int(100 * cur_info_pose_est['vsd_err']),
                                int(100 * cur_info_pose_est['visib_portion']), int(cur_info_pose_est['re_err'])),bgr_est)
            edge_est=cv2.imread(mpath_ce_imgs.format(model_id,idx))
            cv2.imwrite(path_vis_prefix + '_{:02d}_{:d}_{:s}ESTE.png'
                        .format(model_id, id_box, ttag), edge_est)

        # change to axis and angle
        list_info_pose_est.append(cur_info_pose_est)  # Append current item to the list

    return list_info_pose_est