# -*- coding: utf-8 -*-
# This is to sample rotations for the inference template poses \bar(R), or the training reference rotations R_c used in geometric prior
# For each sampled rotation, its rendered BGR image, edgemap, 2D boundingbox, and rotations will be generated and preserved
import numpy as np
import progressbar
import cv2
import os,sys

from pysixd_stuff.pysixd import inout
from pysixd_stuff.pysixd import renderer_vt
from pysixd_stuff.pysixd import view_sampler
import data_utils


def generate_codebook_imgs(path_model,dir_imgs,dir_edges, path_obj_bbs,path_rot,render_dims,cam_K,depth_scale=1.,texture_img=None,start_end=None):
    if not os.path.exists(dir_imgs):
        os.makedirs(dir_imgs)
    if not os.path.exists(dir_edges):
        os.makedirs(dir_edges)

    view_Rs=data_utils.viewsphere_for_embedding_v2(num_sample_views=2500,num_cyclo=36,use_hinter=True)
    #data_utils.viewsphere_for_embedding_v2(num_sample_views=2500,num_cyclo=36,use_hinter=True)
    #For reference R_c: view_Rs=data_utils.viewsphere_for_embedding_v2(num_sample_views=400,num_cyclo=20,use_hinter=False)

    #num_sample_views: number of samples on the unit sphere
    #num_cyclo: number of samples regarding inner-plane rotations
    #use_hinter=True: hinter sampling; use_hinter=False: fabonicci sampling


    np.savez(path_rot,rots=view_Rs)
    embedding_size = view_Rs.shape[0]

    out_shape=(128,128,3)
    if start_end and start_end[0]!=0:
        obj_bbs=np.load(path_obj_bbs+'.npy')
    else:
        obj_bbs = np.empty((embedding_size, 4))
    print('Creating embedding ..')
    bar = progressbar.ProgressBar(
        maxval=embedding_size,
        widgets=[' [', progressbar.Timer(), ' | ', progressbar.Counter('%0{}d / {}'.format(len(str(embedding_size)), embedding_size)), ' ] ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') ']
    )
    bar.start()
    K = np.array(cam_K).reshape(3,3)

    clip_near = float(10)
    clip_far = float(10000)
    pad_factor = float(1.2)

    t = np.array([0, 0, float(700)])
    model = inout.load_ply(path_model)
    model['pts']*=depth_scale

    if start_end is None:
        search_range=range(0,view_Rs.shape[0])
    else:
        start_end[1]=min(start_end[1],view_Rs.shape[0])
        search_range=range(start_end[0],start_end[1])

    for i in search_range:
        bar.update(i)
        R=view_Rs[i]
        rgb_y, depth_y = renderer_vt.render_phong(model, render_dims, K.copy(), R, t, clip_near=clip_near,
                                              clip_far=clip_far,texture=texture_img, mode='rgb+depth', random_light=False)
        ys, xs = np.nonzero(depth_y > 0)
        obj_bb = view_sampler.calc_2d_bbox(xs, ys, render_dims)

        obj_bbs[i] = obj_bb
        bgr_y = rgb_y.copy()
        for cc in range(0, 3):
            bgr_y[:, : ,cc] = rgb_y[:, :,2 - cc]

        resized_bgr_y = data_utils.extract_square_patch(bgr_y, obj_bb, pad_factor,resize=out_shape[:2],interpolation = cv2.INTER_NEAREST)
        resized_bgr_y_edge=cv2.Canny(resized_bgr_y,50,150)
        cv2.imwrite(os.path.join(dir_edges,'{:05d}.png'.format(i)),resized_bgr_y_edge)
        cv2.imwrite(os.path.join(dir_imgs,'{:05d}.png'.format(i)),resized_bgr_y)
    bar.finish()
    np.save(path_obj_bbs,obj_bbs)


if __name__=='__main__':
	#Rendered by batch, with batch size=50
    obj_id=int(sys.argv[1])
    bid=int(sys.argv[2])
    batch_size=50
    path_model ='./ws/meshes/obj_{:02d}.ply'.format(obj_id)#Path of the 3D mesh ply file
    dir_out='./embedding92232s/{:02d}/'.format(obj_id) #dir to save the rendered images, edgemaps, 2D bounding box, sampled rotations
    dir_imgs = os.path.join(dir_out,'imgs')
    dir_edges= os.path.join(dir_out,'in_edges2')
    path_obj_bbs= os.path.join(dir_out,'obj_bbs')
    path_rot=os.path.join(dir_out,'rot_infos')

    path_texture=None
    texture_img_rgb=None
    if path_texture:
        texture_img_bgr=cv2.imread(path_texture['{:02d}'.format(obj_id)])
        texture_img_rgb=texture_img_bgr[:,:,2::-1]

    generate_codebook_imgs(path_model=path_model,
                           dir_imgs=dir_imgs,
                           dir_edges=dir_edges,
                           path_obj_bbs=path_obj_bbs,
                           path_rot=path_rot,
                           render_dims=(720,540),
                           cam_K=[1075.65, 0, 720 / 2, 0, 1073.90, 540 / 2, 0, 0, 1],
                           depth_scale=1.,
                           texture_img=texture_img_rgb,
                           start_end=[bid*batch_size,(bid+1)*batch_size])
