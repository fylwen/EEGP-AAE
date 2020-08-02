import numpy as np
import cv2
from pysixd_stuff.pysixd import view_sampler
import ruamel.yaml as yaml
import math
from pysixd_stuff.pysixd import transform, inout
from scipy.special import gamma,sph_harm,eval_gegenbauer,factorial,gegenbauer

def generate_scene_crops(test_imgs, test_depth_imgs, bboxes,pad_factor=1.2,W_AE=128,H_AE=128,return_non_resized=False):
    estimate_bbs = False
    icp = not (test_depth_imgs is None)

    #W_AE = 128
    #H_AE = 128

    resized_test_img_crops, test_img_depth_crops, bb_scores, bb_vis, bbs= {}, {}, {}, {}, {}
    if return_non_resized:
        test_img_crops,bbs_crops={},{}

    H, W = test_imgs.shape[1:3]
    for view, img in enumerate(test_imgs):
        if icp:
            depth = test_depth_imgs[view]
            test_img_depth_crops[view] = []#{}

        resized_test_img_crops[view], bb_scores[view], bb_vis[view], bbs[view]= [],[],[],[]#{}, {}, {}, {}
        if return_non_resized:
            test_img_crops[view],bbs_crops[view]=[],[]

        if len(bboxes[view]) > 0:
            for bbox_idx, bbox in enumerate(bboxes[view]):
                bb = np.array(bbox['obj_bb'])
                bb_score = bbox['score'] if estimate_bbs else 1.0
                vis_frac = None #if estimate_bbs else visib_gt[view][bbox_idx]['visib_fract']

                x, y, w, h = bb

                size = int(np.maximum(h, w) * pad_factor)
                left = int(np.max([x + w / 2 - size / 2, 0]))
                right = int(np.min([x + w / 2 + size / 2, W]))
                top = int(np.max([y + h / 2 - size / 2, 0]))
                bottom = int(np.min([y + h / 2 + size / 2, H]))

                crop = img[top:bottom, left:right].copy()
                # print 'Original Crop Size: ', crop.shape
                resized_crop = cv2.resize(crop, (H_AE, W_AE))

                if icp:
                    depth_crop = depth[top:bottom, left:right]
                    test_img_depth_crops[view].append(depth_crop)
                    #test_img_depth_crops[view].setdefault(obj_id, []).append(depth_crop)

                resized_test_img_crops[view].append(resized_crop)
                bb_scores[view].append(bb_score)
                bb_vis[view].append(vis_frac)
                bbs[view].append(bb)
                if return_non_resized:
                    test_img_crops[view].append(crop)
                    bbs_crops[view].append([left,top,right-left,bottom-top])
    if return_non_resized:
        return resized_test_img_crops, test_img_depth_crops, bbs, bb_scores, bb_vis, test_img_crops, bbs_crops
    else:
        return resized_test_img_crops, test_img_depth_crops, bbs, bb_scores, bb_vis

def tiles(batch, rows, cols, spacing_x=0, spacing_y=0, scale=1.0):
    if batch.ndim == 4:
        N, H, W, C = batch.shape
    elif batch.ndim == 3:
        N, H, W = batch.shape
        C = 1
    else:
        raise ValueError('Invalid batch shape: {}'.format(batch.shape))

    H = int(H*scale)
    W = int(W*scale)
    img = np.ones((rows*H+(rows-1)*spacing_y, cols*W+(cols-1)*spacing_x, 3))
    i = 0
    for row in range(0,rows):
        for col in range(0,cols):
            start_y = row*(H+spacing_y)
            end_y = start_y + H
            start_x = col*(W+spacing_x)
            end_x = start_x + W
            if i < N:
                if C > 1:
                    img[start_y:end_y,start_x:end_x,:] = cv2.resize(batch[i], (W,H))
                else:
                    for _c in range(0,3):
                        img[start_y:end_y,start_x:end_x,_c] = cv2.resize(batch[i], (W,H))
            i += 1
    return img

def extract_square_patch(scene_img, bb_xywh, pad_factor, resize=(128, 128), interpolation=cv2.INTER_NEAREST,return_non_resized=False):

    x, y, w, h = np.array(bb_xywh).astype(np.int32)
    size = int(np.maximum(h, w) * pad_factor)

    left = int(np.maximum(x + w / 2 - size / 2, 0))
    right = int(np.minimum(x + w / 2 + size / 2, scene_img.shape[1]))
    top = int(np.maximum(y + h / 2 - size / 2, 0))
    bottom = int(np.minimum(y + h / 2 + size / 2, scene_img.shape[0]))

    scene_crop = scene_img[top:bottom, left:right]
    resized_scene_crop = cv2.resize(scene_crop, resize, interpolation=interpolation)
    if return_non_resized:
        return resized_scene_crop,scene_crop
    else:
        return resized_scene_crop

def viewsphere_for_embedding(num_sample_views=1000,num_cyclo=36,render_dist=700.0):
    azimuth_range = (0, 2 * np.pi)
    elev_range = (-0.5 * np.pi, 0.5 * np.pi)
    views, _ = view_sampler.sample_views(
        num_sample_views,
        render_dist,
        azimuth_range,
        elev_range
    )
    Rs = np.empty( (len(views)*num_cyclo, 3, 3) )
    i = 0
    for view in views:
        for cyclo in np.linspace(0, 2.*np.pi, num_cyclo):
            rot_z = np.array([[np.cos(-cyclo), -np.sin(-cyclo), 0], [np.sin(-cyclo), np.cos(-cyclo), 0], [0, 0, 1]])
            Rs[i,:,:] = rot_z.dot(view['R'])
            i += 1
    print('Rs shape: ',Rs.shape)
    return Rs

def viewsphere_for_embedding_v2(num_sample_views=1000,num_cyclo=36,render_dist=700.0,use_hinter=True):
    azimuth_range = (0, 2 * np.pi)
    elev_range = (-0.5 * np.pi, 0.5 * np.pi)
    views, _ = view_sampler.sample_views(
        num_sample_views,
        render_dist,
        azimuth_range,
        elev_range,
        use_hinter
    )
    Rs = np.empty( (len(views)*num_cyclo, 3, 3) )
    i = 0
    cyclo_space = np.linspace(0, 2. * np.pi, num_cyclo + 1)[:-1]
    #print('cyclo_space',cyclo_space)
    for view in views:
        for cyclo in cyclo_space:#np.linspace(0, 2.*np.pi, num_cyclo+1):
            rot_z = np.array([[np.cos(-cyclo), -np.sin(-cyclo), 0], [np.sin(-cyclo), np.cos(-cyclo), 0], [0, 0, 1]])
            Rs[i,:,:] = rot_z.dot(view['R'])
            i += 1
    #print('Rs shape: ',Rs.shape)
    return Rs

def save_errors(path, ests, run_time=-1):
    with open(path, 'w') as f:
        txt = 'run_time: ' + str(run_time) + '\n'  # The first line contains run time
        txt += 'ests:\n'
        line_tpl = '- {{score: {:.8f}, is_visib: {:s}, vsd_correct: {:s}, re_err: {:.8f}, vsd_err: {:.8f}, adi_err: {:.8f}, visib_portion: {:.8f},' \
                   'R: [' + ', '.join(['{:.8f}'] * 9) + '], t: [' + ', '.join(['{:.8f}'] * 3) + ']}}\n'

        # print(line_tpl)
        txt = ''  # ,
        for d_item in ests:
            Rt = d_item['R'].astype(np.float32).flatten().tolist() + d_item['t'].flatten().tolist()
            # print(Rt)
            txt += line_tpl.format(d_item['score'],
                                   ('True' if d_item['is_visib'] else 'False'),
                                   ('True' if d_item['vsd_correct'] else 'False'),
                                   d_item['re_err'],
                                   d_item['vsd_err'],
                                   d_item['adi_err'],
                                   d_item['visib_portion'], *Rt)
        f.write(txt)

def viewsphere_for_embedding_from_file(obj_views,num_cyclo=36,render_dist=700.0,azimuth_range=(0, 2 * math.pi),
                 elev_range=(-0.5 * math.pi, 0.5 * math.pi)):
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(obj_views)
    pts=mesh.vertices
    pts *= np.reshape(render_dist / np.linalg.norm(pts, axis=1), (pts.shape[0], 1))
    xxx=render_dist / np.linalg.norm(pts, axis=1)
    print('check',xxx.max,xxx.min)
    views = []
    for pt in pts:
        # Azimuth from (0, 2 * pi)
        azimuth = math.atan2(pt[1], pt[0])
        if azimuth < 0:
            azimuth += 2.0 * math.pi

        # Elevation from (-0.5 * pi, 0.5 * pi)
        a = np.linalg.norm(pt)
        b = np.linalg.norm([pt[0], pt[1], 0])
        elev = math.acos(b / a)
        if pt[2] < 0:
            elev = -elev

        # if hemisphere and (pt[2] < 0 or pt[0] < 0 or pt[1] < 0):
        if not (azimuth_range[0] <= azimuth <= azimuth_range[1] and
                elev_range[0] <= elev <= elev_range[1]):
            continue

        # Rotation matrix
        f = -np.array(pt) # Forward direction
        f /= np.linalg.norm(f)
        u = np.array([0.0, 0.0, 1.0]) # Up direction
        s = np.cross(f, u) # Side direction
        if np.count_nonzero(s) == 0:
            # f and u are parallel, i.e. we are looking along or against Z axis
            s = np.array([1.0, 0.0, 0.0])
        s /= np.linalg.norm(s)
        u = np.cross(s, f) # Recompute up
        R = np.array([[s[0], s[1], s[2]],
                      [u[0], u[1], u[2]],
                      [-f[0], -f[1], -f[2]]])

        # Convert from OpenGL to OpenCV coordinate system
        R_yz_flip = transform.rotation_matrix(math.pi, [1, 0, 0])[:3, :3]
        R = R_yz_flip.dot(R)

        # Translation vector
        t = -R.dot(np.array(pt).reshape((3, 1)))

        views.append({'R': R, 't': t})

    Rs = np.empty( (len(views)*num_cyclo, 3, 3) )
    i = 0
    cyclo_space = np.linspace(0, 2. * np.pi, num_cyclo + 1)[:-1]
    print('cyclo_space',cyclo_space)
    for view in views:
        for cyclo in cyclo_space:#np.linspace(0, 2.*np.pi, num_cyclo+1):
            rot_z = np.array([[np.cos(-cyclo), -np.sin(-cyclo), 0], [np.sin(-cyclo), np.cos(-cyclo), 0], [0, 0, 1]])
            Rs[i,:,:] = rot_z.dot(view['R'])
            i += 1
    print('Rs shape: ',Rs.shape)
    return Rs

def viewsphere_for_embedding_euler(num_cx,num_cy,num_cz):
    num_views=num_cx*num_cy*num_cz
    Rs = np.empty( (num_views, 3, 3) )
    euler_angles=np.empty((num_views,3))
    i = 0
    for cyclo_x in np.linspace(-0.5*np.pi,0.5*np.pi,num_cx):
        rot_x=np.array([[1,0,0],[0,np.cos(cyclo_x),-np.sin(cyclo_x)],[0,np.sin(cyclo_x),np.cos(cyclo_x)]],dtype=np.float32)
        for cyclo_y in np.linspace(0,2.*np.pi,num_cy+1)[:-1]:
            rot_y=np.array([[np.cos(cyclo_y),0,np.sin(cyclo_y)],[0,1,0],[-np.sin(cyclo_y),0,np.cos(cyclo_y)]],dtype=np.float32)
            for cyclo_z in np.linspace(0,2.*np.pi,num_cz+1)[:-1]:
                rot_z=np.array([[np.cos(cyclo_z),-np.sin(cyclo_z),0],[np.sin(cyclo_z),np.cos(cyclo_z),0],[0,0,1]],dtype=np.float32)
                Rs[i,:,:]=np.dot(rot_z,np.dot(rot_y,rot_x))
                euler_angles[i,:]=[cyclo_x,cyclo_y,cyclo_z]
                i+=1

    print('Rs shape: ',Rs.shape)
    return Rs,euler_angles


