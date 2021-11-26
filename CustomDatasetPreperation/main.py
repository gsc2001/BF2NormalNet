import os
import cv2
import numpy as np
from lu_vp_detect import VPDetection
import argparse
import glob
# everything in bgr

ids = np.load('ids.npy')

def get_picture(_id, _dir):
    rgb = cv2.imread(os.path.join(_dir, f'{_id}_scene.png'))
    seg = cv2.imread(os.path.join(_dir, f'{_id}_seg.png'))
    return (rgb, seg)

def vp_seg(rgb: np.ndarray, seg: np.ndarray, _id, out_dir):
    length_thresh = 0.1
    pp = None
    focal_length = rgb.shape[1]  / 2
    id_image = ids[seg[:,:,0], seg[:,:,1], seg[:,:,2]].copy()
    vpd = VPDetection(length_thresh, pp, focal_length)
    vps = vpd.find_vps(rgb)
    vps_2D = vpd.vps_2D
    x = vpd.create_debug_VP_image()
    lines = np.floor(vpd._VPDetection__lines).astype('int')
    lines = np.clip(lines,np.array([0,0,0,0]), np.array([639,479,639,479]))
    clusters = vpd._VPDetection__clusters
    line_ids = np.ones((lines.shape[0], 2), dtype=int)
    line_ids[:,0] = id_image[lines[:,1], lines[:,0]]
    line_ids[:,1] = id_image[lines[:,3], lines[:,2]]

    freq = np.zeros((300, 6), dtype=int)
    freq1 = np.zeros((300, 3), dtype=int)
    i = 0
    for i, cluster in enumerate(clusters):
        lines_ids_c = line_ids[cluster]
        eq_mask = lines_ids_c[:,0] == lines_ids_c[:,1]
        lines_ids_c = lines_ids_c[eq_mask]
        lines_c = lines[cluster][eq_mask]
        seg_mask = (lines_c[:,0] < vps_2D[0][0]) 
        lines_ids_c1 = lines_ids_c[seg_mask]
        lines_ids_c2 = lines_ids_c[seg_mask == False]
        un1, cnts1 = np.unique(lines_ids_c1[:,0], return_counts=True)
        un2, cnts2 = np.unique(lines_ids_c2[:,0], return_counts=True)
        un, cnt = np.unique(lines_ids_c[:,0], return_counts=True)
        freq[un1, 2*i] += cnts1
        freq[un2, 2*i + 1] += cnts2
        freq1[un, i] += cnt
        # print(lines_c)
    
    freq_min = freq1.argmin(axis=1)
    faces = freq_min.copy()
    faces[(faces == 1) & (freq[:,0] < freq[:,1])] = 3
    # Assign floor id

    faces += 1
    faces[0] = 0
    faces[freq1.max(axis=1) <= 20] = 0
    faces[39] = 3
    face_image = faces[id_image.astype('int')]
    normals_map = np.array([[0,0,0],[0,0,1],[1,0,0], [0,1,0],[-1,0,0]],dtype=np.float32)
    color = ((normals_map * 0.5 + 0.5) * 255).astype('int')
    color[0] = 0
    color = color[:,::-1]
    normals = normals_map[face_image].astype(np.float32)
    normal_viz = color[face_image]
    seg_mask = ((face_image > 0) * 255).astype(np.int16)
    seg = np.repeat(seg_mask[:,:, np.newaxis], repeats=3, axis=2)
    np.save(os.path.join(out_dir, 'normals', f'{_id}.npy'), normals)
    cv2.imwrite(os.path.join(out_dir, 'normals_viz', f'{_id}_normal.png'), normal_viz)
    cv2.imwrite(os.path.join(out_dir, 'seg', f'{_id}_seg.png'), seg_mask)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output', '-o', default='output')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    error = []
    for _id in range(2371):
        print('Started', _id)
        img, seg = get_picture(_id, args.input)
        try:
            vp_seg(img, seg, _id, args.output)
        except Exception:
            print('Error for', _id)
            error.append(_id)
        print('Finished', _id)
    
    np.save('error.npy', np.array(error))


if __name__ == '__main__':
    main()
