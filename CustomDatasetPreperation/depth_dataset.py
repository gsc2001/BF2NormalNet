import numpy as np
import os
import cv2
from numpy.core.fromnumeric import repeat
import argparse

fx = fy = 320
cx = 320
cy = 240

K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
K_inv = np.linalg.inv(K)


def process(rgb: np.ndarray, depth: np.ndarray, seg: np.ndarray):
    depth = depth.astype(np.float64)
    depth_vec = np.zeros((*depth.shape, 3))
    indices_row = np.arange(depth.shape[0])
    indices_r = np.hstack([indices_row.reshape(-1, 1)] * depth.shape[1]).ravel()
    indices_col = np.arange(depth.shape[1])
    indices_c = np.vstack([indices_col.reshape(1, -1)] * depth.shape[0]).ravel()
    ind_mat = np.vstack((indices_r, indices_c, np.ones(indices_r.shape[0])))
    depth_vec = K_inv @ ind_mat
    depth_vec /= np.linalg.norm(depth_vec, axis=0)
    depth_vec *= depth.ravel()
    depth_vec = depth_vec.T.reshape(*depth.shape, -1)
    normals = np.zeros((*depth.shape, 3))
    for i in range(1, depth.shape[0] - 1):
        for j in range(1, depth.shape[1] - 1):
            dzdx = (depth_vec[i + 1, j] - depth_vec[i - 1, j])
            dzdy = (depth_vec[i, j + 1] - depth_vec[i, j - 1])
            normal = np.cross(dzdx, dzdy)
            normals[i, j] = normal / np.linalg.norm(normal)

    seg_mask_2D = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY) > 0
    seg_mask = np.repeat(seg_mask_2D[:, :, np.newaxis], repeats=3, axis=2)
    normals_viz = ((normals * 0.5 + 0.5) * 255).astype('int')
    normals_viz = np.where(seg_mask, normals_viz, 0)
    normals = np.where(seg_mask, normals, 0)
    return (normals, normals_viz, seg_mask * 255)


def save(output_dir, _id, rgb, depth, normals, normals_viz, seg_mask):
    rgb_dir = os.path.join(output_dir, 'rgb')
    depth_dir = os.path.join(output_dir, 'depth')
    normals_dir = os.path.join(output_dir, 'normals')
    normals_viz_dir = os.path.join(output_dir, 'normals_viz')
    seg_mask_dir = os.path.join(output_dir, 'seg_mask')

    for _dir in [rgb_dir, depth_dir, normals_dir, normals_viz_dir, seg_mask_dir]:
        if not os.path.isdir(_dir):
            os.mkdir(_dir)

    cv2.imwrite(os.path.join(rgb_dir, f"{_id}.png"), rgb)
    np.save(os.path.join(depth_dir, f"{_id}.npy"), depth)
    np.save(os.path.join(normals_dir, f"{_id}.npy"), normals)
    cv2.imwrite(os.path.join(normals_viz_dir, f"{_id}.png"), normals_viz)
    cv2.imwrite(os.path.join(seg_mask_dir, f"{_id}.png"), seg_mask)


def read_pics(_id, in_dir):
    rgb = cv2.imread(os.path.join(in_dir, f'{_id}_scene.png'))
    seg = cv2.imread(os.path.join(in_dir, f'{_id}_seg.png'))
    depth = cv2.imread(os.path.join(in_dir, f'{_id}_depth.png'), -1)
    return (rgb, seg, depth)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output', '-o', default='output')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    for i in range(19):
        print("Processing ...", i)
        rgb, seg, depth = read_pics(i, args.input)
        if not os.path.isdir(args.output):
            os.mkdir(args.output)
        normals, normals_viz, seg_mask = process(rgb, depth, seg)
        save(args.output, i, rgb, depth, normals, normals_viz, seg_mask)
        print("Saved ...", i)


if __name__ == '__main__':
    main()
