import numpy as np
import random

def get_forward_pix(focal_length, x, y, image_h, image_w, extend_pixel_cnt):
  if y is None:
    y = x[1]
    x = x[0]
  x -= image_h // 2
  y -= image_w // 2
  y_ = np.arctan(y / focal_length) * focal_length
  x_ = x * focal_length / np.sqrt(y ** 2 + focal_length ** 2)
  x_ += (image_h + extend_pixel_cnt) // 2
  y_ += (image_w + extend_pixel_cnt) // 2
  return x_, y_


def project_to_cylindral(focal_length, img, downsample=0):
    image_h, image_w = img.shape[:2]
    extend_pixel_cnt = 20

    backward_x, backward_y = get_backward_pix_map(focal_length, image_h, image_w, extend_pixel_cnt)
    project_img = np.zeros((image_h + extend_pixel_cnt, image_w + extend_pixel_cnt, 3))

    if downsample == 0:
        from tqdm import tqdm
        for i in tqdm(range(image_h + extend_pixel_cnt)):
            for j in range(image_w + extend_pixel_cnt):
                # Calculate only if backward coordinate is valid
                if backward_x[i, j] >= 0 and backward_x[i, j] <= image_h - 1 and backward_y[i, j] >= 0 and backward_y[i, j] <= image_w - 1:
                    project_img[i, j, :] = bilinear_interpolation(img, (backward_x[i, j], backward_y[i, j]))
    else:
        from tqdm import tqdm
        for i in tqdm(range(0, image_h + extend_pixel_cnt, downsample)):
            for j in range(0, image_w + extend_pixel_cnt, downsample):
                # Calculate only if backward coordinate is valid
                if backward_x[i, j] >= 0 and backward_x[i, j] <= image_h - 1 and backward_y[i, j] >= 0 and backward_y[i, j] <= image_w - 1:
                    project_img[i: i + downsample, j: j + downsample, :] = bilinear_interpolation(img, (backward_x[i, j], backward_y[i, j]))

    return project_img

def bilinear_interpolation(img, pix_location):
    x, y = pix_location
    try:
        pixel_value = img[int(np.floor(x)):int(np.floor(x)) + 2, int(np.floor(y)):int(np.floor(y)) + 2, :]
        pixel_weight = np.array([[1 - x + np.floor(x), x - np.floor(x)]]).T @ np.array([[1 - y + np.floor(y), y - np.floor(y)]])
        output_pixel = np.zeros((3))

        for i in range(3):
            output_pixel[i] = np.sum(pixel_value[:, :, i] * pixel_weight)
        
        return output_pixel
    except:
        print(pix_location)
        return img[int(np.floor(x)), int(np.floor(y)), :]
        
def get_backward_pix_map(focal_length, image_h, image_w, extend_pixel_cnt):
    x = np.arange(image_h + extend_pixel_cnt)
    y = np.arange(image_w + extend_pixel_cnt)
    y_proj, x_proj = np.meshgrid(y, x)
    x_proj -= (image_h + extend_pixel_cnt) // 2
    y_proj -= (image_w + extend_pixel_cnt) // 2

    y = np.tan(y_proj / focal_length) * focal_length
    x = x_proj / focal_length * np.sqrt(y ** 2 + focal_length ** 2)
    
    x += image_h // 2
    y += image_w // 2

    return x, y
    
# Estimate translation with RANSAC
def ransac(points1, points2, iter=100, choose_point=2, thres=5, visualize=False):
    # 2 dof
    length = len(points1)
    max_vote = 0

    for i in range(iter):
        t = np.zeros(2)  # t for transform
        idx = random.sample(range(len(points1)), choose_point)

        # build and solve the normal equation
        for j in range(choose_point):
            source_loc = points2[idx[j], :]
            target_loc = points1[idx[j], :]
            t = [t[k] + target_loc[k] - source_loc[k] for k in range(2)]
        t = [t[k] / choose_point for k in range(2)]

        # Calculate confidence with voting
        shift_points2 = points2 + np.array(t)
        dist = np.linalg.norm(points1 - shift_points2, axis=1)
        if max_vote < np.count_nonzero(dist < thres):
            inlier_idx = np.where(dist < thres)[0]
            max_vote = len(inlier_idx)

    # Calculate final transformation
    t = np.zeros(2)  # t for transform
    for j in inlier_idx:
        source_loc = points2[j, :]
        target_loc = points1[j, :]
        t = [t[k] + target_loc[k] - source_loc[k] for k in range(2)]
    t = [t[k] / len(inlier_idx) for k in range(2)]
  
    # Visualization
    if visualize:
        import matplotlib.pyplot as plt
        shift_points2 = points2 + np.array(t)
        plt.scatter(points1[:, 1], points1[:, 0])
        plt.scatter(shift_points2[:, 1], shift_points2[:, 0])
        plt.show()

    return t
  
# Image stitching
def stitch_together(project_img_lst, t):
    image_h, image_w = project_img_lst[0].shape[:2]
    origin = [[0, 0]]
    for t_ in t:
        origin.append([origin[-1][k] + t_[k] for k in range(2)])
    
    origin = np.array(origin).astype('int')
    min_idx = np.argmin(origin, axis=0)
    offset_0, offset_1 = origin[min_idx[0], 0], origin[min_idx[1], 1]
    origin[:, 0] -= offset_0
    origin[:, 1] -= offset_1

    max_value = np.max(origin.astype('int'), axis=0)
    panel = np.zeros((max_value[0] + image_h, max_value[1] + image_w, 3))

    stitch_idx = np.argsort(origin[:, 1])  # Stitch from left to right

    for idx in range(len(project_img_lst)):
        origin_ = origin[stitch_idx[idx]]

        if idx != 0:
            overlap_center = (origin_[1] + origin[stitch_idx[idx - 1], 1] + image_w) / 2
            half_width = 150
            for j in range(image_w):
                if idx >= 1 and origin_[1] + j < overlap_center - half_width:
                    pass
                elif idx >= 1 and origin_[1] + j >= overlap_center + half_width:
                    panel[origin_[0]:origin_[0] + image_h, origin_[1] + j, :] = project_img_lst[stitch_idx[idx]][:, j, :]

                elif idx >= 1 and origin_[1] + j >= overlap_center - half_width and origin_[1] + j <= overlap_center + half_width:
                    w_0 = (overlap_center + half_width - origin_[1] - j) / half_width / 2
                    w_1 = (origin_[1] + j - overlap_center + half_width) / half_width / 2
                
                    for i in range(image_h):
                        if np.sum(project_img_lst[stitch_idx[idx]][i, j, :]) != 0:
                            if np.sum(panel[origin_[0] + i, origin_[1] + j, :]) != 0:
                                panel[origin_[0] + i, origin_[1] + j, :] = panel[origin_[0] + i, origin_[1] + j, :] * w_0 + project_img_lst[stitch_idx[idx]][i, j, :] * w_1
                            else:
                                panel[origin_[0] + i, origin_[1] + j, :] = project_img_lst[stitch_idx[idx]][i, j, :]
        else:
            # Index = 0
            panel[origin_[0]: origin_[0] + image_h, origin_[1]: origin_[1] + image_w, :] = project_img_lst[stitch_idx[idx]]
    
    # Warp out the black pixel region
    left_margin = 0
    while np.min(np.sum(panel[:, left_margin, :], axis=0)) == 0:
        left_margin += 1
    right_margin = max_value[1] + image_w - 1
    while np.min(np.sum(panel[:, right_margin, :], axis=0)) == 0:
        right_margin -= 1
    panel = panel[:, left_margin:right_margin, :]
    top_margin = 0
    while np.min(np.sum(panel[top_margin, :, :], axis=1)) == 0:
        top_margin += 1
    low_margin = max_value[0] + image_h - 1
    while np.min(np.sum(panel[low_margin, :, :], axis=1)) == 0:
        low_margin -= 1
    panel = panel[top_margin:low_margin, :, :]

    return panel

def stitch_img(img_lst, focal_length_lst, matched_keypoint):
    image_h, image_w = img_lst[0].shape[:2]
    extend_pixel_cnt = 20

    # Project to cylindral
    print("Projecting image to cylindral plane ...")
    proj_img_lst = []
    for img_idx in range(len(img_lst)):
        proj_img = project_to_cylindral(focal_length_lst[img_idx], img_lst[img_idx])
        proj_img_lst.append(proj_img)

    # Adopt RANSAC to estimate translation
    print("Performing RANSAC for translation estimation ...")
    translation = []
    for idx in range(len(img_lst) - 1):
        points1, points2 = matched_keypoint[0][idx], matched_keypoint[1][idx]

        for i in range(len(points1)):
            points1[i, :] = get_forward_pix(focal_length_lst[idx], points1[i, 0], points1[i, 1], image_h, image_w, extend_pixel_cnt)
            points2[i, :] = get_forward_pix(focal_length_lst[idx + 1], points2[i, 0], points2[i, 1], image_h, image_w, extend_pixel_cnt)

        translation.append(ransac(points1, points2, thres=80, visualize=False))

    # Image Stitching and warping
    print("Final stitching ...")
    result = stitch_together(proj_img_lst, translation)
    return result