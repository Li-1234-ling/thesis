import os, glob
import numpy as np
import cv2

# ===================== 路径配置 =====================
COLMAP_DIR = r"C:\Users\Administrator\Desktop\uni\MasterArbeit\Li\finaltest\test7\colmap\sparse_txt"
POSES_W2C_NPY = r"C:\Users\Administrator\Desktop\uni\MasterArbeit\Li\finaltest\test7\colmap\sparse_npy\poses_world2cam.npy"
UNDIST_RGB_GLOB = r"C:\Users\Administrator\Desktop\uni\MasterArbeit\Li\finaltest\test7\images_sub\frame_*.png"
OUT_DIR = r"C:\Users\Administrator\Desktop\uni\MasterArbeit\Li\finaltest\test7\output_homo_multi_modify_best"

# ====== 手动注册多个目标 ======
# 每个目标都手动给：
# 1) 它完整出现的一帧
# 2) 该帧对应的 mask
# 3) 该帧对应的 clean_ref
OBJECTS = [
    {
        "name": "obj1",
        "ref_frame_name": "frame_000001.png",
        "ref_mask_path": r"C:\Users\Administrator\Desktop\uni\MasterArbeit\Li\finaltest\test7\mask_obj1.png",
        "clean_ref_path": r"C:\Users\Administrator\Desktop\uni\MasterArbeit\Li\finaltest\test7\clean_ref_obj1.png",
    },
    {
        "name": "obj2",
        "ref_frame_name": "frame_000673.png",
        "ref_mask_path": r"C:\Users\Administrator\Desktop\uni\MasterArbeit\Li\finaltest\test7\mask_obj2.png",
        "clean_ref_path": r"C:\Users\Administrator\Desktop\uni\MasterArbeit\Li\finaltest\test7\clean_ref_obj2.png",
    },
    {
        "name": "obj3",
        "ref_frame_name": "frame_001174.png",
        "ref_mask_path": r"C:\Users\Administrator\Desktop\uni\MasterArbeit\Li\finaltest\test7\mask_obj3.png",
        "clean_ref_path": r"C:\Users\Administrator\Desktop\uni\MasterArbeit\Li\finaltest\test7\clean_ref_obj3.png",
    },
    {
        "name": "obj4",
        "ref_frame_name": "frame_001666.png",
        "ref_mask_path": r"C:\Users\Administrator\Desktop\uni\MasterArbeit\Li\finaltest\test7\mask_obj4.png",
        "clean_ref_path": r"C:\Users\Administrator\Desktop\uni\MasterArbeit\Li\finaltest\test7\clean_ref_obj4.png",
    },
    {
        "name": "obj5",
        "ref_frame_name": "frame_001990.png",
        "ref_mask_path": r"C:\Users\Administrator\Desktop\uni\MasterArbeit\Li\finaltest\test7\mask_obj5.png",
        "clean_ref_path": r"C:\Users\Administrator\Desktop\uni\MasterArbeit\Li\finaltest\test7\clean_ref_obj5.png",
    },
]
# ====================================================

# ====== undist 坐标系下的内参 ======
K_undist = np.array([
    [855.77871693204497, 0.0, 640.0],
    [0.0, 861.25545338322968, 360.0],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

# ====== 一些阈值，可调 ======
PLANE_INLIER_TH = 0.01 
MIN_OBJ_POINTS = 30 
MIN_SHARED_POINTS = 8
MIN_SHARED_POINTS_EDGE = 2
MIN_SHARED_POINTS_BORDER_TRUST = 1
MAX_REPROJ_ERR = 12.0
MAX_REPROJ_ERR_EDGE = 25.0
MIN_SUPPORT_AREA = 50
TRUST_WARPED_MASK_ON_BORDER = True
# 这里是关键：将过滤深度设为一个极小但非零的值。
# 这个米级（COLMAP单位）的阈值是为了挡住“相机正后方”的物体
MIN_FRONT_Z = 0.001


def read_images_txt(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith("#")]
    i = 0
    while i < len(lines):
        header = lines[i].split()
        image_id = int(header[0])
        qvec = np.array(list(map(float, header[1:5])), dtype=np.float64)
        tvec = np.array(list(map(float, header[5:8])), dtype=np.float64)
        cam_id = int(header[8])
        name = header[9]
        pts_line = lines[i + 1].split()
        xys, p3d_ids = [], []
        for j in range(0, len(pts_line), 3):
            x = float(pts_line[j])
            y = float(pts_line[j + 1])
            pid = int(pts_line[j + 2])
            xys.append([x, y])
            p3d_ids.append(pid)
        data[name] = dict(
            image_id=image_id,
            cam_id=cam_id,
            qvec=qvec,
            tvec=tvec,
            xys=np.array(xys, dtype=np.float64),
            p3d_ids=np.array(p3d_ids, dtype=np.int64),
        )
        i += 2
    return data


def read_points3D_txt(path):
    pts = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            pid = int(parts[0])
            xyz = np.array(list(map(float, parts[1:4])), dtype=np.float64)
            pts[pid] = xyz
    return pts


def fit_plane_ransac(points, iters=3000, inlier_th=0.01):
    assert points.shape[0] >= 3
    rng = np.random.default_rng(0)
    M = points.shape[0]
    best_inliers = None

    for _ in range(iters):
        idx = rng.choice(M, size=3, replace=False)
        p0, p1, p2 = points[idx]
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        d = -np.dot(n, p0)
        dist = np.abs(points @ n + d)
        inliers = dist < inlier_th
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers

    if best_inliers is None or best_inliers.sum() < 3:
        raise RuntimeError("RANSAC 平面拟合失败")

    P = points[best_inliers]
    centroid = P.mean(axis=0)
    Q = P - centroid
    _, _, vh = np.linalg.svd(Q, full_matrices=False)
    n = vh[-1]
    n = n / np.linalg.norm(n)
    d = -np.dot(n, centroid)
    return n, d, best_inliers


def load_poses_w2c(path):
    poses = np.load(path, allow_pickle=True)
    if isinstance(poses, np.ndarray) and poses.shape == () and poses.dtype == object:
        poses = poses.item()
    if isinstance(poses, dict):
        keys = sorted(poses.keys())
        poses = [poses[k] for k in keys]
    if isinstance(poses, (list, tuple)):
        poses = np.stack([np.array(T, dtype=np.float64) for T in poses], axis=0)
    elif isinstance(poses, np.ndarray) and poses.dtype == object:
        poses = np.stack([np.array(T, dtype=np.float64) for T in poses.tolist()], axis=0)
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim == 2 and poses.shape[1] == 16:
        poses = poses.reshape(-1, 4, 4)
    return poses


def invert_Tcw(Tcw):
    R = Tcw[:3, :3]
    t = Tcw[:3, 3]
    Twc = np.eye(4, dtype=np.float64)
    Twc[:3, :3] = R.T
    Twc[:3, 3] = -R.T @ t
    return Twc


# def plane_world_to_cam(n_w, d_w, Tcw):
#     """
#     world plane: n_w^T X + d_w = 0
#     convert to camera coords of this Tcw (world->cam):
#     n_c = Rcw * n_w
#     d_c = n_w^T Cw + d_w, where Cw = camera center in world
#     """
#     Rcw = Tcw[:3, :3]
#     tcw = Tcw[:3, 3]
#     Cw = -Rcw.T @ tcw
#     n_c = Rcw @ n_w
#     d_c = float(n_w @ Cw + d_w)
#     return n_c, d_c

def plane_world_to_cam(n_w, d_w, Tcw):
    Rcw = Tcw[:3, :3].astype(np.float64)
    tcw = Tcw[:3, 3].astype(np.float64)
    
    # 1. 计算法线在相机系下的方向
    n_c = Rcw @ n_w
    
    # 2. 计算平面到相机中心的垂直距离 d_c
    # 公式：d_c = |n_w^T * Cw + d_w| / |n_w|
    # 由于 n_w 是单位向量，简化为：
    Cw = -Rcw.T @ tcw
    d_c = n_w @ Cw + d_w
    
    # 【核心修正】：如果 d_c 是负的，说明法线向量 n_w 指向了相机背面。
    # 我们必须翻转 n_c 和 d_c，确保对于相机来说，平面参数是统一的。
    if d_c < 0:
        n_c = -n_c
        d_c = -d_c
        
    return n_c, d_c


def relative_ref_to_cur(Tcw_ref, Tcw_cur):
    """
    X_cur = R * X_ref + t
    where (R,t) from ref-cam coords to cur-cam coords
    """
    Twc_ref = invert_Tcw(Tcw_ref)
    T_cur_ref = Tcw_cur @ Twc_ref
    R = T_cur_ref[:3, :3]
    t = T_cur_ref[:3, 3]
    return R, t


def homography_from_plane(K, R, t, n_ref, d_ref):
    """
    H = K (R - t n^T / d) K^-1
    注意 d_ref 不能太接近 0
    """
    Kinv = np.linalg.inv(K)
    denom = float(d_ref)
    if abs(denom) < 1e-8:
        raise RuntimeError(f"d_ref 太小，无法稳定计算 H: {denom}")
    H = K @ (R - (t.reshape(3, 1) @ n_ref.reshape(1, 3)) / denom) @ Kinv
    return H


def order_corners(pts):
    """
    输入 4x2，无序点
    输出顺序：tl, tr, br, bl
    """
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def extract_board_corners_from_mask(mask):
    """
    从参考帧 mask 中提取目标四角
    假设 mask 里目标区域是最大的主要连通域
    """
    mask_bin = (mask > 127).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("参考帧 mask 中没有找到轮廓")

    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)

    approx = None
    for eps_ratio in [0.01, 0.02, 0.03, 0.05, 0.08]:
        poly = cv2.approxPolyDP(cnt, eps_ratio * peri, True)
        if len(poly) == 4:
            approx = poly[:, 0, :]
            break

    if approx is None:
        rect = cv2.minAreaRect(cnt)
        approx = cv2.boxPoints(rect)

    corners = order_corners(approx)
    return corners


def warp_points(H, pts):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H)
    return out.reshape(-1, 2)


def build_pid_xy_map(img_info):
    pid_to_xy = {}
    for xy, pid in zip(img_info["xys"], img_info["p3d_ids"]):
        if pid >= 0:
            pid_to_xy[int(pid)] = np.asarray(xy, dtype=np.float64)
    return pid_to_xy


def mask_touches_image_border(mask):
    return bool(
        np.any(mask[0, :] > 127) or
        np.any(mask[-1, :] > 127) or
        np.any(mask[:, 0] > 127) or
        np.any(mask[:, -1] > 127)
    )


def polygon_to_mask(corners, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    poly = np.round(corners).astype(np.int32)
    cv2.fillConvexPoly(mask, poly, 255)
    return mask


def draw_corners(img, corners, color=(0, 255, 0), radius=6):
    vis = img.copy()
    corners_i = np.round(corners).astype(np.int32)
    for p in corners_i:
        cv2.circle(vis, tuple(p), radius, color, -1, lineType=cv2.LINE_AA)
    for i in range(4):
        p1 = tuple(corners_i[i])
        p2 = tuple(corners_i[(i + 1) % 4])
        cv2.line(vis, p1, p2, color, 2, lineType=cv2.LINE_AA)
    return vis


def build_registered_object(obj_cfg, frames, imgs, pts3d, poses, out_dir):
    ref_frame_name = obj_cfg["ref_frame_name"]
    ref_mask_path = obj_cfg["ref_mask_path"]
    clean_ref_path = obj_cfg["clean_ref_path"]
    obj_name = obj_cfg["name"]

    name2idx = {os.path.basename(p): i for i, p in enumerate(frames)}
    if ref_frame_name not in name2idx:
        raise RuntimeError(f"[{obj_name}] 找不到参考帧 {ref_frame_name}")

    base2key = {os.path.basename(k): k for k in imgs.keys()}
    if ref_frame_name not in base2key:
        raise RuntimeError(f"[{obj_name}] images.txt 里找不到 {ref_frame_name}")

    ref_idx = name2idx[ref_frame_name]
    ref_key = base2key[ref_frame_name]
    ref_info = imgs[ref_key]

    ref_img = cv2.imread(frames[ref_idx], cv2.IMREAD_COLOR)
    ref_mask = cv2.imread(ref_mask_path, cv2.IMREAD_GRAYSCALE)
    clean_ref = cv2.imread(clean_ref_path, cv2.IMREAD_COLOR)

    if ref_img is None:
        raise RuntimeError(f"[{obj_name}] 无法读取参考帧图像: {frames[ref_idx]}")
    if ref_mask is None:
        raise RuntimeError(f"[{obj_name}] 无法读取参考 mask: {ref_mask_path}")
    if clean_ref is None:
        raise RuntimeError(f"[{obj_name}] 无法读取 clean_ref: {clean_ref_path}")

    H_img, W_img = ref_img.shape[:2]

    if ref_mask.shape[:2] != (H_img, W_img):
        print(f"[{obj_name}] resize ref_mask from {ref_mask.shape[:2]} to {(H_img, W_img)}")
        ref_mask = cv2.resize(ref_mask, (W_img, H_img), interpolation=cv2.INTER_NEAREST)

    if clean_ref.shape[:2] != (H_img, W_img):
        print(f"[{obj_name}] resize clean_ref from {clean_ref.shape[:2]} to {(H_img, W_img)}")
        clean_ref = cv2.resize(clean_ref, (W_img, H_img), interpolation=cv2.INTER_LINEAR)

    # 1) 用该对象自己的 mask 从参考帧挑 3D 点
    obj_pts = []
    obj_ref_xys = []
    obj_pids = []
    for (x, y), pid in zip(ref_info["xys"], ref_info["p3d_ids"]):
        if pid < 0 or pid not in pts3d:
            continue
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W_img and 0 <= yi < H_img and ref_mask[yi, xi] > 127:
            obj_pts.append(pts3d[pid])
            obj_ref_xys.append([x, y])
            obj_pids.append(pid)

    obj_pts = np.array(obj_pts, dtype=np.float64)
    obj_ref_xys = np.array(obj_ref_xys, dtype=np.float64)
    obj_pids = np.array(obj_pids, dtype=np.int64)
    print(f"[{obj_name}] 3D points in mask: {obj_pts.shape[0]}")

    if obj_pts.shape[0] < 30:
        raise RuntimeError(
            f"[{obj_name}] mask 内 3D 点太少：{obj_pts.shape[0]}，"
            f"换一个更完整参考帧或更好的 mask"
        )

    # 2) 拟合该对象自己的世界平面
    n_w, d_w, inl = fit_plane_ransac(obj_pts, iters=3000, inlier_th=0.01)
    print(f"[{obj_name}] plane(world): n={n_w}, d={d_w}, inliers={int(inl.sum())}/{obj_pts.shape[0]}")
    support_ref_pts = obj_ref_xys[inl]
    support_pids = obj_pids[inl]
    if support_ref_pts.shape[0] < MIN_SHARED_POINTS:
        raise RuntimeError(f"[{obj_name}] 可用于验证的平面内点太少：{support_ref_pts.shape[0]}")

    # 3) 提取该对象自己的四角
    ref_corners = extract_board_corners_from_mask(ref_mask)
    print(f"[{obj_name}] ref corners:\n{ref_corners}")

    # 保存参考帧四角可视化
    obj_debug_dir = os.path.join(out_dir, "debug_registered", obj_name)
    os.makedirs(obj_debug_dir, exist_ok=True)

    ref_vis = draw_corners(ref_img, ref_corners, color=(0, 255, 0))
    cv2.imwrite(os.path.join(obj_debug_dir, "ref_corners_vis.png"), ref_vis)
    cv2.imwrite(os.path.join(obj_debug_dir, "ref_mask.png"), ref_mask)
    cv2.imwrite(os.path.join(obj_debug_dir, "ref_img.png"), ref_img)
    cv2.imwrite(os.path.join(obj_debug_dir, "clean_ref.png"), clean_ref)

    # 4) 记录该对象自己的参考相机位姿
    Tcw_ref = poses[ref_idx].astype(np.float64)

    registered = {
        "name": obj_name,
        "ref_idx": ref_idx,
        "ref_frame_name": ref_frame_name,
        "ref_img_h": H_img,
        "ref_img_w": W_img,
        "ref_mask": ref_mask,
        "clean_ref": clean_ref,
        "ref_corners": ref_corners.astype(np.float64),
        "Tcw_ref": Tcw_ref,
        "n_w": n_w,
        "d_w": float(d_w),
        "support_ref_pts": support_ref_pts,
        "support_pids": support_pids,
    }
    return registered


def render_object_on_frame(obj, img, Tcw_cur, K, cur_img_info):
    H_img, W_img = img.shape[:2]
    
    # ====== 第一重约束：深度保护（过滤身后） ======
    # 计算当前帧相机到平面的距离 d_cur (使用 float64 提高精度)
    Tcw_cur = Tcw_cur.astype(np.float64)
    # n_cur = Rcw * n_w, d_cur = n_w^T Cw + d_w
    _, d_cur = plane_world_to_cam(obj["n_w"], obj["d_w"], Tcw_cur)
    
    # 彻底挡住“相机正后方”的物体（漂移根源）
    # 当相机转过身背对 Obj1 时，d_cur 会变小或为负，此时必须切断渲染。
    # 这个阈值（0.005）可以微调，保证 Obj2 的大斜角贴合能出来，但切掉相机的偽影。
    if d_cur < 0.01:
        return img, None, None, False

    # ====== 第二重约束：法线可见性（过滤极侧视角） ======
    # 之前删除法线检查是个错误，它虽然让 Obj2 出来了，但引入了不稳定。
    # 我们需要一个“极其宽松”的角度校验。
    n_cur, _ = plane_world_to_cam(obj["n_w"], obj["d_w"], Tcw_cur)
    
    # 【核心调整】：宽松的角度过滤
    # 在相机坐标系中，Z轴向前。正常可见的平面的 n_cur[2] 应当是负值。
    # 之前是 -0.15，太严了。这里放宽到 0.05（甚至可以放宽到 0.1）。
    # 这能挡掉那些“平行于相机光轴”的极侧投影，但不挡 Obj2 的斜向贴合。
    if n_cur[2] > -0.01:
        return img, None, None, False

    # ====== 第三重约束：角点畸变保护 ======
    n_ref, d_ref = plane_world_to_cam(obj["n_w"], obj["d_w"], obj["Tcw_ref"])
    # 统一符号（Homography标准公式 d > 0）
    if d_ref < 0: n_ref = -n_ref; d_ref = -d_ref
    R, t = relative_ref_to_cur(obj["Tcw_ref"], Tcw_cur)
    
    try:
        # 使用 float64 计算，不做 NORMAL_Z_TH 角度校验，d_cur 和 n_cur[2] 足够了
        Hmat = homography_from_plane(K, R, t, n_ref, d_ref)
        if not np.isfinite(Hmat).all():
            return img, None, None, False
            
        # 投影 Mask (INTER_NEAREST 快速得到 Mask)
        warped_mask = cv2.warpPerspective(obj["ref_mask"], Hmat, (W_img, H_img), flags=cv2.INTER_NEAREST)
        auto_mask = (warped_mask > 127).astype(np.uint8) * 255
        touches_border = mask_touches_image_border(auto_mask)
        min_shared_points = MIN_SHARED_POINTS_EDGE if touches_border else MIN_SHARED_POINTS
        max_reproj_err = MAX_REPROJ_ERR_EDGE if touches_border else MAX_REPROJ_ERR

        cur_pid_to_xy = build_pid_xy_map(cur_img_info)
        shared_ref_pts = []
        shared_cur_pts = []
        for pid, ref_xy in zip(obj["support_pids"], obj["support_ref_pts"]):
            cur_xy = cur_pid_to_xy.get(int(pid))
            if cur_xy is None:
                continue
            shared_ref_pts.append(ref_xy)
            shared_cur_pts.append(cur_xy)

        if touches_border and TRUST_WARPED_MASK_ON_BORDER:
            if len(shared_ref_pts) < MIN_SHARED_POINTS_BORDER_TRUST:
                return img, None, None, False
            shared_ref_pts = None
            shared_cur_pts = None
        elif len(shared_ref_pts) < min_shared_points:
            return img, None, None, False

        if shared_ref_pts is not None:
            shared_ref_pts = np.asarray(shared_ref_pts, dtype=np.float64)
            shared_cur_pts = np.asarray(shared_cur_pts, dtype=np.float64)
            pred_cur_pts = warp_points(Hmat, shared_ref_pts)
            reproj_err = np.linalg.norm(pred_cur_pts - shared_cur_pts, axis=1)
            if float(np.median(reproj_err)) > max_reproj_err:
                return img, None, None, False

        # 畸变过滤：如果 Mask 坐标极其离谱，飞出了宇宙，说明 H 矩阵崩溃了
        # 这一步能彻底挡住爆炸的畸变投影
        # 如果 Mask 占比小于 50 个像素（极小或投影出界），跳过
        if np.sum(auto_mask > 127) < MIN_SUPPORT_AREA:
            return img, auto_mask, None, False

        # ====== 最終貼合渲染 ======
        warped_img = cv2.warpPerspective(obj["clean_ref"], Hmat, (W_img, H_img), flags=cv2.INTER_LINEAR)
        
        # 叠加处理
        res_img = img.copy()
        mask_bool = auto_mask > 127
        res_img[mask_bool] = warped_img[mask_bool]
        
        # 计算线框 debug 角点
        cur_corners = warp_points(Hmat, obj["ref_corners"])
        
        return res_img, auto_mask, cur_corners, True

    except Exception as e:
        return img, None, None, False


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "debug_masks"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "debug_overlay"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "debug_registered"), exist_ok=True)

    images_txt = os.path.join(COLMAP_DIR, "images.txt")
    points_txt = os.path.join(COLMAP_DIR, "points3D.txt")

    imgs = read_images_txt(images_txt)
    pts3d = read_points3D_txt(points_txt)
    poses = load_poses_w2c(POSES_W2C_NPY)
    print("poses:", poses.shape)
    img_info_by_base = {os.path.basename(k): v for k, v in imgs.items()}

    frames = sorted(glob.glob(UNDIST_RGB_GLOB))
    if len(frames) == 0:
        raise RuntimeError("没有找到 undist_rgb 帧")

    if len(poses) != len(frames):
        print(f"警告：poses 数量 {len(poses)} 与 frames 数量 {len(frames)} 不一致")
        print("默认按索引对齐，请确认 poses_world2cam.npy 顺序与 frame_*.png 一致")

    # ===================== 阶段1：构建注册库 =====================
    registered_objects = []
    for obj_cfg in OBJECTS:
        obj = build_registered_object(
            obj_cfg=obj_cfg,
            frames=frames,
            imgs=imgs,
            pts3d=pts3d,
            poses=poses,
            out_dir=OUT_DIR,
        )
        registered_objects.append(obj)
        print(f"registered: {obj['name']} @ {obj['ref_frame_name']}")

    print(f"\n共注册 {len(registered_objects)} 个对象\n")

    # 用第一个对象的图像尺寸作为输出尺寸基准
    base_h = registered_objects[0]["ref_img_h"]
    base_w = registered_objects[0]["ref_img_w"]

    # ===================== 阶段2：对每一帧进行渲染 =====================
    for i, fp in enumerate(frames):
        img = cv2.imread(fp, cv2.IMREAD_COLOR)
        if img is None:
            continue

        if img.shape[:2] != (base_h, base_w):
            img = cv2.resize(img, (base_w, base_h), interpolation=cv2.INTER_LINEAR)

        if i >= len(poses):
            print(f"skip {os.path.basename(fp)}: 没有对应 pose")
            continue

        Tcw_cur = poses[i].astype(np.float64)
        out = img.copy()

        debug_overlay = img.copy()
        any_valid = False
        base = os.path.basename(fp)
        cur_img_info = img_info_by_base.get(base)
        if cur_img_info is None:
            print(f"skip {base}: images.txt 中没有该帧")
            continue

        for obj in registered_objects:
            out, auto_mask, cur_corners, valid = render_object_on_frame(
                obj,
                out,
                Tcw_cur,
                K_undist,
                cur_img_info,
            )

            if valid:
                any_valid = True
                # 叠加调试显示
                debug_overlay = draw_corners(debug_overlay, cur_corners, color=(0, 255, 0))
                tmp = debug_overlay.copy()
                tmp[auto_mask > 127] = (
                    0.6 * tmp[auto_mask > 127] + 0.4 * np.array([0, 255, 0], dtype=np.float32)
                ).astype(np.uint8)
                debug_overlay = tmp

        cv2.imwrite(os.path.join(OUT_DIR, base), out)

        if i % 20 == 0 or i < 5:
            cv2.imwrite(os.path.join(OUT_DIR, "debug_overlay", base), debug_overlay)

        if i % 20 == 0:
            print(f"[{i}/{len(frames)}] wrote {base}, any_valid={any_valid}")

    print("Done:", OUT_DIR)


if __name__ == "__main__":
    main()
