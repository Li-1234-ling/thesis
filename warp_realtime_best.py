import glob
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

import sqlite3


# Required inputs:
# 1) COLMAP_DIR: contains images.txt and points3D.txt from the sparse model.
# 2) MODEL_IMAGE_GLOB: undistorted model images used by COLMAP.
# 3) DATABASE_PATH: COLMAP database, e.g. database.db or skeleton.db.
# 4) VIDEO_SOURCE: camera index or prerecorded video path.
# 5) K_undist: intrinsics for the same undistorted resolution as MODEL_IMAGE_GLOB.
# 6) OBJECTS: each target needs ref_frame_name, ref_mask_path, clean_ref_path.


COLMAP_DIR = r"C:\Users\Administrator\Desktop\uni\MasterArbeit\Li\finaltest\test7\colmap\sparse_txt"
MODEL_IMAGE_GLOB = r"C:\Users\Administrator\Desktop\uni\MasterArbeit\Li\finaltest\test7\images_sub\frame_*.png"
DATABASE_PATH = r"C:\Users\Administrator\Desktop\uni\MasterArbeit\Li\finaltest\test7\colmap\skeleton.db"
VIDEO_SOURCE: Any = r"C:\Users\Administrator\Desktop\uni\MasterArbeit\Li\finaltest\realtime\realtime.mp4"
OUTPUT_VIDEO_PATH: Optional[str] = r"C:\Users\Administrator\Desktop\uni\MasterArbeit\Li\finaltest\realtime\output_realtime_8.mp4"
DEBUG_VIDEO_PATH: Optional[str] = r"C:\Users\Administrator\Desktop\uni\MasterArbeit\Li\finaltest\realtime\debug_realtime_8.mp4"

DEBUG_DIR = os.path.join("_tmp_run_output", "warp_realtime")
MAP_CACHE_PATH = os.path.join(DEBUG_DIR, "colmap_descriptor_map_db_sift.npz")
REBUILD_DESCRIPTOR_MAP = False

SHOW_DEBUG_WINDOW = True
SAVE_DEBUG_FRAME_EVERY = 0
RESIZE_INPUT_TO_MODEL = True
WINDOW_DEBUG = "warp_realtime_debug"
WINDOW_OUTPUT = "warp_realtime_output"
OUTPUT_FPS = 20.0

K_undist = np.array([
    [855.77871693204497, 0.0, 640.0],
    [0.0, 861.25545338322968, 360.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

OBJECTS: List[Dict[str, Any]] = [
    {
        "name": "obj1",
        "ref_frame_name": "frame_000001.png",
        "ref_mask_path": r"C:\Users\Administrator\Desktop\uni\MasterArbeit\Li\finaltest\test7\mask_obj1.png",
        "clean_ref_path": r"C:\Users\Administrator\Desktop\uni\MasterArbeit\Li\finaltest\test7\clean_ref_obj1.png",
        "priority": 10,
    },
    {
        "name": "obj2",
        "ref_frame_name": "frame_000673.png",
        "ref_mask_path": r"C:\Users\Administrator\Desktop\uni\MasterArbeit\Li\finaltest\test7\mask_obj2.png",
        "clean_ref_path": r"C:\Users\Administrator\Desktop\uni\MasterArbeit\Li\finaltest\test7\clean_ref_obj2.png",
        "priority": 9,
    },
    {
        "name": "obj3",
        "ref_frame_name": "frame_001174.png",
        "ref_mask_path": r"C:\Users\Administrator\Desktop\uni\MasterArbeit\Li\finaltest\test7\mask_obj3.png",
        "clean_ref_path": r"C:\Users\Administrator\Desktop\uni\MasterArbeit\Li\finaltest\test7\clean_ref_obj3.png",
        "priority": 8,
    },
    {
        "name": "obj4",
        "ref_frame_name": "frame_001666.png",
        "ref_mask_path": r"C:\Users\Administrator\Desktop\uni\MasterArbeit\Li\finaltest\test7\mask_obj4.png",
        "clean_ref_path": r"C:\Users\Administrator\Desktop\uni\MasterArbeit\Li\finaltest\test7\clean_ref_obj4.png",
        "priority": 8,
    },
    {
        "name": "obj5",
        "ref_frame_name": "frame_001990.png",
        "ref_mask_path": r"C:\Users\Administrator\Desktop\uni\MasterArbeit\Li\finaltest\test7\mask_obj5.png",
        "clean_ref_path": r"C:\Users\Administrator\Desktop\uni\MasterArbeit\Li\finaltest\test7\clean_ref_obj5.png",
        "priority": 8,
    },
]

SIFT_NFEATURES = 3000
COLMAP_POINT_KEYPOINT_SIZE = 16.0
MODEL_IMAGE_STRIDE = 1
MAX_POINTS_PER_IMAGE = 5000
MAX_DESCRIPTORS_PER_POINT = 6
COLMAP_POINT_SAFE_MARGIN = 12
DB_POINT_ALIGN_TH = 1.5
DB_INDEX_ALIGN_MEDIAN_TH = 0.5
DB_INDEX_ALIGN_MAX_TH = 3.0

RATIO_TEST = 0.75
MIN_LOCALIZE_MATCHES = 25
MIN_LOCALIZE_INLIERS = 12
PNP_REPROJ_ERR = 6.0
PNP_ITERATIONS = 200
PNP_CONFIDENCE = 0.999

TRACK_ENABLED = True
TRACK_RELOCALIZE_INTERVAL = 20
TRACK_MAX_POINTS = 300
TRACK_MIN_POINTS = 24
TRACK_MIN_INLIERS = 12
TRACK_MAX_ERROR = 24.0
TRACK_REPROJ_ERR = 8.0
TRACK_WIN_SIZE = (21, 21)
TRACK_MAX_LEVEL = 3
TRACK_MIN_SHARED_POINTS = 3
TRACK_MIN_SHARED_POINTS_EDGE = 2
TRACK_MAX_REPROJ_ERR = 12.0
TRACK_MAX_REPROJ_ERR_EDGE = 18.0
ENABLE_OBJECT_REPROJ_VALIDATION = False

PLANE_INLIER_TH = 0.01
MIN_OBJ_POINTS = 30
MIN_SHARED_POINTS = 8
MIN_SHARED_POINTS_EDGE = 2
MIN_SHARED_POINTS_BORDER_TRUST = 1
MAX_REPROJ_ERR = 12.0
MAX_REPROJ_ERR_EDGE = 25.0
TRUST_WARPED_MASK_ON_BORDER = True
MIN_SUPPORT_AREA = 50
MIN_VISIBLE_SUPPORT_POINTS = 2
MIN_PLANE_DEPTH = 0.01
MIN_FRONT_Z = 0.001
MAX_VISIBLE_NORMAL_Z = -0.01
MAX_SUPPORT_POINTS_TO_TEST = 256


def qvec2rotmat(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
    ], dtype=np.float64)


def pose_from_qvec_tvec(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    Tcw = np.eye(4, dtype=np.float64)
    Tcw[:3, :3] = qvec2rotmat(qvec)
    Tcw[:3, 3] = np.asarray(tvec, dtype=np.float64)
    return Tcw


def read_images_txt(path: str) -> Dict[str, Dict[str, Any]]:
    data: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f.readlines() if line.strip() and not line.startswith("#")]

    i = 0
    while i < len(lines):
        header = lines[i].split()
        image_id = int(header[0])
        qvec = np.array(list(map(float, header[1:5])), dtype=np.float64)
        tvec = np.array(list(map(float, header[5:8])), dtype=np.float64)
        cam_id = int(header[8])
        base = os.path.basename(header[9])

        pts_line = lines[i + 1].split()
        xys = []
        p3d_ids = []
        for j in range(0, len(pts_line), 3):
            xys.append([float(pts_line[j]), float(pts_line[j + 1])])
            p3d_ids.append(int(pts_line[j + 2]))

        data[base] = {
            "image_id": image_id,
            "cam_id": cam_id,
            "name": base,
            "qvec": qvec,
            "tvec": tvec,
            "xys": np.asarray(xys, dtype=np.float64),
            "p3d_ids": np.asarray(p3d_ids, dtype=np.int64),
        }
        i += 2

    return data


def read_points3D_txt(path: str) -> Dict[int, np.ndarray]:
    pts: Dict[int, np.ndarray] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            pts[int(parts[0])] = np.array(list(map(float, parts[1:4])), dtype=np.float64)
    return pts


def fit_plane_ransac(points: np.ndarray, iters: int = 3000, inlier_th: float = PLANE_INLIER_TH) -> Tuple[np.ndarray, float, np.ndarray]:
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
        if best_inliers is None or int(inliers.sum()) > int(best_inliers.sum()):
            best_inliers = inliers

    if best_inliers is None or int(best_inliers.sum()) < 3:
        raise RuntimeError("Plane fitting failed.")

    P = points[best_inliers]
    centroid = P.mean(axis=0)
    Q = P - centroid
    _, _, vh = np.linalg.svd(Q, full_matrices=False)
    n = vh[-1]
    n = n / np.linalg.norm(n)
    d = -np.dot(n, centroid)
    return n.astype(np.float64), float(d), best_inliers


def invert_Tcw(Tcw: np.ndarray) -> np.ndarray:
    R = Tcw[:3, :3]
    t = Tcw[:3, 3]
    Twc = np.eye(4, dtype=np.float64)
    Twc[:3, :3] = R.T
    Twc[:3, 3] = -R.T @ t
    return Twc


def plane_world_to_cam(n_w: np.ndarray, d_w: float, Tcw: np.ndarray) -> Tuple[np.ndarray, float]:
    Rcw = Tcw[:3, :3].astype(np.float64)
    tcw = Tcw[:3, 3].astype(np.float64)
    n_c = Rcw @ n_w
    Cw = -Rcw.T @ tcw
    d_c = float(n_w @ Cw + d_w)
    if d_c < 0:
        n_c = -n_c
        d_c = -d_c
    return n_c, d_c


def relative_ref_to_cur(Tcw_ref: np.ndarray, Tcw_cur: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Twc_ref = invert_Tcw(Tcw_ref)
    T_cur_ref = Tcw_cur @ Twc_ref
    return T_cur_ref[:3, :3], T_cur_ref[:3, 3]


def homography_from_plane(K: np.ndarray, R: np.ndarray, t: np.ndarray, n_ref: np.ndarray, d_ref: float) -> np.ndarray:
    Kinv = np.linalg.inv(K)
    denom = float(d_ref)
    if abs(denom) < 1e-8:
        raise RuntimeError(f"d_ref is too small for a stable homography: {denom}")
    return K @ (R - (t.reshape(3, 1) @ n_ref.reshape(1, 3)) / denom) @ Kinv


def order_corners(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def extract_corners_from_mask(mask: np.ndarray) -> np.ndarray:
    mask_bin = (mask > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Reference mask has no valid contour.")

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

    return order_corners(approx)


def warp_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H)
    return out.reshape(-1, 2)


def build_pid_xy_map(pids: np.ndarray, xys: np.ndarray) -> Dict[int, np.ndarray]:
    pid_to_xy: Dict[int, np.ndarray] = {}
    if pids is None or xys is None:
        return pid_to_xy

    pids = np.asarray(pids, dtype=np.int64).reshape(-1)
    xys = np.asarray(xys, dtype=np.float64).reshape(-1, 2)
    count = min(pids.shape[0], xys.shape[0])
    for idx in range(count):
        pid = int(pids[idx])
        if pid >= 0:
            pid_to_xy[pid] = xys[idx]
    return pid_to_xy


def mask_touches_image_border(mask: np.ndarray) -> bool:
    return bool(
        np.any(mask[0, :] > 127) or
        np.any(mask[-1, :] > 127) or
        np.any(mask[:, 0] > 127) or
        np.any(mask[:, -1] > 127)
    )


def draw_corners(img: np.ndarray, corners: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), radius: int = 6) -> np.ndarray:
    vis = img.copy()
    corners_i = np.round(corners).astype(np.int32)
    for p in corners_i:
        cv2.circle(vis, tuple(p), radius, color, -1, lineType=cv2.LINE_AA)
    for i in range(4):
        cv2.line(vis, tuple(corners_i[i]), tuple(corners_i[(i + 1) % 4]), color, 2, lineType=cv2.LINE_AA)
    return vis


def ensure_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize_like(img: np.ndarray, ref_shape: Tuple[int, int], is_mask: bool) -> np.ndarray:
    h, w = ref_shape
    if img.shape[:2] == (h, w):
        return img
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(img, (w, h), interpolation=interp)


def normalize_descriptor(desc: np.ndarray) -> np.ndarray:
    desc = np.asarray(desc, dtype=np.float32)
    norm = float(np.linalg.norm(desc))
    if norm < 1e-12:
        return desc
    return desc / norm


def normalize_descriptors_matrix(desc: np.ndarray) -> np.ndarray:
    desc = np.asarray(desc, dtype=np.float32)
    if desc.ndim != 2 or desc.shape[0] == 0:
        return desc
    norms = np.linalg.norm(desc, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return desc / norms


def limit_samples_evenly(items: List[Tuple[int, float, float]], limit: int) -> List[Tuple[int, float, float]]:
    if limit <= 0 or len(items) <= limit:
        return items
    idx = np.linspace(0, len(items) - 1, num=limit, dtype=np.int32)
    return [items[int(i)] for i in idx]


def parse_video_source(source: Any) -> Any:
    if isinstance(source, int):
        return source
    if isinstance(source, str) and source.isdigit():
        return int(source)
    return source


def make_video_writer(path: Optional[str], frame_size: Tuple[int, int]) -> Optional[cv2.VideoWriter]:
    if not path:
        return None
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = frame_size
    return cv2.VideoWriter(path, fourcc, OUTPUT_FPS, (w, h))


def create_sift() -> Any:
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("OpenCV SIFT is not available in this environment.")
    return cv2.SIFT_create(nfeatures=SIFT_NFEATURES)


def create_flann_matcher() -> Any:
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=64)
    return cv2.FlannBasedMatcher(index_params, search_params)


def build_base_to_path(frames: List[str]) -> Dict[str, str]:
    return {os.path.basename(path): path for path in frames}


def open_colmap_db_readonly(path: str) -> sqlite3.Connection:
    db_path = Path(path)
    if not db_path.is_file():
        raise FileNotFoundError(f"COLMAP database not found: {path}")
    try:
        return sqlite3.connect(str(db_path))
    except sqlite3.Error:
        return sqlite3.connect(db_path.resolve().as_uri(), uri=True)


def load_db_image_name_to_id(con: sqlite3.Connection) -> Dict[str, int]:
    cur = con.cursor()
    rows = cur.execute("SELECT image_id, name FROM images").fetchall()
    return {os.path.basename(name): int(image_id) for image_id, name in rows}


def load_db_keypoints_and_descriptors(con: sqlite3.Connection, image_id: int) -> Tuple[np.ndarray, np.ndarray]:
    cur = con.cursor()
    kp_row = cur.execute(
        "SELECT rows, cols, data FROM keypoints WHERE image_id=?",
        (int(image_id),),
    ).fetchone()
    desc_row = cur.execute(
        "SELECT rows, cols, data FROM descriptors WHERE image_id=?",
        (int(image_id),),
    ).fetchone()
    if kp_row is None or desc_row is None:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 128), dtype=np.float32)

    kp_rows, kp_cols, kp_blob = kp_row
    desc_rows, desc_cols, desc_blob = desc_row
    keypoints = np.frombuffer(kp_blob, dtype=np.float32).reshape(int(kp_rows), int(kp_cols))
    descriptors = np.frombuffer(desc_blob, dtype=np.uint8).reshape(int(desc_rows), int(desc_cols)).astype(np.float32)
    descriptors = normalize_descriptors_matrix(descriptors)
    return keypoints, descriptors


def align_sparse_points_to_db_keypoints(xys: np.ndarray, p3d_ids: np.ndarray, db_keypoints: np.ndarray) -> List[Tuple[int, int]]:
    if xys.shape[0] == 0 or db_keypoints.shape[0] == 0:
        return []

    sparse_xy = np.asarray(xys[:, :2], dtype=np.float32)
    db_xy = np.asarray(db_keypoints[:, :2], dtype=np.float32)
    valid_sparse_idx = np.flatnonzero(p3d_ids >= 0)
    if valid_sparse_idx.size == 0:
        return []

    if sparse_xy.shape[0] == db_xy.shape[0]:
        dif = np.linalg.norm(sparse_xy - db_xy, axis=1)
        if float(np.median(dif)) <= DB_INDEX_ALIGN_MEDIAN_TH and float(np.max(dif)) <= DB_INDEX_ALIGN_MAX_TH:
            return [(int(idx), int(idx)) for idx in valid_sparse_idx.tolist()]

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = matcher.match(sparse_xy[valid_sparse_idx], db_xy)
    matches = sorted(matches, key=lambda m: m.distance)

    pairs: List[Tuple[int, int]] = []
    used_db = set()
    for m in matches:
        if float(m.distance) > DB_POINT_ALIGN_TH:
            continue
        sparse_idx = int(valid_sparse_idx[m.queryIdx])
        db_idx = int(m.trainIdx)
        if db_idx in used_db:
            continue
        used_db.add(db_idx)
        pairs.append((sparse_idx, db_idx))
    return pairs


def build_registered_object(
    obj_cfg: Dict[str, Any],
    base2path: Dict[str, str],
    imgs: Dict[str, Dict[str, Any]],
    pts3d: Dict[int, np.ndarray],
    out_dir: str,
) -> Dict[str, Any]:
    ref_frame_name = obj_cfg["ref_frame_name"]
    ref_mask_path = obj_cfg["ref_mask_path"]
    clean_ref_path = obj_cfg["clean_ref_path"]
    obj_name = obj_cfg["name"]

    if ref_frame_name not in base2path:
        raise RuntimeError(f"[{obj_name}] Missing reference image on disk: {ref_frame_name}")
    if ref_frame_name not in imgs:
        raise RuntimeError(f"[{obj_name}] Missing reference image in images.txt: {ref_frame_name}")

    ref_info = imgs[ref_frame_name]
    ref_img = cv2.imread(base2path[ref_frame_name], cv2.IMREAD_COLOR)
    ref_mask = cv2.imread(ref_mask_path, cv2.IMREAD_GRAYSCALE)
    clean_ref = cv2.imread(clean_ref_path, cv2.IMREAD_COLOR)

    if ref_img is None:
        raise RuntimeError(f"[{obj_name}] Failed to read ref image: {base2path[ref_frame_name]}")
    if ref_mask is None:
        raise RuntimeError(f"[{obj_name}] Failed to read ref mask: {ref_mask_path}")
    if clean_ref is None:
        raise RuntimeError(f"[{obj_name}] Failed to read clean_ref: {clean_ref_path}")

    H_img, W_img = ref_img.shape[:2]
    ref_mask = resize_like(ref_mask, (H_img, W_img), is_mask=True)
    clean_ref = resize_like(clean_ref, (H_img, W_img), is_mask=False)

    obj_pts = []
    obj_ref_xys = []
    obj_pids = []
    for (x, y), pid in zip(ref_info["xys"], ref_info["p3d_ids"]):
        pid = int(pid)
        if pid < 0 or pid not in pts3d:
            continue
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W_img and 0 <= yi < H_img and ref_mask[yi, xi] > 127:
            obj_pts.append(pts3d[pid])
            obj_ref_xys.append([x, y])
            obj_pids.append(pid)

    obj_pts = np.asarray(obj_pts, dtype=np.float64)
    obj_ref_xys = np.asarray(obj_ref_xys, dtype=np.float64)
    obj_pids = np.asarray(obj_pids, dtype=np.int64)
    print(f"[{obj_name}] 3D points in mask: {obj_pts.shape[0]}")

    if obj_pts.shape[0] < MIN_OBJ_POINTS:
        raise RuntimeError(f"[{obj_name}] Too few 3D points inside mask: {obj_pts.shape[0]}")

    n_w, d_w, inliers = fit_plane_ransac(obj_pts, iters=3000, inlier_th=PLANE_INLIER_TH)
    print(f"[{obj_name}] plane(world): inliers={int(inliers.sum())}/{obj_pts.shape[0]}")

    support_xyz = obj_pts[inliers]
    support_ref_pts = obj_ref_xys[inliers]
    support_pids = obj_pids[inliers]
    ref_corners = extract_corners_from_mask(ref_mask).astype(np.float64)
    Tcw_ref = pose_from_qvec_tvec(ref_info["qvec"], ref_info["tvec"])

    obj_debug_dir = os.path.join(out_dir, "registered", obj_name)
    os.makedirs(obj_debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(obj_debug_dir, "ref_img.png"), ref_img)
    cv2.imwrite(os.path.join(obj_debug_dir, "ref_mask.png"), ref_mask)
    cv2.imwrite(os.path.join(obj_debug_dir, "clean_ref.png"), clean_ref)
    cv2.imwrite(os.path.join(obj_debug_dir, "ref_corners_vis.png"), draw_corners(ref_img, ref_corners))

    return {
        "name": obj_name,
        "priority": int(obj_cfg.get("priority", 0)),
        "ref_frame_name": ref_frame_name,
        "ref_img_h": H_img,
        "ref_img_w": W_img,
        "ref_mask": ref_mask,
        "clean_ref": clean_ref,
        "ref_corners": ref_corners,
        "Tcw_ref": Tcw_ref,
        "n_w": n_w,
        "d_w": float(d_w),
        "support_xyz": support_xyz.astype(np.float64),
        "support_ref_pts": support_ref_pts.astype(np.float64),
        "support_pids": support_pids.astype(np.int64),
    }


def maybe_load_descriptor_map(cache_path: str) -> Optional[Dict[str, np.ndarray]]:
    if REBUILD_DESCRIPTOR_MAP or not os.path.isfile(cache_path):
        return None
    data = np.load(cache_path)
    descriptors = np.asarray(data["descriptors"], dtype=np.float32)
    xyz = np.asarray(data["xyz"], dtype=np.float64)
    pids = np.asarray(data["pids"], dtype=np.int64)
    if descriptors.ndim != 2 or descriptors.shape[0] == 0:
        return None
    print(f"[INFO] Loaded descriptor map cache: {cache_path}")
    return {"descriptors": descriptors, "xyz": xyz, "pids": pids}


def save_descriptor_map(cache_path: str, descriptors: np.ndarray, xyz: np.ndarray, pids: np.ndarray) -> None:
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez(cache_path, descriptors=descriptors, xyz=xyz, pids=pids)
    print(f"[INFO] Saved descriptor map cache: {cache_path}")


def build_descriptor_map(
    model_frames: List[str],
    imgs: Dict[str, Dict[str, Any]],
    pts3d: Dict[int, np.ndarray],
) -> Dict[str, np.ndarray]:
    cached = maybe_load_descriptor_map(MAP_CACHE_PATH)
    if cached is not None:
        return cached

    pid_to_descs: Dict[int, List[np.ndarray]] = defaultdict(list)
    db_con = open_colmap_db_readonly(DATABASE_PATH)
    db_name_to_id = load_db_image_name_to_id(db_con)

    used_images = 0
    total_input_points = 0
    total_aligned_points = 0
    total_kept_desc = 0
    sampled_frames = model_frames[::max(1, MODEL_IMAGE_STRIDE)]

    try:
        for idx, path in enumerate(sampled_frames):
            base = os.path.basename(path)
            info = imgs.get(base)
            image_id = db_name_to_id.get(base)
            if info is None or image_id is None:
                continue

            db_keypoints, db_descriptors = load_db_keypoints_and_descriptors(db_con, image_id)
            if db_keypoints.shape[0] == 0 or db_descriptors.shape[0] == 0:
                continue

            xys = info["xys"]
            p3d_ids = info["p3d_ids"]
            aligned_pairs = align_sparse_points_to_db_keypoints(xys, p3d_ids, db_keypoints)
            if not aligned_pairs:
                continue

            kept_this_image = 0
            for sparse_idx, db_idx in aligned_pairs[:MAX_POINTS_PER_IMAGE]:
                pid = int(p3d_ids[sparse_idx])
                if pid < 0 or pid not in pts3d:
                    continue
                x, y = xys[sparse_idx]
                if x < COLMAP_POINT_SAFE_MARGIN or y < COLMAP_POINT_SAFE_MARGIN:
                    continue
                desc = db_descriptors[db_idx]
                if float(np.linalg.norm(desc)) < 1e-12:
                    continue
                if len(pid_to_descs[pid]) < MAX_DESCRIPTORS_PER_POINT:
                    pid_to_descs[pid].append(desc)
                    kept_this_image += 1
                    total_kept_desc += 1

            used_images += 1
            total_input_points += int(np.sum(p3d_ids >= 0))
            total_aligned_points += len(aligned_pairs)
            if (idx + 1) % 50 == 0 or idx == len(sampled_frames) - 1:
                print(
                    f"[INFO] descriptor map build: {idx + 1}/{len(sampled_frames)} images, "
                    f"used_images={used_images}, aligned={total_aligned_points}, kept_desc={total_kept_desc}"
                )
    finally:
        db_con.close()

    pids = []
    xyz = []
    descriptors = []
    for pid, desc_list in pid_to_descs.items():
        if not desc_list:
            continue
        desc_stack = np.stack(desc_list, axis=0).astype(np.float32)
        desc_mean = normalize_descriptor(desc_stack.mean(axis=0))
        if float(np.linalg.norm(desc_mean)) < 1e-12:
            continue
        pids.append(pid)
        xyz.append(pts3d[pid])
        descriptors.append(desc_mean)

    if not descriptors:
        raise RuntimeError("Failed to build any 3D descriptors from the COLMAP model.")

    descriptors_arr = np.asarray(descriptors, dtype=np.float32)
    xyz_arr = np.asarray(xyz, dtype=np.float64)
    pids_arr = np.asarray(pids, dtype=np.int64)

    save_descriptor_map(MAP_CACHE_PATH, descriptors_arr, xyz_arr, pids_arr)
    print(
        f"[INFO] descriptor map ready: 3d_points={len(pids_arr)}, "
        f"used_images={used_images}, input_points={total_input_points}, aligned_points={total_aligned_points}"
    )
    return {"descriptors": descriptors_arr, "xyz": xyz_arr, "pids": pids_arr}


def build_pnp_correspondences(
    cur_keypoints: List[Any],
    cur_descriptors: np.ndarray,
    descriptor_map: Dict[str, np.ndarray],
    matcher: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if cur_descriptors is None or len(cur_descriptors) == 0:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
            0,
        )

    raw_matches = matcher.knnMatch(
        np.asarray(cur_descriptors, dtype=np.float32),
        np.asarray(descriptor_map["descriptors"], dtype=np.float32),
        k=2,
    )

    best_by_map: Dict[int, Any] = {}
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance >= RATIO_TEST * n.distance:
            continue
        prev = best_by_map.get(m.trainIdx)
        if prev is None or m.distance < prev.distance:
            best_by_map[m.trainIdx] = m

    if not best_by_map:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
            0,
        )

    image_pts = []
    object_pts = []
    matched_pids = []
    for map_idx, match in sorted(best_by_map.items(), key=lambda kv: kv[1].distance):
        image_pts.append(cur_keypoints[match.queryIdx].pt)
        object_pts.append(descriptor_map["xyz"][map_idx])
        matched_pids.append(int(descriptor_map["pids"][map_idx]))

    return (
        np.asarray(image_pts, dtype=np.float64),
        np.asarray(object_pts, dtype=np.float64),
        np.asarray(matched_pids, dtype=np.int64),
        len(best_by_map),
    )


def Tcw_to_rvec_tvec(Tcw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rvec, _ = cv2.Rodrigues(Tcw[:3, :3].astype(np.float64))
    tvec = Tcw[:3, 3].reshape(3, 1).astype(np.float64)
    return rvec, tvec


def subsample_correspondences(
    image_pts: np.ndarray,
    object_pts: np.ndarray,
    pids: np.ndarray,
    limit: int,
    priority_pids: Optional[Set[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if limit <= 0 or image_pts.shape[0] <= limit:
        return image_pts, object_pts, pids

    if priority_pids:
        priority_mask = np.array([int(pid) in priority_pids for pid in pids], dtype=bool)
        keep_idx = np.flatnonzero(priority_mask)
        if keep_idx.shape[0] >= limit:
            sel = np.linspace(0, keep_idx.shape[0] - 1, num=limit, dtype=np.int32)
            idx = keep_idx[sel]
            return image_pts[idx], object_pts[idx], pids[idx]

        remain_budget = limit - keep_idx.shape[0]
        remain_idx = np.flatnonzero(~priority_mask)
        if remain_idx.shape[0] > remain_budget and remain_budget > 0:
            sel = np.linspace(0, remain_idx.shape[0] - 1, num=remain_budget, dtype=np.int32)
            remain_idx = remain_idx[sel]
        idx = np.concatenate([keep_idx, remain_idx], axis=0)
        idx = np.sort(idx.astype(np.int32))
        return image_pts[idx], object_pts[idx], pids[idx]

    idx = np.linspace(0, image_pts.shape[0] - 1, num=limit, dtype=np.int32)
    return image_pts[idx], object_pts[idx], pids[idx]


def build_tracking_state(
    image_pts: np.ndarray,
    object_pts: np.ndarray,
    pids: np.ndarray,
    Tcw: np.ndarray,
    priority_pids: Optional[Set[int]] = None,
) -> Optional[Dict[str, Any]]:
    if image_pts is None or object_pts is None or pids is None:
        return None
    if image_pts.shape[0] < TRACK_MIN_POINTS or object_pts.shape[0] < TRACK_MIN_POINTS or pids.shape[0] < TRACK_MIN_POINTS:
        return None
    image_pts, object_pts, pids = subsample_correspondences(
        np.asarray(image_pts, dtype=np.float64),
        np.asarray(object_pts, dtype=np.float64),
        np.asarray(pids, dtype=np.int64),
        TRACK_MAX_POINTS,
        priority_pids=priority_pids,
    )
    return {
        "image_points": image_pts.astype(np.float32),
        "object_points": object_pts.astype(np.float64),
        "pids": pids.astype(np.int64),
        "Tcw": np.asarray(Tcw, dtype=np.float64),
    }


def solve_pose_from_tracked_correspondences(
    object_pts: np.ndarray,
    image_pts: np.ndarray,
    pids: np.ndarray,
    K: np.ndarray,
    initial_Tcw: np.ndarray,
) -> Optional[Dict[str, Any]]:
    if object_pts.shape[0] < TRACK_MIN_POINTS or image_pts.shape[0] < TRACK_MIN_POINTS or pids.shape[0] < TRACK_MIN_POINTS:
        return None

    rvec_init, tvec_init = Tcw_to_rvec_tvec(initial_Tcw)
    ok, rvec_est, tvec_est = cv2.solvePnP(
        objectPoints=object_pts,
        imagePoints=image_pts,
        cameraMatrix=K,
        distCoeffs=None,
        rvec=rvec_init,
        tvec=tvec_init,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    proj, _ = cv2.projectPoints(object_pts, rvec_est, tvec_est, K, None)
    proj = proj.reshape(-1, 2)
    reproj_err = np.linalg.norm(proj - image_pts.reshape(-1, 2), axis=1)
    inlier_mask = reproj_err < TRACK_REPROJ_ERR
    if int(np.sum(inlier_mask)) < TRACK_MIN_INLIERS:
        return None

    object_pts_in = object_pts[inlier_mask]
    image_pts_in = image_pts[inlier_mask]
    pids_in = pids[inlier_mask]
    ok, rvec_est, tvec_est = cv2.solvePnP(
        objectPoints=object_pts_in,
        imagePoints=image_pts_in,
        cameraMatrix=K,
        distCoeffs=None,
        rvec=rvec_est,
        tvec=tvec_est,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    Rcw, _ = cv2.Rodrigues(rvec_est)
    Tcw = np.eye(4, dtype=np.float64)
    Tcw[:3, :3] = Rcw
    Tcw[:3, 3] = tvec_est.reshape(3)
    return {
        "ok": True,
        "keypoints": int(image_pts.shape[0]),
        "matches": int(image_pts.shape[0]),
        "inliers": int(np.sum(inlier_mask)),
        "Tcw": Tcw,
        "image_points_eval": np.asarray(image_pts, dtype=np.float64),
        "pids_eval": np.asarray(pids, dtype=np.int64),
        "image_points_in": np.asarray(image_pts_in, dtype=np.float64),
        "object_points_in": np.asarray(object_pts_in, dtype=np.float64),
        "pids_in": np.asarray(pids_in, dtype=np.int64),
        "mode": "track",
    }


def localize_current_frame(
    frame_gray: np.ndarray,
    K: np.ndarray,
    detector: Any,
    matcher: Any,
    descriptor_map: Dict[str, np.ndarray],
) -> Optional[Dict[str, Any]]:
    cur_keypoints, cur_descriptors = detector.detectAndCompute(frame_gray, None)
    cur_keypoints = cur_keypoints or []
    if cur_descriptors is None or len(cur_keypoints) == 0:
        return None
    cur_descriptors = normalize_descriptors_matrix(cur_descriptors)

    image_pts, object_pts, matched_pids, match_count = build_pnp_correspondences(
        cur_keypoints,
        cur_descriptors,
        descriptor_map,
        matcher,
    )
    if match_count < MIN_LOCALIZE_MATCHES:
        return {
            "ok": False,
            "keypoints": len(cur_keypoints),
            "matches": match_count,
            "inliers": 0,
            "Tcw": None,
            "image_points_eval": image_pts,
            "pids_eval": matched_pids,
            "image_points_in": np.empty((0, 2), dtype=np.float64),
            "object_points_in": np.empty((0, 3), dtype=np.float64),
            "pids_in": np.empty((0,), dtype=np.int64),
            "mode": "global",
        }

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=object_pts,
        imagePoints=image_pts,
        cameraMatrix=K,
        distCoeffs=None,
        flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=PNP_REPROJ_ERR,
        iterationsCount=PNP_ITERATIONS,
        confidence=PNP_CONFIDENCE,
    )
    if not ok or inliers is None or len(inliers) < MIN_LOCALIZE_INLIERS:
        return {
            "ok": False,
            "keypoints": len(cur_keypoints),
            "matches": match_count,
            "inliers": 0 if inliers is None else int(len(inliers)),
            "Tcw": None,
            "image_points_eval": image_pts,
            "pids_eval": matched_pids,
            "image_points_in": np.empty((0, 2), dtype=np.float64),
            "object_points_in": np.empty((0, 3), dtype=np.float64),
            "pids_in": np.empty((0,), dtype=np.int64),
            "mode": "global",
        }

    inlier_idx = inliers.ravel().astype(np.int32)
    object_pts_in = object_pts[inlier_idx]
    image_pts_in = image_pts[inlier_idx]
    pids_in = matched_pids[inlier_idx]

    ok_refine, rvec, tvec = cv2.solvePnP(
        objectPoints=object_pts_in,
        imagePoints=image_pts_in,
        cameraMatrix=K,
        distCoeffs=None,
        rvec=rvec,
        tvec=tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok_refine:
        return {
            "ok": False,
            "keypoints": len(cur_keypoints),
            "matches": match_count,
            "inliers": int(len(inlier_idx)),
            "Tcw": None,
            "image_points_eval": image_pts,
            "pids_eval": matched_pids,
            "image_points_in": np.empty((0, 2), dtype=np.float64),
            "object_points_in": np.empty((0, 3), dtype=np.float64),
            "pids_in": np.empty((0,), dtype=np.int64),
            "mode": "global",
        }

    Rcw, _ = cv2.Rodrigues(rvec)
    Tcw = np.eye(4, dtype=np.float64)
    Tcw[:3, :3] = Rcw
    Tcw[:3, 3] = tvec.reshape(3)

    return {
        "ok": True,
        "keypoints": len(cur_keypoints),
        "matches": match_count,
        "inliers": int(len(inlier_idx)),
        "Tcw": Tcw,
        "image_points_eval": np.asarray(image_pts, dtype=np.float64),
        "pids_eval": np.asarray(matched_pids, dtype=np.int64),
        "image_points_in": np.asarray(image_pts_in, dtype=np.float64),
        "object_points_in": np.asarray(object_pts_in, dtype=np.float64),
        "pids_in": np.asarray(pids_in, dtype=np.int64),
        "mode": "global",
    }


def track_current_frame(
    prev_gray: np.ndarray,
    frame_gray: np.ndarray,
    track_state: Dict[str, Any],
    K: np.ndarray,
) -> Optional[Dict[str, Any]]:
    prev_pts = np.asarray(track_state["image_points"], dtype=np.float32).reshape(-1, 1, 2)
    object_pts = np.asarray(track_state["object_points"], dtype=np.float64)
    pids = np.asarray(track_state["pids"], dtype=np.int64)
    if prev_pts.shape[0] < TRACK_MIN_POINTS or object_pts.shape[0] < TRACK_MIN_POINTS or pids.shape[0] < TRACK_MIN_POINTS:
        return None

    next_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        frame_gray,
        prev_pts,
        None,
        winSize=TRACK_WIN_SIZE,
        maxLevel=TRACK_MAX_LEVEL,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )
    if next_pts is None or status is None:
        return None

    status = status.reshape(-1).astype(bool)
    err = np.zeros(prev_pts.shape[0], dtype=np.float32) if err is None else err.reshape(-1)
    good = status & np.isfinite(next_pts.reshape(-1, 2)).all(axis=1) & (err <= TRACK_MAX_ERROR)
    if int(np.sum(good)) < TRACK_MIN_POINTS:
        return None

    image_pts = next_pts.reshape(-1, 2)[good]
    object_pts = object_pts[good]
    pids = pids[good]
    result = solve_pose_from_tracked_correspondences(object_pts, image_pts, pids, K, track_state["Tcw"])
    return result


def count_visible_support_points(points_w: np.ndarray, Tcw: np.ndarray, K: np.ndarray, h: int, w: int) -> int:
    if points_w.size == 0:
        return 0

    pts = points_w
    if pts.shape[0] > MAX_SUPPORT_POINTS_TO_TEST:
        idx = np.linspace(0, pts.shape[0] - 1, num=MAX_SUPPORT_POINTS_TO_TEST, dtype=np.int32)
        pts = pts[idx]

    Rcw = Tcw[:3, :3]
    tcw = Tcw[:3, 3]
    pts_c = (Rcw @ pts.T).T + tcw.reshape(1, 3)

    z = pts_c[:, 2]
    valid_z = z > MIN_FRONT_Z
    if not np.any(valid_z):
        return 0

    pts_c = pts_c[valid_z]
    z = pts_c[:, 2]
    u = K[0, 0] * (pts_c[:, 0] / z) + K[0, 2]
    v = K[1, 1] * (pts_c[:, 1] / z) + K[1, 2]
    visible = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return int(np.sum(visible))


def render_object_on_frame(
    obj: Dict[str, Any],
    img: np.ndarray,
    Tcw_cur: np.ndarray,
    K: np.ndarray,
    cur_pid_to_xy: Dict[int, np.ndarray],
    pose_mode: str,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], bool]:
    H_img, W_img = img.shape[:2]
    Tcw_cur = Tcw_cur.astype(np.float64)

    _, d_cur = plane_world_to_cam(obj["n_w"], obj["d_w"], Tcw_cur)
    if d_cur < MIN_PLANE_DEPTH:
        return img, None, None, False

    n_cur, _ = plane_world_to_cam(obj["n_w"], obj["d_w"], Tcw_cur)
    if n_cur[2] > MAX_VISIBLE_NORMAL_Z:
        return img, None, None, False

    visible_support = count_visible_support_points(obj["support_xyz"], Tcw_cur, K, H_img, W_img)
    if visible_support < MIN_VISIBLE_SUPPORT_POINTS:
        return img, None, None, False

    n_ref, d_ref = plane_world_to_cam(obj["n_w"], obj["d_w"], obj["Tcw_ref"])
    if d_ref < 0:
        n_ref = -n_ref
        d_ref = -d_ref
    R, t = relative_ref_to_cur(obj["Tcw_ref"], Tcw_cur)

    try:
        Hmat = homography_from_plane(K, R, t, n_ref, d_ref)
        if not np.isfinite(Hmat).all():
            return img, None, None, False

        warped_mask = cv2.warpPerspective(obj["ref_mask"], Hmat, (W_img, H_img), flags=cv2.INTER_NEAREST)
        auto_mask = (warped_mask > 127).astype(np.uint8) * 255
        if ENABLE_OBJECT_REPROJ_VALIDATION:
            touches_border = mask_touches_image_border(auto_mask)
            if pose_mode == "track":
                min_shared_points = TRACK_MIN_SHARED_POINTS_EDGE if touches_border else TRACK_MIN_SHARED_POINTS
                max_reproj_err = TRACK_MAX_REPROJ_ERR_EDGE if touches_border else TRACK_MAX_REPROJ_ERR
            else:
                min_shared_points = MIN_SHARED_POINTS_EDGE if touches_border else MIN_SHARED_POINTS
                max_reproj_err = MAX_REPROJ_ERR_EDGE if touches_border else MAX_REPROJ_ERR

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
                shared_ref_pts_arr = np.asarray(shared_ref_pts, dtype=np.float64)
                shared_cur_pts_arr = np.asarray(shared_cur_pts, dtype=np.float64)
                pred_cur_pts = warp_points(Hmat, shared_ref_pts_arr)
                reproj_err = np.linalg.norm(pred_cur_pts - shared_cur_pts_arr, axis=1)
                if float(np.median(reproj_err)) > max_reproj_err:
                    return img, None, None, False

        if int(np.sum(auto_mask > 127)) < MIN_SUPPORT_AREA:
            return img, auto_mask, None, False

        warped_img = cv2.warpPerspective(obj["clean_ref"], Hmat, (W_img, H_img), flags=cv2.INTER_LINEAR)
        res_img = img.copy()
        mask_bool = auto_mask > 127
        res_img[mask_bool] = warped_img[mask_bool]
        cur_corners = warp_points(Hmat, obj["ref_corners"])
        if not np.isfinite(cur_corners).all():
            return img, None, None, False
        return res_img, auto_mask, cur_corners, True

    except Exception:
        return img, None, None, False


def annotate_debug(frame: np.ndarray, pose_result: Optional[Dict[str, Any]], active_objects: List[str], fps: float) -> np.ndarray:
    vis = frame.copy()
    lines = [f"fps: {fps:.1f}"]

    if pose_result is None:
        lines.append("pose: no features")
    elif not pose_result["ok"]:
        lines.append(
            f"pose: {pose_result.get('mode', 'unknown')} lost | kps={pose_result['keypoints']} "
            f"matches={pose_result['matches']} inliers={pose_result['inliers']}"
        )
    else:
        lines.append(
            f"pose: {pose_result.get('mode', 'unknown')} ok | kps={pose_result['keypoints']} "
            f"matches={pose_result['matches']} inliers={pose_result['inliers']}"
        )

    lines.append("objects: " + (", ".join(active_objects) if active_objects else "none"))

    y = 24
    for line in lines:
        cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24
    return vis


def main() -> None:
    os.makedirs(DEBUG_DIR, exist_ok=True)
    os.makedirs(os.path.join(DEBUG_DIR, "frames"), exist_ok=True)

    images_txt = os.path.join(COLMAP_DIR, "images.txt")
    points_txt = os.path.join(COLMAP_DIR, "points3D.txt")

    imgs = read_images_txt(images_txt)
    pts3d = read_points3D_txt(points_txt)
    model_frames = sorted(glob.glob(MODEL_IMAGE_GLOB))
    if not model_frames:
        raise RuntimeError(f"No model images found with MODEL_IMAGE_GLOB: {MODEL_IMAGE_GLOB}")

    base2path = build_base_to_path(model_frames)
    descriptor_map = build_descriptor_map(model_frames, imgs, pts3d)
    print(f"[INFO] descriptor map points: {descriptor_map['descriptors'].shape[0]}")

    registered_objects = []
    for obj_cfg in OBJECTS:
        obj = build_registered_object(
            obj_cfg=obj_cfg,
            base2path=base2path,
            imgs=imgs,
            pts3d=pts3d,
            out_dir=DEBUG_DIR,
        )
        registered_objects.append(obj)
        print(f"[INFO] registered object: {obj['name']} @ {obj['ref_frame_name']}")

    if not registered_objects:
        raise RuntimeError("No objects were registered.")

    tracking_priority_pids: Set[int] = set()
    for obj in registered_objects:
        tracking_priority_pids.update(int(pid) for pid in obj["support_pids"].tolist())

    base_h = registered_objects[0]["ref_img_h"]
    base_w = registered_objects[0]["ref_img_w"]

    detector = create_sift()
    matcher = create_flann_matcher()

    cap = cv2.VideoCapture(parse_video_source(VIDEO_SOURCE))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {VIDEO_SOURCE}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    writer: Optional[cv2.VideoWriter] = None
    debug_writer: Optional[cv2.VideoWriter] = None
    frame_idx = 0
    last_time = time.perf_counter()
    prev_gray: Optional[np.ndarray] = None
    track_state: Optional[Dict[str, Any]] = None
    frames_since_global = TRACK_RELOCALIZE_INTERVAL

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if RESIZE_INPUT_TO_MODEL and frame.shape[:2] != (base_h, base_w):
                frame = cv2.resize(frame, (base_w, base_h), interpolation=cv2.INTER_LINEAR)

            frame_gray = ensure_gray(frame)
            pose_result: Optional[Dict[str, Any]] = None

            if (
                TRACK_ENABLED and
                track_state is not None and
                prev_gray is not None and
                frames_since_global < TRACK_RELOCALIZE_INTERVAL
            ):
                pose_result = track_current_frame(
                    prev_gray=prev_gray,
                    frame_gray=frame_gray,
                    track_state=track_state,
                    K=K_undist,
                )
                if pose_result is not None and pose_result["ok"]:
                    track_state = build_tracking_state(
                        pose_result["image_points_in"],
                        pose_result["object_points_in"],
                        pose_result["pids_in"],
                        pose_result["Tcw"],
                        priority_pids=tracking_priority_pids,
                    )
                    frames_since_global += 1
                else:
                    track_state = None
                    frames_since_global = TRACK_RELOCALIZE_INTERVAL

            if pose_result is None or not pose_result["ok"]:
                pose_result = localize_current_frame(
                    frame_gray=frame_gray,
                    K=K_undist,
                    detector=detector,
                    matcher=matcher,
                    descriptor_map=descriptor_map,
                )
                if pose_result is not None and pose_result["ok"]:
                    track_state = build_tracking_state(
                        pose_result["image_points_in"],
                        pose_result["object_points_in"],
                        pose_result["pids_in"],
                        pose_result["Tcw"],
                        priority_pids=tracking_priority_pids,
                    )
                    frames_since_global = 0
                else:
                    track_state = None
                    frames_since_global = TRACK_RELOCALIZE_INTERVAL

            out = frame.copy()
            debug_overlay = frame.copy()
            active_objects: List[str] = []

            if pose_result is not None and pose_result["ok"]:
                Tcw_cur = pose_result["Tcw"]
                cur_pid_to_xy = build_pid_xy_map(
                    pose_result.get("pids_eval", pose_result["pids_in"]),
                    pose_result.get("image_points_eval", pose_result["image_points_in"]),
                )
                ordered_objects = sorted(registered_objects, key=lambda x: (-x["priority"], x["name"]))
                for obj in ordered_objects:
                    out, auto_mask, cur_corners, valid = render_object_on_frame(
                        obj,
                        out,
                        Tcw_cur,
                        K_undist,
                        cur_pid_to_xy,
                        pose_result.get("mode", "global"),
                    )
                    if valid:
                        active_objects.append(obj["name"])
                        debug_overlay = draw_corners(debug_overlay, cur_corners, color=(0, 255, 0))
                        tmp = debug_overlay.copy()
                        tmp[auto_mask > 127] = (
                            0.6 * tmp[auto_mask > 127] + 0.4 * np.array([0, 255, 0], dtype=np.float32)
                        ).astype(np.uint8)
                        debug_overlay = tmp

            now = time.perf_counter()
            dt = max(now - last_time, 1e-6)
            fps = 1.0 / dt
            last_time = now
            debug_overlay = annotate_debug(debug_overlay, pose_result, active_objects, fps)

            if writer is None:
                writer = make_video_writer(OUTPUT_VIDEO_PATH, (base_w, base_h))
            if writer is not None:
                writer.write(out)
            if debug_writer is None:
                debug_writer = make_video_writer(DEBUG_VIDEO_PATH, (base_w, base_h))
            if debug_writer is not None:
                debug_writer.write(debug_overlay)

            if SAVE_DEBUG_FRAME_EVERY > 0 and (frame_idx % SAVE_DEBUG_FRAME_EVERY == 0):
                cv2.imwrite(os.path.join(DEBUG_DIR, "frames", f"debug_{frame_idx:06d}.png"), debug_overlay)
                cv2.imwrite(os.path.join(DEBUG_DIR, "frames", f"out_{frame_idx:06d}.png"), out)

            if SHOW_DEBUG_WINDOW:
                cv2.imshow(WINDOW_DEBUG, debug_overlay)
                cv2.imshow(WINDOW_OUTPUT, out)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
            else:
                if (frame_idx % 10) == 0:
                    mode = "none" if pose_result is None else pose_result.get("mode", "unknown")
                    status = "pose_ok" if (pose_result is not None and pose_result["ok"]) else "pose_lost"
                    print(f"[INFO] frame={frame_idx} {mode} {status} objects={active_objects}")

            prev_gray = frame_gray
            frame_idx += 1

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if debug_writer is not None:
            debug_writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()