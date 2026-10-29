"""Horizontal rays and signed distances against circles and oriented boxes, batched over envs.

`obstacles` is a tuple (circles, circle_mask, boxes, box_mask): circles [E, C, 3] as (x, y, r),
boxes [E, B, 6] as (x, y, cos, sin, half length, half width), masks [E, C] and [E, B] mark the
real primitives among the padding.
"""

import math

import torch

from dronerl import contract

INF = math.inf

# all pixels of an image column share one horizontal direction, so one ray per column is enough
U = torch.arange(contract.IMG_W) + 0.5
V = torch.arange(contract.IMG_H) + 0.5
COLUMN_ANGLES = torch.atan((contract.IMG_W / 2 - U) / contract.FOCAL)  # left is positive
ROW_SLOPES = (contract.IMG_H / 2 - V) / contract.FOCAL  # up is positive
LIDAR_ANGLES = torch.arange(contract.LIDAR_BEAMS) * 2 * math.pi / contract.LIDAR_BEAMS

# horizontal fan of the tilted student camera: image columns every FAN_STEP px, extended by 24 px
# (±60°) for pitch and roll; the column rays are part of it, so a level camera hits them exactly
FAN_STEP = 0.5
FAN_U = torch.arange(-24, contract.IMG_W + 24 - 1 + FAN_STEP / 2, FAN_STEP) + 0.5
FAN_ANGLES = torch.atan((contract.IMG_W / 2 - FAN_U) / contract.FOCAL)
FAN_COLUMNS = ((U - FAN_U[0]) / FAN_STEP).round().long()
SAME_SURFACE = 0.1  # relative difference of neighbouring fan rays still on one surface


def boxes_from_maps(boxes):
    """(x, y, hx, hy, angle) -> (x, y, cos, sin, hx, hy)."""
    angle = boxes[..., 4:5]
    return torch.cat([boxes[..., :2], angle.cos(), angle.sin(), boxes[..., 2:4]], -1)


def to_box_frame(x, y, boxes):
    """Vector components [E, ..., B] in the world -> components in the frame of each box."""
    c, s = boxes[..., 2], boxes[..., 3]
    return c * x + s * y, c * y - s * x


def cast(origin, angles, obstacles):
    """Distance [E, R] along horizontal rays to the first primitive, inf without a hit.

    A ray starting inside a primitive hits it at distance 0.
    """
    circles, circle_mask, boxes, box_mask = obstacles
    dx, dy = angles.cos()[..., None], angles.sin()[..., None]  # [E, R, 1]

    rel = origin[:, None] - circles[..., :2]  # [E, C, 2]
    b = dx * rel[:, None, :, 0] + dy * rel[:, None, :, 1]  # [E, R, C]
    disc = b**2 - ((rel**2).sum(-1) - circles[..., 2] ** 2)[:, None]
    root = disc.clamp(min=0).sqrt()
    hit = (disc >= 0) & (root - b > 0) & circle_mask[:, None]
    t_circle = torch.where(hit, (-b - root).clamp(min=0), INF).amin(-1)

    rel = origin[:, None] - boxes[..., :2]  # [E, B, 2]
    ox, oy = (o[:, None] for o in to_box_frame(rel[..., 0], rel[..., 1], boxes))
    b = boxes[:, None]
    ux, uy = to_box_frame(dx, dy, b)  # [E, R, B]
    # slab method; a zero component means infinite slab distances, which is the right limit
    ux = torch.where(ux.abs() < 1e-12, 1e-12, ux)
    uy = torch.where(uy.abs() < 1e-12, 1e-12, uy)
    tx1, tx2 = (-b[..., 4] - ox) / ux, (b[..., 4] - ox) / ux
    ty1, ty2 = (-b[..., 5] - oy) / uy, (b[..., 5] - oy) / uy
    near = torch.maximum(torch.minimum(tx1, tx2), torch.minimum(ty1, ty2))
    far = torch.minimum(torch.maximum(tx1, tx2), torch.maximum(ty1, ty2))
    hit = (far >= near.clamp(min=0)) & box_mask[:, None]
    t_box = torch.where(hit, near.clamp(min=0), INF).amin(-1)
    return torch.minimum(t_circle, t_box)


def distance(points, obstacles):
    """Signed distance [E] from points [E, 2] to the nearest primitive, negative inside."""
    circles, circle_mask, boxes, box_mask = obstacles
    d_circle = (points[:, None] - circles[..., :2]).norm(dim=-1) - circles[..., 2]
    d_circle = torch.where(circle_mask, d_circle, INF).amin(-1)

    rel = points[:, None] - boxes[..., :2]
    x, y = to_box_frame(rel[..., 0], rel[..., 1], boxes)
    qx, qy = x.abs() - boxes[..., 4], y.abs() - boxes[..., 5]
    d_box = torch.hypot(qx.clamp(min=0), qy.clamp(min=0)) + torch.maximum(qx, qy).clamp(max=0)
    d_box = torch.where(box_mask, d_box, INF).amin(-1)
    return torch.minimum(d_circle, d_box)


def depth_image(columns, camera_height, obstacle_height):
    """Metric depth [E, H, W] along the optical axis of a level camera.

    columns [E, W] are the horizontal distances to the first obstacle in each column. Obstacles
    are prisms from the floor to obstacle_height [E], so a ray rising above the first one also
    passes above everything behind it, and the floor is the only other surface.
    """
    slopes = ROW_SLOPES.to(columns.device)[:, None]  # [H, 1]
    z_obs = (columns * COLUMN_ANGLES.to(columns.device).cos())[:, None]  # [E, 1, W]
    below_top = camera_height[:, None, None] + z_obs * slopes <= obstacle_height[:, None, None]
    z_obs = torch.where(below_top, z_obs, INF)
    z_floor = torch.where(slopes < 0, camera_height[:, None, None] / -slopes, INF)  # [E, H, 1]
    return torch.minimum(z_obs, z_floor)


def tilted_depth(fan, camera_height, obstacle_height, pitch, roll):
    """Metric depth [E, H, W] along the optical axis of a camera with pitch [E] (down is positive)
    and roll [E] (right side down is positive) in the heading frame.

    fan [E, R] are the horizontal distances along FAN_ANGLES. A pixel interpolates between the two
    rays around its azimuth when they see the same surface and takes the nearer one in azimuth
    otherwise, so oblique walls stay exact and edges stay sharp. The prism argument of depth_image
    holds for any ray direction, so only the azimuth is discretised.
    """
    dev = fan.device
    left = ((contract.IMG_W / 2 - U.to(dev)) / contract.FOCAL)[None, None, :]
    up = ROW_SLOPES.to(dev)[None, :, None]
    cp, sp = pitch.cos()[:, None, None], pitch.sin()[:, None, None]
    cr, sr = roll.cos()[:, None, None], roll.sin()[:, None, None]
    # pixel ray (1, left, up) of the camera rotated by R_y(pitch) R_x(roll), FLU
    y, z = left * cr - up * sr, left * sr + up * cr
    x, z = (cp + z * sp).clamp(min=1e-6), z * cp - sp
    h = torch.hypot(x, y)
    u = contract.IMG_W / 2 - contract.FOCAL * y / x
    pos = ((u - FAN_U[0].item()) / FAN_STEP).clamp(0, len(FAN_U) - 1)
    k = pos.floor().long().clamp(max=len(FAN_U) - 2)
    w = pos - k
    r0 = fan.gather(1, k.flatten(1)).view_as(k)
    r1 = fan.gather(1, (k + 1).flatten(1)).view_as(k)
    same = (r0 - r1).abs() <= SAME_SURFACE * torch.minimum(r0, r1)
    r = torch.where(same, r0 + w * (r1 - r0), torch.where(w < 0.5, r0, r1))
    # the forward component of the pixel ray is 1, so depth is distance along it over |ray|_xy
    below_top = camera_height[:, None, None] + r * z / h <= obstacle_height[:, None, None]
    z_obs = torch.where(below_top, r / h, INF)
    z_floor = torch.where(z < 0, camera_height[:, None, None] / -z, INF)
    return torch.minimum(z_obs, z_floor)
