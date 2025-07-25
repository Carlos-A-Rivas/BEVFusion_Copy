#!/usr/bin/env python3
"""Extract a synchronized image and point cloud from an MCAP file.

This helper loads the first messages on the given topics *after* the
provided timestamp and saves them in the tensor format used by
BEVFusion.  The output directory will contain ``points.tensor``,
``images.tensor`` and a ``0-image.jpg`` file.  Use ``--ros-distro``
to select the ROS 2 message definitions that match your bag.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from tensor import save


DTYPE_MAP = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    7: np.float32,
    8: np.float64,
}


def pointcloud2_to_array(msg) -> np.ndarray:
    """Convert ``sensor_msgs/msg/PointCloud2`` to a structured array."""
    dtype = np.dtype({
        'names': [f.name for f in msg.fields],
        'formats': [DTYPE_MAP[f.datatype] for f in msg.fields],
        'offsets': [f.offset for f in msg.fields],
        'itemsize': msg.point_step,
    })
    arr = np.frombuffer(msg.data, dtype=dtype, count=msg.width * msg.height)
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path, help='Path to MCAP directory or file')
    parser.add_argument('--cam-topic', required=True)
    parser.add_argument('--lidar-topic', required=True)
    parser.add_argument('--timestamp', type=float, required=True,
                        help='Unix timestamp (sec) to seek to')
    parser.add_argument('-o', '--out-dir', type=Path, default=Path('custom-example'))
    parser.add_argument('--ros-distro', default='humble',
                        help='ROS 2 distro for message definitions (e.g. foxy, humble)')
    args = parser.parse_args()

    ts_ns = int(args.timestamp * 1e9)

    try:
        store_enum = getattr(Stores, f'ROS2_{args.ros_distro.upper()}')
    except AttributeError:
        raise ValueError(f'Unknown ROS 2 distro: {args.ros_distro}')
    store = get_typestore(store_enum)
    with AnyReader([args.bag]) as reader:
        reader.open()
        reader.typestore = store
        cam_conn = next(c for c in reader.connections if c.topic == args.cam_topic)
        lidar_conn = next(c for c in reader.connections if c.topic == args.lidar_topic)

        cam_msg = None
        for conn, ts, raw in reader.messages(connections=[cam_conn], start=ts_ns):
            cam_msg = reader.deserialize(raw, conn.msgtype)
            break
        lidar_msg = None
        for conn, ts, raw in reader.messages(connections=[lidar_conn], start=ts_ns):
            lidar_msg = reader.deserialize(raw, conn.msgtype)
            break

    if cam_msg is None or lidar_msg is None:
        raise RuntimeError('No messages found after the requested timestamp')

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Handle image message (compressed or raw)
    if hasattr(cam_msg, 'format'):
        img = Image.open(BytesIO(cam_msg.data.tobytes()))
    else:
        array = np.array(cam_msg.data).reshape(cam_msg.height, cam_msg.width, -1)
        img = Image.fromarray(array)
    img.save(args.out_dir / '0-image.jpg')

    # Preprocess image to BEVFusion resolution (256x704)
    img_tensor = np.asarray(img.resize((704, 256))).astype(np.float32) / 255.0
    img_tensor = img_tensor.transpose(2, 0, 1)[None, None]
    save(img_tensor.astype(np.float16), args.out_dir / 'images.tensor')

    # Point cloud
    pc = pointcloud2_to_array(lidar_msg)
    fields = pc.dtype.names
    xyzi = np.stack([pc['x'], pc['y'], pc['z'], pc[fields[3]] if len(fields) > 3 else np.zeros(len(pc))], axis=1)
    fifth = np.zeros((xyzi.shape[0], 1), dtype=np.float32)
    save(np.hstack([xyzi, fifth]).astype(np.float16), args.out_dir / 'points.tensor')


if __name__ == '__main__':
    main()
