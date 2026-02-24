

import h5py
import numpy as np
import os
import fnmatch
from tqdm import tqdm
import cv2

# ===============utils=================
def hdf5_groups_to_dict(hdf5_path):
    """
    读取 HDF5 文件，返回真正的嵌套 dict
    - dict.keys() 只包含第一层
    - 子 group / dataset 保持原始层级
    """
    import h5py

    def read_group(group):
        out = {}
        for key, item in group.items():
            if isinstance(item, h5py.Dataset):
                out[key] = item[()]
            elif isinstance(item, h5py.Group):
                out[key] = read_group(item)
        return out

    with h5py.File(hdf5_path, "r") as f:
        result = read_group(f)

    return result

def get_item(Dict_data: dict, item):
    if isinstance(item, str):
        keys = item.split(".")
        data = Dict_data
        for key in keys:
            data = data[key]
    elif isinstance(item, list):
        key_item = None
        for it in item:
            now_data = get_item(Dict_data, it)
            # import pdb;pdb.set_trace()
            if key_item is None:
                key_item = now_data
            else:
                key_item = np.column_stack((key_item, now_data))
        data = key_item
    else:
        raise ValueError(f"input type is not allow!")
    return data

def get_files(directory, extension):
    """使用pathlib获取所有匹配的文件"""
    file_paths = []
    for root, _, files in os.walk(directory):
            for filename in fnmatch.filter(files, extension):
                file_path = os.path.join(root, filename)
                file_paths.append(file_path)
    return file_paths
# ===============end utils=================

map = {
    "cam_high": "cam_head.color",
    "cam_left_wrist": "cam_left_wrist.color",
    "cam_right_wrist": "cam_right_wrist.color",
    "qpos": ["left_arm.joint","left_arm.gripper","right_arm.joint","right_arm.gripper", "slamware.move_velocity"],
}

def images_encoding(imgs):
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode('.jpg', imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b'\0'))
    return encode_data, max_len

def convert(hdf5_paths, output_path, start_index=0):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    index = start_index
    for hdf5_path in hdf5_paths:
        data = hdf5_groups_to_dict(hdf5_path)
        
        hdf5_output_path = os.path.join(output_path, f"episode_{index}.hdf5")
        index += 1
        print(data.keys())
        with h5py.File(hdf5_output_path, "w") as f:
            # 降采样
            input_data = {}

            for key in map.keys():
                input_data[key] = get_item(data, map[key])[:]

            qpos = np.array(input_data["qpos"]).astype(np.float32)
            
            actions = np.zeros_like(qpos)
            actions[:-1] = qpos[1:]
            # 最后一帧结束无动作（保持零向量）
            f.create_dataset('action', data=actions.astype(np.float32), dtype="float32")

            obs = f.create_group("observations")
            '''
            Basic robot arm parameters: if you're using joint values, 
            you can rename them to avoid confusion instead of calling them qpos, 
            but remember to update the corresponding model's data loading phase accordingly.
            '''

            obs.create_dataset('qpos', data=np.array(qpos), dtype="float32")
            obs.create_dataset("left_arm_dim", data=np.array(6))
            obs.create_dataset("right_arm_dim", data=np.array(6))

            images = obs.create_group("images")
            
            # Retrieve data based on your camera/view names, then encode and compress it for storage.
            def decode(imgs):
                if isinstance(imgs, np.ndarray) and imgs.ndim == 4:
                    return imgs

                imgs_array = []

                for data in imgs:
                    if isinstance(data, (bytes, bytearray)):
                        data = np.frombuffer(data, dtype=np.uint8)

                    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    if img is None:
                        raise ValueError("Failed to decode JPEG image")

                    # imgs_array.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    imgs_array.append(img)

                return np.stack(imgs_array, axis=0)


            cam_high = decode(input_data["cam_high"])
            cam_left_wrist = decode(input_data["cam_left_wrist"])
            cam_right_wrist = decode(input_data["cam_right_wrist"])
            
            images.create_dataset("cam_high", data=np.stack(cam_high), dtype=np.uint8)
            images.create_dataset("cam_right_wrist", data=np.stack(cam_right_wrist), dtype=np.uint8)
            images.create_dataset("cam_left_wrist", data=np.stack(cam_left_wrist), dtype=np.uint8)

        print(f"convert {hdf5_path} to rdt data format at {hdf5_output_path}")

if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description='Transform datasets typr to HDF5.')
    parser.add_argument('data_path', type=str,
                        help="your data dir like: datasets/task/")
    parser.add_argument('outout_path', type=str,default=None,
                        help='output path commanded like datasets/RDT/...')
    
    args = parser.parse_args()
    data_path = args.data_path
    output_path = args.outout_path

    if output_path is None:
        data_config = json.load(os.path.join(data_path, "config.json"))
        output_path = f"./datasets/RDT/{data_config['task_name']}"
    
    hdf5_paths = get_files(data_path, "*.hdf5")
    print("hdf5 files:\n",hdf5_paths)
    convert(hdf5_paths, output_path)