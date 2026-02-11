"""
简易 HDF5 结构查看与调试工具（不打印数据内容，只展示结构信息）。

使用方法（示例）：
1) 常规运行：
	python tools/read_hdf5.py --file tools/63.hdf5 --dataset /images

2) 进入 pdb 单步调试（推荐）：
	python -m pdb tools/read_hdf5.py --file tools/63.hdf5 --dataset /images

进入 pdb 后，可使用以下命令：
- l  查看当前代码上下文
- n  单步执行
- p obj  查看变量（例如 obj、data）
- c  继续运行到下一个断点
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py


def describe_item(name: str, obj: h5py.Dataset | h5py.Group) -> None:
	"""打印数据集或分组的简要信息（只展示结构）。"""
	if isinstance(obj, h5py.Dataset):
		shape = obj.shape
		dtype = obj.dtype
		print(f"[DATASET] {name} | shape={shape} dtype={dtype}")
	elif isinstance(obj, h5py.Group):
		print(f"[GROUP] {name} | keys={list(obj.keys())}")


def explore_file(file_path: Path, dataset: str | None) -> None:
	if not file_path.exists():
		raise FileNotFoundError(f"文件不存在: {file_path}")

	with h5py.File(file_path, "r") as f:
		print(f"已打开 HDF5 文件: {file_path}")
		# breakpoint()
		if dataset:
			obj = f[dataset]
			describe_item(dataset, obj)

			# 在这里设置断点，进入 pdb 后可以检查 obj、data 等变量。
			# 示例：在命令行运行 `python -m pdb tools/read_hdf5.py --file tools/63.hdf5 --dataset /cam_head`
			breakpoint()

			if isinstance(obj, h5py.Dataset):
				data = obj[()]
				print(f"读取完成: dataset={dataset}, shape={obj.shape}, dtype={obj.dtype}")
			else:
				print(f"目标是分组而非数据集: {dataset}")
		else:
			print("未指定 dataset，遍历文件结构：")
			f.visititems(lambda name, obj: describe_item(name, obj))


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="读取并调试 HDF5 文件的小工具",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument("--file", required=True, type=Path, help="HDF5 文件路径")
	parser.add_argument(
		"--dataset",
		type=str,
		default=None,
		help="要读取的 dataset 路径（例如 /images）。留空则仅打印文件结构",
	)
	return parser


def main() -> None:
	parser = build_arg_parser()
	args = parser.parse_args()
	explore_file(args.file, args.dataset)


if __name__ == "__main__":
	main()
