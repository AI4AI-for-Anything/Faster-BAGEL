# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

from .interleave_datasets import UnifiedEditIterableDataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
}


DATASET_INFO = {
    't2i_pretrain': {
        't2i': {
            'data_dir': 'your_data_path/bagel_example/t2i', # path of the parquet files
            'num_files': 10, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 1000, # number of total samples in the dataset
        },
    },
    'unified_edit':{
        'seedxedit_multi': {
            'data_dir': 'your_data_path/bagel_example/editing/seedxedit_multi',
            'num_files': 10,
            'num_total_samples': 1000,
            "parquet_info_path": 'your_data_path/bagel_example/editing/parquet_info/seedxedit_multi_nas.json', # information of the parquet files
		},
    },
    'vlm_sft': {
        'llava_ov': {
			'data_dir': 'your_data_path/bagel_example/vlm/images',
			'jsonl_path': 'your_data_path/bagel_example/vlm/llava_ov_si.jsonl',
			'num_total_samples': 1000
		},
    },
}

# Optional portable paths. Without overrides, keep the upstream example paths.
if os.environ.get("BAGEL_DATA_ROOT"):
    _root = Path(os.environ["BAGEL_DATA_ROOT"]).expanduser().resolve()
    DATASET_INFO["t2i_pretrain"]["t2i"]["data_dir"] = str(_root / "t2i")
    DATASET_INFO["unified_edit"]["seedxedit_multi"]["data_dir"] = str(_root / "editing/seedxedit_multi")
    DATASET_INFO["unified_edit"]["seedxedit_multi"]["parquet_info_path"] = str(_root / "editing/parquet_info/seedxedit_multi_nas.json")
    DATASET_INFO["vlm_sft"]["llava_ov"]["data_dir"] = str(_root / "vlm/images")
    DATASET_INFO["vlm_sft"]["llava_ov"]["jsonl_path"] = str(_root / "vlm/llava_ov_si.jsonl")
if os.environ.get("BAGEL_TRAIN_JSONL"):
    DATASET_INFO["vlm_sft"]["llava_ov"]["jsonl_path"] = os.environ["BAGEL_TRAIN_JSONL"]
if os.environ.get("BAGEL_EDIT_META"):
    DATASET_INFO["unified_edit"]["seedxedit_multi"]["parquet_info_path"] = os.environ["BAGEL_EDIT_META"]
