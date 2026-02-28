import json
import zipfile
import os.path as osp
import warnings

import pandas as pd
from glob import glob

import torch
from PIL import ImageFile
from .preprocessing import *
from .sampler import CustomSampler, AspectRatioSampler
from .collate import CustomCollateFn

import torch.utils.data as data

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=UserWarning)  # Too many RGBA warnings

DEFAULT_AESTHETIC_SCORE = 6


class TripletDataset(data.Dataset):
    def __init__(
            self,
            dataroot: str = None,
            mode: str = "train",
            image_key: str = "color",
            control_key: Union[str|list[str]] = "sketch",
            condition_key: str = "reference",
            text_key: str = None,
            json_key: str = "data_index",
            json_files: Union[str|list[str]] = None,
            score_threshold: float = 5,
            minimum_image_size: int = 768,
            offset = 4,
            use_real_reference = False,
            multi_reference = False,
            gt_path: str = None,
    ):
        super().__init__()
        if isinstance(control_key, str):
            control_key = [control_key]
        else:
            control_key = list(control_key)

        dataroot = osp.abspath(dataroot)
        self.dataroot = dataroot
        self.sketch_root = [osp.join(dataroot, key) for key in control_key]
        self.color_root = osp.abspath(gt_path) if exists(gt_path) else osp.join(dataroot, image_key)
        self.ref_root = osp.join(dataroot, condition_key)
        self.text_root = osp.join(dataroot, text_key) if exists(text_key) else None
        self.use_real_reference = use_real_reference
        self.multi_reference = multi_reference

        print(f"dataroot: {dataroot}, json_key: {json_key}, json_files: {json_files}")
        self.load_image_list(osp.join(dataroot, json_key), json_files, score_threshold, minimum_image_size)
        self.data_size = len(self)
        self.offset = offset if mode == "validation" else 0

        # self.offset = random.randint(1, self.data_size) if mode == 'validation' else 0

    def load_image_list(self, dataroot, json_files, score_threshold, minimum_image_size):
        print(f"loading image list from {dataroot} with {json_files}")
        if exists(json_files):
            all_files = []
            for jf in json_files:
                parquet_path = osp.join(dataroot, f"{jf}.parquet")
                json_path = osp.join(dataroot, f"{jf}.json")

                if osp.exists(parquet_path):
                    df = pd.read_parquet(parquet_path)
                    df = df[
                        (df['aesthetic_score'] >= score_threshold) &
                        (df['exist_sketch'] == True) &
                        (df['width'] >= minimum_image_size) &
                        (df['height'] >= minimum_image_size)
                    ]
                    all_files.extend(df['image_path'].tolist())
                elif osp.exists(json_path):
                    d = json.load(open(json_path, "r"))
                    all_files.extend([
                        file for file in d
                        if check_json(d[file], score_threshold, minimum_image_size)
                    ])
                else:
                    raise FileNotFoundError(
                        f"Neither {parquet_path} nor {json_path} exists."
                    )

            self.data_items = [
                osp.join(self.sketch_root[0], file) for file in all_files
            ]

        else:
            self.data_items = [
                file for ext in IMAGE_EXTENSIONS
                for file in glob(osp.join(self.sketch_root[0], f'*.{ext}'), recursive=True)
            ]

    def get_images(self, index):
        filename = self.data_items[index]

        ske = Image.open(filename).convert('RGB')
        col = Image.open(filename.replace(self.sketch_root[0], self.color_root)).convert('RGB')
        w, h = col.size

        if self.offset > 0:
            ref_file = self.data_items[(index + self.offset) % self.data_size]
            ref = Image.open(ref_file.replace(self.sketch_root[0], self.color_root)).convert('RGB')
        else:
            ref = Image.open(filename.replace(self.sketch_root[0], self.ref_root)).convert('RGB')

        return {
            "control": ske,
            "image": col,
            "reference": [ref],
            "size": [h, w],
            "aesthetic_score": torch.tensor(DEFAULT_AESTHETIC_SCORE),
        }

    def __getitem__(self, index):
        return self.get_images(index)

    def __len__(self):
        return len(self.data_items)


class ZipTripletDataset(TripletDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zip_sketch = None
        self.zip_color = None
        self.zip_text = None
        self.current_zipid = None

    def _append_item(self, file, aesthetic_score, tags, reference_list, width, height):
        """Append a data item with reference handling and aspect ratio categorization."""
        if exists(reference_list) and len(reference_list) > 0 and self.use_real_reference:
            if self.multi_reference:
                for ref in reference_list:
                    current_idx = len(self.data_items)
                    self.data_items.append([file, aesthetic_score, tags, ref, width, height])
                    self._categorize_by_aspect_ratio(current_idx, width, height)
            else:
                current_idx = len(self.data_items)
                self.data_items.append([file, aesthetic_score, tags, reference_list, width, height])
                self._categorize_by_aspect_ratio(current_idx, width, height)
        else:
            current_idx = len(self.data_items)
            self.data_items.append([file, aesthetic_score, tags, None, width, height])
            self._categorize_by_aspect_ratio(current_idx, width, height)

    def _load_from_parquet(self, parquet_path, score_threshold, minimum_image_size):
        """Load and filter data items from a parquet file."""
        df = pd.read_parquet(parquet_path)
        df = df[
            (df['aesthetic_score'] >= score_threshold) &
            (df['exist_sketch'] == True) &
            (df['width'] >= minimum_image_size) &
            (df['height'] >= minimum_image_size)
        ]
        has_ref_col = 'reference_list' in df.columns
        has_caption_col = 'caption' in df.columns

        for row in df.itertuples(index=False):
            file = row.image_path
            aesthetic_score = row.aesthetic_score
            tags = row.caption if has_caption_col else ''
            width = row.width
            height = row.height

            reference_list = None
            if has_ref_col:
                raw_ref = row.reference_list
                if isinstance(raw_ref, str):
                    reference_list = json.loads(raw_ref)
                elif isinstance(raw_ref, list):
                    reference_list = raw_ref

            self._append_item(file, aesthetic_score, tags, reference_list, width, height)


    def _load_from_json(self, json_path, score_threshold, minimum_image_size):
        """Load and filter data items from a JSON file."""
        d = json.load(open(json_path, "r"))
        count = 0
        for file in d:
            if check_json(d[file], score_threshold, minimum_image_size):
                reference_list = d[file].get("reference_list", None)
                aesthetic_score = d[file].get("aesthetic_score", DEFAULT_AESTHETIC_SCORE)
                tags = d[file].get("tags", "")
                width = d[file].get("width", 1024)
                height = d[file].get("height", 1024)
                self._append_item(file, aesthetic_score, tags, reference_list, width, height)
                count += 1

    def load_image_list(self, dataroot, json_files, score_threshold, minimum_image_size):
        if exists(json_files):
            self.data_items = []
            self.landscape_indices = []
            self.portrait_indices = []
            self.square_indices = []

            for jf in json_files:
                parquet_path = osp.join(dataroot, f"{jf}.parquet")
                json_path = osp.join(dataroot, f"{jf}.json")

                if osp.exists(parquet_path):
                    self._load_from_parquet(parquet_path, score_threshold, minimum_image_size)
                elif osp.exists(json_path):
                    self._load_from_json(json_path, score_threshold, minimum_image_size)
                else:
                    raise FileNotFoundError(
                        f"Neither {parquet_path} nor {json_path} exists."
                    )

        else:
            self.data_items = []
            self.landscape_indices = []
            self.portrait_indices = []
            self.square_indices = []
            
            for zip_path in glob(osp.join(self.sketch_root[0], "*.zip"), recursive=True):
                for file in zipfile.ZipFile(zip_path).namelist():
                    if not file.endswith("/"):
                        current_idx = len(self.data_items)
                        self.data_items.append([file, DEFAULT_AESTHETIC_SCORE, "", None, 1024, 1024])
                        self.square_indices.append(current_idx)
    
    def _categorize_by_aspect_ratio(self, idx, width, height):
        if height > width:
            self.portrait_indices.append(idx)
        elif width > height:
            self.landscape_indices.append(idx)
        else:
            self.square_indices.append(idx)

    def fresh_zip_files(self, zipid):
        if self.current_zipid == zipid:
            return
            
        if exists(self.zip_sketch):
            self.zip_sketch.close()
        if exists(self.zip_color):
            self.zip_color.close()
        if exists(self.zip_text):
            self.zip_text.close()
        
        self.current_zipid = zipid
        sketch_root = self.sketch_root[0]
        self.zip_sketch, self.zip_color, self.zip_text = map(
            lambda t: zipfile.ZipFile(osp.join(t, f"{zipid}.zip"), "r") if exists(t) else None,
            (sketch_root, self.color_root, self.text_root)
        )

    def get_images(self, index):
        filename, aesthetic_score, tags, reference_filename, w, h = self.data_items[index]
        self.fresh_zip_files(osp.dirname(filename))

        with self.zip_sketch.open(filename, "r") as f:
            ske = Image.open(f).convert('RGB')
        with self.zip_color.open(filename, "r") as f:
            col = Image.open(f).convert('RGB')
        
        has_reference_list = exists(reference_filename)
        
        if has_reference_list and self.use_real_reference:
            if isinstance(reference_filename, list):
                reference_filename = random.choice(reference_filename)
            with self.zip_color.open(reference_filename, "r") as f:
                ref = Image.open(f).convert('RGB')
        else:
            if self.offset > 0:
                filename, aesthetic_score, tags, reference_filename, *_ = self.data_items[(index + self.offset) % self.data_size]
                self.fresh_zip_files(osp.dirname(filename))
            with self.zip_color.open(filename, "r") as f:
                ref = Image.open(f).convert('RGB')

        if exists(self.zip_text):
            with self.zip_text.open(filename, "r") as f:
                text = f.read().decode('utf-8')
        else:
            text = None

        return {
            "control": ske,
            "image": col,
            "reference": [ref],
            "size": [h, w],
            "text": text if exists(text) else tags,
            "aesthetic_score": torch.tensor(aesthetic_score),
        }

    def __del__(self):
        try:
            if exists(self.zip_sketch):
                self.zip_sketch.close()
            if exists(self.zip_color):
                self.zip_color.close()
            if exists(self.zip_text):
                self.zip_text.close()
        except:
            pass
        finally:
            self.zip_sketch = None
            self.zip_color = None  
            self.zip_text = None


class QuartetDataset(ZipTripletDataset):
    def __init__(
            self,
            dataroot: str,
            *args,
            **kwargs
    ):
        super().__init__(dataroot=dataroot, *args, **kwargs)
        self.mask_root = osp.join(osp.abspath(dataroot), "mask")
        self.zip_sketch = None
        self.zip_color = None
        self.zip_mask = None
        self.zip_text = None

    def fresh_zip_files(self, zipid):
        if self.current_zipid == zipid:
            return
            
        if exists(self.zip_sketch):
            self.zip_sketch.close()
        if exists(self.zip_color):
            self.zip_color.close()
        if exists(self.zip_mask):
            self.zip_mask.close()
        if exists(self.zip_text):
            self.zip_text.close()
            
        self.current_zipid = zipid
        sketch_root = self.sketch_root[0]
        self.zip_sketch, self.zip_color, self.zip_mask, self.zip_text = map(
            lambda t: zipfile.ZipFile(osp.join(t, f"{zipid}.zip"), "r") if exists(t) else None,
            (sketch_root, self.color_root, self.mask_root, self.text_root)
        )

    def get_images(self, index):
        filename, aesthetic_score, tags, reference_filename, width, height = self.data_items[index]
        self.fresh_zip_files(osp.dirname(filename))

        with self.zip_sketch.open(filename, "r") as f:
            ske = Image.open(f).convert('RGB')
        with self.zip_color.open(filename, "r") as f:
            col = Image.open(f).convert('RGB')
        with self.zip_mask.open(filename, "r") as f:
            mask = Image.open(f).convert('L')
        w, h = col.size

        has_reference_list = exists(reference_filename)
        
        if has_reference_list:
            if isinstance(reference_filename, list):
                reference_filename = random.choice(reference_filename)
            with self.zip_color.open(reference_filename, "r") as f:
                ref = Image.open(f).convert('RGB')
            with self.zip_mask.open(reference_filename, "r") as f:
                rmask = Image.open(f).convert('L')
        else:
            if self.offset > 0:
                ref_filename, aesthetic_score, tags, reference_filename, width, height = self.data_items[(index + self.offset) % self.data_size]
                self.fresh_zip_files(osp.dirname(ref_filename))
                with self.zip_color.open(ref_filename, "r") as f:
                    ref = Image.open(f).convert('RGB')
                with self.zip_mask.open(ref_filename, "r") as f:
                    rmask = Image.open(f).convert('L')
            else:
                with self.zip_color.open(filename, "r") as f:
                    ref = Image.open(f).convert('RGB')
                rmask = mask

        if exists(self.zip_text):
            with self.zip_text.open(filename, "r") as f:
                text = f.read().decode('utf-8')
        else:
            text = None

        return {
            "control": ske,
            "image": col,
            "reference": [ref],
            "smask": mask,
            "rmask": rmask,
            "has_reference_list": has_reference_list,
            "size": [h, w],
            "text": text if exists(text) else tags,
            "aesthetic_score": torch.tensor(aesthetic_score),
        }

    def __getitem__(self, index):
        while True:
            try:
                return self.get_images(index)

            except Exception as e:
                index += 1
                print(f"Cannot open file {self.data_items[index]} due to {e} !!!")

    def __del__(self):
        try:
            if exists(self.zip_color) and exists(self.zip_sketch) and exists(self.zip_mask):
                self.zip_sketch.close()
                self.zip_color.close()
                self.zip_mask.close()
        finally:
            self.zip_sketch = None
            self.zip_color = None
            self.zip_mask = None


def create_dataloader(opt, cfg, device_num, eval_load_size = None, sample_per_zip=1):
    DATALOADER = {
        "TripletLoader": TripletDataset,
        "ZipTripletLoader": ZipTripletDataset,
        "QuartLoader": QuartetDataset,
    }

    loader_cls = cfg["class"]
    assert loader_cls in DATALOADER.keys(), f"DataLoader {loader_cls} does not exist."
    loader = DATALOADER[loader_cls]
    
    # Get aspect_ratio_threshold from config or use default
    use_aspect_ratio_sampler = cfg.get("use_aspect_ratio_sampler", False)

    dataset = loader(
        mode = opt.mode,
        dataroot = opt.dataroot,
        **cfg.dataset_params
    )
    custom_collate = CustomCollateFn(
        eval=opt.eval, 
        eval_load_size=eval_load_size, 
        **cfg["transforms"]
    )
    
    # Choose sampler based on configuration
    if use_aspect_ratio_sampler and not opt.eval:
        crop_sizes = cfg["transforms"].get("crop_size", None)
        dataloader_sampler = AspectRatioSampler(
            dataset, opt.batch_size, crop_sizes, 
            shuffle=cfg.get("shuffle", True),
        ) if exists(crop_sizes) else None
    elif sample_per_zip > 1:
        dataloader_sampler = CustomSampler(
            dataset, opt.batch_size, sample_per_zip, shuffle=cfg.get("shuffle", True) and not opt.eval
        )
    else:
        dataloader_sampler = None
    
    if exists(dataloader_sampler):
        # When using batch_sampler, don't pass batch_size and shuffle separately
        dataloader = data.DataLoader(
            dataset = dataset,
            batch_size = 1,
            batch_sampler = dataloader_sampler,
            num_workers = opt.num_threads,
            drop_last = False,  # batch_sampler handles this
            pin_memory = True,
            prefetch_factor = 2 if opt.num_threads > 0 else None,
            collate_fn = custom_collate,
        )
    else:
        # Standard DataLoader without custom sampler
        dataloader = data.DataLoader(
            dataset = dataset,
            batch_size = opt.batch_size,
            shuffle = cfg.get("shuffle", True) and not opt.eval,
            num_workers = opt.num_threads,
            drop_last = device_num > 1,
            pin_memory = True,
            prefetch_factor = 2 if opt.num_threads > 0 else None,
            collate_fn = custom_collate,
        )
    return dataloader, len(dataset), dataloader_sampler