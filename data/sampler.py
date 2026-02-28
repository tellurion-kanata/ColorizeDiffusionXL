import random
import os.path as osp
from collections import defaultdict

ASPECT_PORTRAIT = 0
ASPECT_LANDSCAPE = 1
ASPECT_SQUARE = 2


class CustomSampler:
    """
    Custom sampler that ensures even sampling from multiple zip files.
    batch_size must be a multiple of sample_per_zip.
    """

    def __init__(self, dataset, batch_size, sample_per_zip, shuffle=True, seed=42):
        if batch_size % sample_per_zip != 0:
            raise ValueError(f"batch_size ({batch_size}) must be a multiple of sample_per_zip ({sample_per_zip})")

        self.dataset = dataset
        self.batch_size = batch_size
        self.sample_per_zip = sample_per_zip
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        # Group indices by zip file
        zip_groups_raw = defaultdict(list)
        for idx, item in enumerate(dataset.data_items):
            filename = item[0]
            zip_id = osp.dirname(filename)
            zip_groups_raw[zip_id].append(idx)
        
        # Filter out zip files with insufficient samples
        self.zip_groups = {
            zip_id: indices for zip_id, indices in zip_groups_raw.items()
            if len(indices) >= sample_per_zip
        }
        
        if len(self.zip_groups) == 0:
            raise ValueError(f"No zip files have at least {sample_per_zip} samples")
        
        self.zip_ids = list(self.zip_groups.keys())
        self.num_zips = len(self.zip_ids)
        self.zips_per_batch = batch_size // sample_per_zip
        
        # Calculate segments for each zip (ensuring all samples are used)
        self.zip_segments = {}
        total_segments = 0
        for zip_id in self.zip_ids:
            num_samples = len(self.zip_groups[zip_id])
            # Each zip provides ceil(num_samples / sample_per_zip) segments
            segments = (num_samples + sample_per_zip - 1) // sample_per_zip
            self.zip_segments[zip_id] = segments
            total_segments += segments
        
        # Calculate total number of batches
        self.num_batches = total_segments // self.zips_per_batch

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        # Create segments for each zip file
        all_segments = []

        for zip_id in self.zip_ids:
            zip_indices = self.zip_groups[zip_id][:]
            if self.shuffle:
                rng.shuffle(zip_indices)
            
            # Create segments ensuring all samples are used
            num_samples = len(zip_indices)
            segments_count = self.zip_segments[zip_id]
            
            for seg_idx in range(segments_count):
                segment = []
                for i in range(self.sample_per_zip):
                    # Use modulo to cycle through samples if needed
                    sample_idx = (seg_idx * self.sample_per_zip + i) % num_samples
                    segment.append(zip_indices[sample_idx])
                all_segments.append(segment)
        
        # Shuffle segments if needed
        if self.shuffle:
            rng.shuffle(all_segments)
        
        # Create batches from segments
        for batch_idx in range(self.num_batches):
            batch_indices = []
            start_seg = batch_idx * self.zips_per_batch
            end_seg = start_seg + self.zips_per_batch
            
            for seg_idx in range(start_seg, end_seg):
                if seg_idx < len(all_segments):
                    batch_indices.extend(all_segments[seg_idx])
            
            yield batch_indices
    
    def __len__(self):
        return self.num_batches


class AspectRatioSampler:
    def __init__(self, dataset, batch_size, crop_sizes, shuffle=True, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        if isinstance(crop_sizes, int):
            self.crop_sizes = [crop_sizes]
        elif isinstance(crop_sizes, (list, tuple)):
            self.crop_sizes = crop_sizes
        else:
            self.crop_sizes = []
            for cs in crop_sizes:
                self.crop_sizes.append(list(cs))
        
        self.landscape_indices = getattr(dataset, 'landscape_indices', [])
        self.portrait_indices = getattr(dataset, 'portrait_indices', [])
        self.square_indices = getattr(dataset, 'square_indices', [])
        
        self.landscape_batches = len(self.landscape_indices) // batch_size
        self.portrait_batches = len(self.portrait_indices) // batch_size
        self.square_batches = len(self.square_indices) // batch_size
        
        self.total_batches = self.landscape_batches + self.portrait_batches + self.square_batches

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _get_aspect_ratio_type(self, crop_size):
        if isinstance(crop_size, int):
            h = w = crop_size
        else:
            h, w = crop_size
        
        if h > w:
            return ASPECT_PORTRAIT
        elif w > h:
            return ASPECT_LANDSCAPE
        else:
            return ASPECT_SQUARE
    
    def _get_aspect_ratio(self, idx):
        item = self.dataset.data_items[idx]
        width, height = item[4], item[5]
        return width / height
    
    def _get_filename(self, idx):
        item = self.dataset.data_items[idx]
        return item[0]
    
    def _sort_by_aspect_ratio(self, indices, rng):
        """
        Sort by aspect ratio, then add random offset to disperse duplicate files.
        Uses (aspect_ratio, random_value, index) as sort key to ensure:
        1. Primary sort by aspect ratio for efficient batching
        2. Random secondary sort to disperse duplicates with same aspect ratio
        3. Stable fallback to original index
        """
        random_offsets = {idx: rng.random() for idx in indices}
        sorted_indices = sorted(
            indices,
            key=lambda idx: (self._get_aspect_ratio(idx), random_offsets[idx], idx)
        )
        return sorted_indices

    def _block_shuffle(self, sorted_indices, rng, block_multiplier=8):
        """
        Shuffle within blocks of similar aspect ratios.
        Sorted indices are divided into blocks of (batch_size * block_multiplier),
        then shuffled within each block to break source adjacency while keeping
        aspect ratios compatible for crop selection.
        """
        block_size = self.batch_size * block_multiplier
        indices = list(sorted_indices)
        for start in range(0, len(indices), block_size):
            block = indices[start:start + block_size]
            rng.shuffle(block)
            indices[start:start + block_size] = block
        return indices

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        all_batches = []

        for indices_pool, num_batches in [
            (self.landscape_indices, self.landscape_batches),
            (self.portrait_indices, self.portrait_batches),
            (self.square_indices, self.square_batches),
        ]:
            if num_batches > 0:
                indices = self._sort_by_aspect_ratio(indices_pool, rng)
                indices = self._block_shuffle(indices, rng)
                for i in range(num_batches):
                    batch_indices = indices[i * self.batch_size:(i + 1) * self.batch_size]
                    all_batches.append(batch_indices)

        # Shuffle all batches
        if self.shuffle:
            rng.shuffle(all_batches)

        for batch in all_batches:
            yield batch
    
    def __len__(self):
        return self.total_batches

