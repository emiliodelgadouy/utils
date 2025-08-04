import os
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.StatsRenderer import StatsRenderer

import math
from multiprocessing import Pool, cpu_count

class Dataset(pd.DataFrame):
    _metadata = ['_cache']

    def preload_cache(self, dtype=np.uint8):
        """
        Carga todas las imágenes en self._cache como numpy arrays de tipo `dtype`.
        Ahorrarás memoria usando uint8 (0–255) y convertirás a float justo antes del modelo.
        """
        for path in tqdm(self['path'].unique(), desc="Preloading cache"):
            if path in self._cache:
                continue
            try:
                img = Image.open(path).convert("L")
                arr = np.array(img, dtype=dtype)
                self._cache[path] = arr
            except Exception as e:
                print(f"Error cargando {path}: {e}")

    @property
    def _constructor(self):
        return Dataset

    def save_dump_parallel(cache: dict,
                           base_filename: str = 'dump_cache',
                           num_workers: int = None):
        """
        Divide `cache` en num_workers trozos y guarda cada uno en
        base_filename_0.npz, base_filename_1.npz, …
        """
        if num_workers is None:
            num_workers = cpu_count()
        items = list(cache.items())
        chunk_size = math.ceil(len(items) / num_workers)
        # crear lista de (items_chunk, filename)
        tasks = []
        for i in range(num_workers):
            chunk = items[i * chunk_size:(i + 1) * chunk_size]
            if not chunk:
                break
            fname = f"{base_filename}_{i}.npz"
            tasks.append((chunk, fname))
        with Pool(num_workers) as pool:
            pool.map(self._save_chunk, tasks)

    def _save_chunk(args):
        items, filename = args
        # items es lista de tuplas (clave, array)
        np.savez_compressed(filename, **dict(items))
    def save_dump(self, filename: str = None):
        if filename is None:
            filename = 'dump_cache.npz'
        np.savez_compressed(filename, cache=self._cache)

    def load_dump(self, filename: str = None):
        if filename is None:
            filename = 'dump_cache.npz'
        data = np.load(filename, allow_pickle=True)
        self._cache = data['cache'].item()

    def __init__(self, data=None, lateralize=False, reduced=False, n=100, *args, **kwargs):
        if data is None:
            self._cache = {}
            base = os.path.dirname(os.path.abspath(__file__))
            data = pd.read_csv(
                os.path.join(base, "data.csv"),
                usecols=[
                    'path', 'laterality', 'view',
                    'breast_birads', 'finding_birads', 'No_Finding',
                    'resized_xmin', 'resized_ymin',
                    'resized_xmax', 'resized_ymax', 'split'
                ],
                low_memory=False
            )
            data = data.rename(columns={
                'resized_xmin': 'xmin',
                'resized_ymin': 'ymin',
                'resized_xmax': 'xmax',
                'resized_ymax': 'ymax'
            })
            data['findings'] = 1 - data['No_Finding']
            data = data.drop(columns=['No_Finding'])
            data['breast_birads'] = data['breast_birads'].apply(Dataset.map_birads).astype("Int64")
            data['finding_birads'] = data['finding_birads'].apply(Dataset.map_birads).astype("Int64")

        if reduced:
            data = data.head(n)

        super().__init__(data, *args, **kwargs)

        self._current_page = 0
        self._page_size = 10
        self._render_width = 60

        # normalizar rutas
        self['path'] = self['path'].apply(self._normalize_path)

        if lateralize:
            self._apply_lateralization()

        # cache de PIL Images en memoria
        self._cache = {}

    def _normalize_path(self, path):
        return os.path.normpath(path.replace(
            "/content/VinDr-Mammo/images_png",
            "./utils/images_original"
        ))

    def _apply_lateralization(self):
        for idx, row in tqdm(self.iterrows(), total=len(self), desc="Lateralización"):
            if row['laterality'] == 'R':
                orig = row['path']
                stem, ext = os.path.splitext(orig)
                lat_path = f"{stem}_lateralized{ext}"
                try:
                    img = Image.open(orig)
                    w, _ = img.size
                    flip = img.transpose(Image.FLIP_LEFT_RIGHT)
                    if not os.path.exists(lat_path):
                        flip.save(lat_path)
                    xmin, xmax = row['xmin'], row['xmax']
                    self.at[idx, 'path'] = lat_path
                    self.at[idx, 'xmin'] = w - xmax
                    self.at[idx, 'xmax'] = w - xmin
                except Exception as e:
                    print(f"Error lateralizando {orig}: {e}")

    def _get_image(self, path, dtype=np.uint8):
        if path not in self._cache:
            img = Image.open(path).convert("L")
            arr = np.array(img, dtype=dtype)
            self._cache[path] = arr
            print("cache miss")
        else:
            # print("cache hit")
            arr = self._cache[path]
        arr3 = np.stack([arr] * 3, axis=-1)
        return arr3

    @staticmethod
    def _draw_bb(img, rois):
        xmin, ymin, xmax, ymax = list(map(int, rois))
        img = img.convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(
            [(xmin, ymin), (xmax, ymax)],
            fill=(255, 0, 0, 100),
            outline=(255, 0, 0, 255),
            width=2
        )
        return Image.alpha_composite(img, overlay)

    def sample_random_patch(self, path, patch_w, patch_h, rois=None, jitter_frac=0.25, draw_bounding_box=True):
        if draw_bounding_box and rois is not None and not any(pd.isna(rois)):
            img = self.full_image_with_rois(path, rois)
        else:
            img = self.full_image_without_rois(path)
        h, w, _ = img.shape
        if patch_w > w or patch_h > h:
            return None
        if rois is not None and not any(pd.isna(rois)):
            x, y = Dataset._patch_centered_jittered_roi(rois, w, h, patch_w, patch_h, jitter_frac)
        else:
            x, y = Dataset._patch_fully_random_coords(w, h, patch_w, patch_h)
        patch = img[y:y + patch_h, x:x + patch_w, :]
        return patch

    @staticmethod
    def _patch_centered_jittered_roi(rois, img_w, img_h, patch_w, patch_h, jitter_frac):
        xmin, ymin, xmax, ymax = list(map(int, rois))
        cx = (xmin + xmax) // 2
        cy = (ymin + ymax) // 2

        max_dx = int(patch_w * jitter_frac)
        max_dy = int(patch_h * jitter_frac)

        dx = random.randint(-max_dx, max_dx)
        dy = random.randint(-max_dy, max_dy)

        new_cx = cx + dx
        new_cy = cy + dy

        x = max(0, min(img_w - patch_w, new_cx - patch_w // 2))
        y = max(0, min(img_h - patch_h, new_cy - patch_h // 2))
        return x, y

    @staticmethod
    def _patch_fully_random_coords(img_w, img_h, patch_w, patch_h):
        x = random.randint(0, img_w - patch_w)
        y = random.randint(0, img_h - patch_h)
        return x, y

    def full_image(self, path, rois=None, draw_bounding_box=True, plot=False):
        if draw_bounding_box:
            return self.full_image_with_rois(path, rois)
        else:
            return self.full_image_without_rois(path)

    def plot(self, img):
        height, width = img.shape[:2]
        channels = img.shape[2] if img.ndim == 3 else 1
        min_px, max_px = img.min(), img.max()

        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.axis('off')

        info = (
            f"Size: {height}×{width} px\n"
            f"Channels: {channels}\n"
            f"Pixel range:[{min_px}] – [{max_px}]"
        )
        ax.text(
            0.99, 0.01, info,
            transform=ax.transAxes,
            fontsize=8,
            color='white',
            backgroundcolor='black',
            va='bottom',
            ha='right'
        )

        plt.show()
    def full_image_without_rois(self, path):
        img = self._get_image(path).copy()
        return img

    def full_image_with_rois(self, path, rois=None):
        img = self.full_image_without_rois(path)
        if rois is not None and not any(pd.isna(rois)):
            img_pil = Image.fromarray(img.astype(np.uint8))
            img_pil = Dataset._draw_bb(img_pil, rois)
            img = np.array(img_pil)
        return img

    @staticmethod
    def map_birads(cell):
        if pd.isna(cell):
            return pd.NA
        s = str(cell).strip()
        return {
            'BI-RADS 0': 0, 'BI-RADS 1': 1, 'BI-RADS 2': 2,
            'BI-RADS 3': 3, 'BI-RADS 4': 4, 'BI-RADS 5': 5,
            'BI-RADS 6': 6
        }.get(s, pd.NA)

    def to_3_channel_patch(self, path, width, height, preprocess_fn=None):
        img = self.sample_random_patch(path, patch_w=width, patch_h=height, draw_bounding_box=False)
        arr = np.array(img).astype(np.float32)
        return preprocess_fn(arr) if preprocess_fn else arr


@pd.api.extensions.register_dataframe_accessor("render")
class _RenderAccessor1:
    def __init__(self, pandas_obj):
        self._df = pandas_obj

    def __call__(self, width=400, page_size=10, render_sample_patch=True, render_sample_patch_height=400,
                 render_sample_patch_width=400):
        from utils.DatasetRenderer import DatasetRenderer
        renderer = DatasetRenderer(self._df, render_sample_patch=render_sample_patch,
                                   render_sample_patch_width=render_sample_patch_width,
                                   render_sample_patch_height=render_sample_patch_height)
        renderer.render(width=width, page_size=page_size)


@pd.api.extensions.register_dataframe_accessor("stats")
class _RenderAccessor2:
    def __init__(self, pandas_obj):
        self._df = pandas_obj

    def __call__(self, columns=None, ancho=20, alto=10, order=None, title=None, subtitle=None):
        if columns is None:
            columns = ['breast_birads', 'laterality', 'view', 'findings']
        statsRenderer = StatsRenderer(self._df)
        statsRenderer.plot_multiple_frequency_distributions(columns, ancho=20, alto=10, order=None, title=None,
                                                            subtitle=None)
