import base64
import random
from io import BytesIO
from PIL import Image
import numpy as np
import ipywidgets as widgets
from IPython.display import HTML, display, clear_output

from utils.Dataset import Dataset

import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")
PALETTE = "viridis"


class DatasetRenderer:
    def __init__(self, df: pd.DataFrame, render_sample_patch=False, render_sample_patch_height=100,
                 render_sample_patch_width=100):
        self.df = df
        self.render_sample_patch = render_sample_patch
        self.render_sample_patch_height = render_sample_patch_height
        self.render_sample_patch_width = render_sample_patch_width
        object.__setattr__(self, "hide_columns", ["xmin", "ymin", "xmax", "ymax"])
        self._current_page = 0
        self._page_size = 10
        self._pagination_controls = None
        self._output_widget = None

    def _get_page_data(self):
        i = self._current_page * self._page_size
        return self.df.iloc[i:i + self._page_size]

    def _get_total_pages(self):
        return (len(self.df) + self._page_size - 1) // self._page_size

    def _render_page(self):
        page = self._get_page_data()

        html = [
            '<div style="font-family:Arial;background:#000;color:#fff;padding:10px;">',
            f'<div style="color:#fff;">Página {self._current_page + 1} de {self._get_total_pages()}</div>',
            '<table style="width:100%;border-collapse:collapse;table-layout:fixed;">',
            '<thead><tr>',
            '<th style="padding:6px;border-bottom:1px solid #444;text-align:center;vertical-align:middle;color:#fff;">Atributos</th>',
            '<th style="padding:6px;border-bottom:1px solid #444;text-align:center;vertical-align:middle;color:#fff;">Full Image</th>',
            '<th style="padding:6px;border-bottom:1px solid #444;text-align:center;vertical-align:middle;color:#fff;">Patch</th>',
            '</tr></thead><tbody>'
        ]

        for idx, (_, row) in enumerate(page.iterrows()):
            bg = "#2a2a2a" if idx % 2 else "#1a1a1a"
            html.append(f'<tr style="background:{bg};height:100%;">')

            # Card de atributos (centrada)
            attr_card = ['<div style="border:1px solid #555;border-radius:8px;padding:6px;color:#fff;">']
            for col in self.df.columns:
                if col not in ["patch"]:
                    value = row[col]
                    value = str(value) if pd.notna(value) else ""
                    attr_card.append(f'<div><b>{col}:</b> {value}</div>')
            attr_card.append('</div>')

            centered_card = (
                '<div style="display:inline-block;text-align:left;color:#fff;">'
                f'{"".join(attr_card)}'
                '</div>'
            )
            html.append(
                f'<td style="padding:4px;height:100%;text-align:center;vertical-align:middle;color:#fff;">{centered_card}</td>'
            )

            # Imagen principal
            img_html = self.thumb_html(
                row["path"], self._render_width,
                rois=[row['xmin'], row['ymin'], row['xmax'], row['ymax']]
            )
            html.append(
                f'<td style="padding:4px;text-align:center;vertical-align:middle;color:#fff;">{img_html}</td>'
            )

            # Patch
            if self.render_sample_patch:
                patch_html = self.random_patch_html(
                    row["path"], self.render_sample_patch_width, self.render_sample_patch_height,
                    rois=[row['xmin'], row['ymin'], row['xmax'], row['ymax']]
                )
            else:
                patch_html = ""
            html.append(
                f'<td style="padding:4px;text-align:center;vertical-align:middle;color:#fff;">{patch_html}</td>'
            )

            html.append('</tr>')

        html.extend(['</tbody></table></div>'])
        return "".join(html)

    def _update_display(self, *_):
        with self._output_widget:
            clear_output(wait=True)
            display(HTML(self._render_page()))

    def _on_nav(self, btn):
        total = self._get_total_pages()
        desc = btn.description
        if desc == "Primera":
            self._current_page = 0
        elif desc == "Anterior":
            self._current_page = max(0, self._current_page - 1)
        elif desc == "Siguiente":
            self._current_page = min(total - 1, self._current_page + 1)
        elif desc == "Última":
            self._current_page = total - 1
        self._update_display()

    def render(self, width=400, page_size=10):
        self._render_width = width
        self._page_size = page_size

        nav = []
        for d in ("Primera", "Anterior", "Siguiente", "Última"):
            b = widgets.Button(description=d, button_style='info',
                               layout=widgets.Layout(width='80px'))
            b.on_click(self._on_nav)
            nav.append(b)

        page_in = widgets.IntText(
            value=1, min=1, max=self._get_total_pages(),
            description='Página:', layout=widgets.Layout(width='120px')
        )
        page_in.observe(lambda ch: setattr(self, "_current_page", ch['new'] - 1), names='value')

        size_sel = widgets.Dropdown(
            options=[5, 10, 20, 50], value=page_size,
            description='Por página:', layout=widgets.Layout(width='120px')
        )
        size_sel.observe(lambda ch: setattr(self, "_page_size", ch['new']), names='value')

        self._pagination_controls = widgets.HBox([widgets.HBox(nav), page_in, size_sel])
        self._output_widget = widgets.Output()

        display(self._pagination_controls, self._output_widget)
        self._update_display()

    def thumb_html(self, path, width, rois=None, draw_bounding_box=True):
        try:
            arr = self.df.full_image(path, rois=rois, draw_bounding_box=draw_bounding_box)
            img_pil = Image.fromarray(arr.astype(np.uint8))
            img_pil.thumbnail((width, width))
            buf = BytesIO()
            img_pil.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            return f'<img src="data:image/png;base64,{b64}" width="{width}" />'
        except Exception as e:
            print("thumb_html error:", e)
            return f'<span>Error imagen: {e}</span>'

    def random_patch_html(self, path, patch_w, patch_h, rois=None):
        try:
            patch = self.df.sample_random_patch(path, patch_w, patch_h, rois)
            if patch is None:
                return "<span>Patch inválido</span>"

            img_pil = Image.fromarray(patch.astype(np.uint8))
            buf = BytesIO()
            img_pil.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()

            return f'<img src="data:image/png;base64,{b64}" width="{patch_w}" height="{patch_h}"/>'
        except Exception as e:
            return f'<span>Error patch: {e}</span>'
