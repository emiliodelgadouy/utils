import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")
PALETTE = "viridis"


class StatsRenderer:
    def __init__(self, df):
        self.df = df
    def plot_multiple_frequency_distributions(self, columns, ancho=20, alto=10, order=None, title=None, subtitle=None):
        max_len = 20
        n = len(columns)
        ncols = 3
        nrows = math.ceil(n / ncols)

        fig, axs = plt.subplots(nrows, ncols, figsize=(ancho, alto))
        axs = axs.flatten()

        for i, col in enumerate(columns):
            # explotar listas si las hay
            series = self.df[col]
            if series.apply(lambda x: isinstance(x, list)).any():
                series = series.explode()

            # conteo
            freqs = series.value_counts()

            # si es numérico y no pasó 'order', ordenar por valor de x
            if order is None and pd.api.types.is_numeric_dtype(freqs.index):
                freqs = freqs.sort_index()
            # si se pasó 'order', respetar ese orden
            elif order is not None:
                freqs = freqs.reindex(order).fillna(0)

            # preparar etiquetas truncadas
            labels = [
                (str(x)[:max_len] + "…") if len(str(x)) > max_len else str(x)
                for x in freqs.index
            ]
            vals = freqs.values

            # seaborn barplot con hue para evitar warning
            sns.barplot(
                x=labels, y=vals,
                hue=labels, palette=PALETTE,
                dodge=False, legend=False,
                ax=axs[i]
            )

            axs[i].set_xlabel(col, fontsize=12, fontweight="bold")
            axs[i].set_ylabel("Frecuencia", fontsize=12)
            axs[i].tick_params(axis="x", rotation=90, labelsize=10)
            axs[i].set_title(
                title or f"Distribución de frecuencia de “{col}” {subtitle}",
                fontsize=14
            )

            # anotaciones dentro de cada barra
            for p in axs[i].patches:
                h = p.get_height()
                if h > 0:
                    axs[i].annotate(
                        f"{int(h)}",
                        (p.get_x() + p.get_width() / 2, h / 2),
                        ha="center", va="center",
                        color="white", fontsize=10
                    )

        # eliminar ejes vacíos
        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        plt.show()

    def plot_multiple_repetition_distributions(
            df,
            columns,
            ancho=20, alto=10,  # ancho y alto en pulgadas
            title=None
    ):
        max_len = 20
        n = len(columns)
        ncols = 3
        nrows = math.ceil(n / ncols)

        fig, axs = plt.subplots(nrows, ncols, figsize=(ancho, alto))
        axs = axs.flatten()

        for i, col in enumerate(columns):
            # agrupar y contar repeticiones
            if isinstance(col, list):
                label = " + ".join(col)
                grouped = df.groupby(col, dropna=False).size()
            else:
                label = col
                grouped = df[col].value_counts(dropna=False)

            repetidos = grouped.value_counts()
            repetidos.index = [
                (str(x)[:max_len] + "…") if len(str(x)) > max_len else str(x)
                for x in repetidos.index
            ]

            sns.barplot(
                x=repetidos.index, y=repetidos.values,
                hue=repetidos.index, palette=PALETTE,
                dodge=False, legend=False,
                ax=axs[i]
            )

            axs[i].set_xlabel("Veces repetido", fontsize=12)
            axs[i].set_ylabel("Cantidad", fontsize=12)
            axs[i].tick_params(axis="x", rotation=90, labelsize=10)
            axs[i].set_title(
                title or f"Repetición en “{label}”",
                fontsize=14
            )

            for p in axs[i].patches:
                h = p.get_height()
                if h > 0:
                    axs[i].annotate(
                        f"{int(h)}",
                        (p.get_x() + p.get_width() / 2, h / 2),
                        ha="center", va="center",
                        color="white", fontsize=9
                    )

        # eliminar ejes vacíos
        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        plt.show()