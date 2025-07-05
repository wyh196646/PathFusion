import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
colors = {
    "CONCH": "#A3DBD2",
    "CHIEF": "#99C0DB",
    "PRISM": "#CAC7E1",
    "Gigapath": "#DCDCDC",
    "FMBC_Slide": "#FDC281",
    "FMBC": "#FB998E",
}

def format_model_label(model):
    if model == "FMBC_Slide":
        return r"$\mathrm{BRFound}_V$"
    elif model == "FMBC":
        return r"$\mathrm{BRFound}$"
    else:
        return model
def plot_common_genes_top_n(file_paths, top_n=10, nrows=1, ncols=2, figsize=(16, 8)):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, subplot_kw=dict(polar=True), figsize=figsize)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for idx, file_path in enumerate(file_paths):
        if idx >= len(axes):
            break

        df = pd.read_csv(file_path)
        title = os.path.basename(file_path).replace('.csv', '')

        models = df['Model'].unique()
        gene_sets = []

        # Step 1: Determine genes available in all models
        for model in models:
            gene_sets.append(set(df[df['Model'] == model]['Gene'].tolist()))
        common_genes = sorted(set.intersection(*gene_sets))

        if not common_genes:
            continue

        # Step 2: Select top-N genes from FMBC (or fallback to first available model)
        ranking_model = 'FMBC' if 'FMBC' in models else models[0]
        fmbc_df = df[(df['Model'] == ranking_model) & (df['Gene'].isin(common_genes))]
        selected_genes = fmbc_df.nlargest(top_n, 'PCC')['Gene'].tolist()

        # Step 3: Prepare angles
        angles = np.linspace(0, 2 * np.pi, len(selected_genes), endpoint=False).tolist()
        angles += angles[:1]

        ax = axes[idx]
        for model in models:
            model_df = df[(df['Model'] == model) & (df['Gene'].isin(selected_genes))]
            model_df = model_df.set_index('Gene').reindex(selected_genes).fillna(0)
            values = model_df['PCC'].tolist() + [model_df['PCC'].tolist()[0]]
            label = format_model_label(model)
            ax.plot(angles, values, label=label, color=colors.get(model, None))
            ax.fill(angles, values, color=colors.get(model, None), alpha=0.1)

        # Axis and aesthetics
        ax.set_title(title, size=14, pad=20)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(selected_genes, size=8)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment('center')
            label.set_verticalalignment('center_baseline')
            label.set_rotation(0)
        ax.set_yticklabels([])
        ax.tick_params(pad=5)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    return fig

pcc_by_gene_path ='/home/yuhaowang/project/FMBC/downstream/finetune/script/plot/original_regresssion_save'

file_paths=os.listdir(pcc_by_gene_path)
file_paths = [os.path.join(pcc_by_gene_path, f) for f in file_paths if f.startswith('pcc')]

fig = plot_common_genes_top_n(file_paths, top_n=15, nrows=1, ncols=6)
plt.show()
