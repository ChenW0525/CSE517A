import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def basic_stat(target):
    stats = {
        'mean': target.mean(),
        'std': target.std(),
        'min': target.min(),
        'max': target.max(),
        'range': target.max() - target.min()
    }
    return stats

# def target_kde(target, bw_adj=0.5):
#     plt.figure(figsize=(10, 6))
#     sns.kdeplot(target, bw_adjust=bw_adj)
#     plt.title('Kernel density estimation of the target')
#     plt.xlabel('Target')
#     plt.ylabel('Density')
#     plt.grid(True)
#     plt.show()

def series_kde(series, names):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(series, bw_adjust=1)
    # y_name = series.columns
    plt.title(f'Kernel density estimation of the {names}')
    plt.xlabel(f'{names}')
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()

def multi_kde(df, ftr):
    n_cols = 4
    n_rows = (len(ftr) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    # fig.subplots_adjust(hspace=0.9, wspace=0.3)
    for i, f in enumerate(ftr):
        i_row = i // n_cols
        i_col = i % n_cols
        sns.kdeplot(df[f], ax=axes[i_row, i_col], bw_adjust=1)
        # axes[i_row, i_col].set_title(f'KDE of feature {f}', fontsize=5)
        axes[i_row, i_col].set_xlabel(f'{f}', fontsize=5)
        axes[i_row, i_col].set_ylabel('Density', fontsize=5)
    for j in range(i + 1, n_rows * n_cols):
        axes.flat[j].axis('off')
    plt.tight_layout()
    plt.show()

def xy_corrl(df, ftr_name, tgt):
    cor = df[ftr_name].corrwith(tgt)
    cor_sort = cor.abs().sort_values(ascending=False)
    print(f"sorted xy correlation: \n{cor_sort}")
    cor_3 = cor.abs().sort_values(ascending=False).head(3).index
    print(f"3 correlation: {cor_3.tolist()}")
    for c in cor_3:
        plt.figure(figsize=(8, 6))
        plt.scatter(df[c], tgt, alpha=0.5)
        plt.title(f'Scatter plot between {c} and {tgt.name}')
        plt.xlabel(f'{c}')
        plt.ylabel(f'{tgt.name}')
        plt.show()
    # for f in cor_3:
    #     sns.scatterplot(data=ftr, x=ftr_name, y=tgt.name)
    return cor_3[0]

def xx_corrl(df, ftr_name, xy_ftr):
    xx_corrl_mat = df[ftr_name].corr()
    print(f"correlation matrix for features:\n{xx_corrl_mat}")
    plt.matshow(xx_corrl_mat)
    plt.show()
    corrl_ftrs = xx_corrl_mat[xy_ftr].drop(xy_ftr)
    top_ftrs = corrl_ftrs.abs().sort_values(ascending=False).head(3).index
    top_ftrs = pd.Index(list(top_ftrs) + [xy_ftr])
    sns.pairplot(df[top_ftrs], kind='scatter', diag_kind='kde')
    plt.suptitle(f'Correlation feature plot for{xy_ftr}')
    plt.show()

if __name__ == "__main__":
    # df = pd.read_csv('..\\data\\train.csv')
    print(f"Current working directory: {os.getcwd()}")
    data_dir = os.path.join('.', 'data')
    data_path = os.path.join(data_dir, 'train.csv')
    df = pd.read_csv(data_path)
    target = df.iloc[:, -1]
    # feature = df.iloc[:, :-1]
    features = df.columns[:-1]
    # print(f"target: {target}")

    # #_________Staticstics_for_target______
    # t_stat = basic_stat(target)
    # print(f"statistics for target: ")
    # for key, value in t_stat.items():
    #     print(f"{key}: {value}")
    # print(f'\n')
    
    # #_________Staticstics_for_features______
    std_dict = {}
    for ftr in features:
        print(f"statistics for feature {ftr}: ")
        f_stat = basic_stat(df[ftr])
        for key, value in f_stat.items():
            if key == 'std':
                std_dict[ftr] = value
            print(f"{key}: {value}")
        print(f'\n')
    # std_10 = sorted(std_dict.items(), key=lambda x: x[1])[:10]
    # print(f"least std features: \n{std_10}")

    #_________Kernel_density_estimation_________
    # series_kde(target, df.columns[-1])
    # multi_kde(df, features)

    #_________Correlated_features________
    xy_top_ftr = xy_corrl(df, features, target)
    print(f"xy most correlation feature:{xy_top_ftr}")
    xx_corrl(df, features, xy_top_ftr)