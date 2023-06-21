from sys import argv
import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pylab as plt
from matplotlib import figure
import multiprocessing as mp

os.system("cls" if os.name == "nt" else "clear")

# Using LaTex for the rendering of the mathematical expression
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "monospace",
        "font.monospace": "Computer Modern Typewriter",
    }
)


def split_dataframe(df, chunk_size=1000):
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i * chunk_size : (i + 1) * chunk_size])
    return chunks


def Tex2fig(serie: pd.Series):
    latex = serie[0]
    img_name = serie[2]

    fig = figure.Figure(dpi=15)
    ax = fig.add_subplot(111)
    # plt.figure(dpi=15)
    # fig, ax = plt.subplots()

    left, width = 0.25, 0.5
    bottom, height = 0.25, 0.5
    right = left + width
    top = bottom + height

    # Alinhamento no meio da imagem
    ax.text(
        0.5 * (left + right),
        0.5 * (bottom + top),
        rf"${latex}$",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
        fontsize=150,
    )

    # Ajusta o tamanho dos eixos
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)

    # # Salvar Imagem
    ax.grid(False)
    ax.axis("off")
    ax.set_axis_off()
    fig.savefig(f"./img_out/{img_name}.png", bbox_inches="tight")
    fig.clear()
    fig.clf()

    plt.cla()
    plt.clf()
    plt.close("all")
    # Ajusta o tamanho dos eixos
    # plt.xlim(0, 2)
    # plt.ylim(0, 2)

    # # Salvar Imagem
    # plt.grid(False)
    # plt.axis("off")
    # ax.set_axis_off()
    # plt.savefig(f"./img_out/{img_name}.png", bbox_inches="tight")
    # plt.cla()
    # plt.close(fig)
    return


def procpar(df: pd.DataFrame):
    df.apply(Tex2fig, axis=1)
    return


def main():
    # Arquivo a ser lido
    fstring = argv[1]

    # Cria a pasta de saída das imagens
    if not os.path.exists("./img_out/"):
        os.mkdir("./img_out/")
        print("Directory './img_out/' created")
    else:
        print("Directory './img_out/' exists")

    # Esperado um csv com as colunas "latex", "operation", "img_name"
    df = pd.read_csv(
        fstring,
        sep=";",
        quotechar="'",
        names=["latex", "operation", "img_name"],
        header=None,
    )

    # df.apply(Tex2fig, axis=1)

    cpus = 20
    chunks = split_dataframe(df)
    len_chunks = len(chunks)
    print(f"Total rows: {len(df)}")
    print(f"Chunks: {len_chunks}")

    with mp.Pool(cpus) as pool:
        _ = pool.map(procpar, chunks)


if __name__ == "__main__":
    main()
