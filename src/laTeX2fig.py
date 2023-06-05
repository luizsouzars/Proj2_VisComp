from sys import argv
import os
import pandas as pd
import matplotlib.pylab as plt

# Using LaTex for the rendering of the mathematical expression
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "monospace",
        "font.monospace": "Computer Modern Typewriter",
    }
)


def Tex2fig(serie: pd.Series):
    latex = serie[0]
    img_name = serie[2]

    plt.figure(dpi=25)
    fig, ax = plt.subplots()

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
    plt.xlim(0, 2)
    plt.ylim(0, 2)

    # Salvar Imagem
    plt.grid(False)
    plt.axis("off")
    ax.set_axis_off()
    plt.savefig(f"./img_out/{img_name}.png", bbox_inches="tight")


def main():
    # Arquivo a ser lido
    fstring = argv[1]

    # Cria a pasta de saída das imagens
    if not os.path.exists("./img_out/"):
        os.mkdir("./img_out/")
        print("Directory './img_out/' created")

    print("Directory './img_out/' exists")

    # Esperado um csv com as colunas "latex", "operation", "img_name"
    df = pd.read_csv(fstring, sep=";", quotechar="'")
    df.apply(Tex2fig, axis=1)


if __name__ == "__main__":
    main()
