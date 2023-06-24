from sys import argv
import sys
from itertools import combinations
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import time

pd.set_option("display.max_rows", None)

le = LabelEncoder()
ohe = OneHotEncoder(sparse_output=False)

os.system("cls" if os.name == "nt" else "clear")

if len(argv) > 1 and argv[1] == "-p":
    comb_oprs = [[op for op in argv[3].split("-")]]
    qnts = [int(q) for q in argv[2].split("-")]

else:
    """'comb_oprs' contém as listas de combinações entre operações"""
    opr = ["sub", "add", "mult", "div"]
    comb_oprs = []
    for comb in combinations(opr, 2):
        comb_oprs.append(list(comb))
    for comb in combinations(opr, 3):
        comb_oprs.append(list(comb))
    comb_oprs.append(list(opr))

    # qnts = [100, 500, 1000, 5000, 10000, 50000, 100000]
    qnts = [5000, 10000, 50000, 100000]


def write(regs: list):
    file_path = "./CNN_Resultados.csv"
    if not os.path.exists(file_path):
        with open(file_path, "a+") as f:  # Cria o cabeçalho do CSV
            f.write("qnt_imagens;")
            f.write("qnt_opr;")
            f.write("regs_add;")
            f.write("regs_sub;")
            f.write("regs_mult;")
            f.write("regs_div;")
            f.write("regs_train;")
            f.write("train_add;")
            f.write("train_sub;")
            f.write("train_mult;")
            f.write("train_div;")
            f.write("regs_test;")
            f.write("test_add;")
            f.write("test_sub;")
            f.write("test_mult;")
            f.write("test_div;")
            f.write("tempo_proc;")
            f.write("test_loss;")
            f.write("test_accuracy;")
            f.write("pred_add;")
            f.write("pred_sub;")
            f.write("pred_mult;")
            f.write("pred_div")
            f.write("\n")
    with open(file_path, "a+") as f:
        for r in regs:
            f.write(f"{r};")
        f.write("\n")


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    resized = cv2.resize(threshold, (28, 28))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (28, 28, 1))
    return reshaped


"""Leitura do diretório de imagens"""
img_path = os.listdir("./img_out/")

name_list = []
for img in img_path:
    name_list.append(img.split(".")[0])

# Leitura do CSV Base (Todas as equações geradas)
data = pd.read_csv(
    r"./SimpleEQ.csv", quotechar="'", sep=";", names=["latex", "opr", "img_name"]
)

# Dataframe somente com as equações que possuem imagens criadas no diretório
df_filtered = data.loc[data["img_name"].isin(name_list)].reset_index(drop=True)

print(f"Quantidade Total de imagens: {len(df_filtered)}")
print(
    f"Distribuição de Operações:\n{df_filtered['opr'].value_counts(normalize=True).sort_index().to_string()}"
)

"""Variáveis"""
for q in qnts:
    qnt_imgs = q
    for comb in comb_oprs:
        oprs = comb

        init_time = time.time()
        print("\nParâmetros: ")
        print(f"{qnt_imgs} imagens -> {oprs}")
        small_df = df_filtered.loc[df_filtered["opr"].isin(oprs)][0:qnt_imgs]
        print("Distribuição:")
        print(small_df["opr"].value_counts(normalize=True).sort_index().to_string())

        # Model / data parameters
        num_classes = len(small_df["opr"].unique())
        input_shape = (28, 28, 1)

        image_data = []

        count = 0
        print("Lendo Imagens")
        for img in small_df["img_name"]:
            # Carregar a imagem de entrada
            image = cv2.imread(f"./img_out/{img}.png")

            # Pré-processar a imagem
            preprocessed_image = preprocess_image(image)

            image_data.append(preprocessed_image)
            count += 1
            print(f"\r>> {count}/{qnt_imgs}", end="")
            sys.stdout.flush()

        print()
        x = np.array(image_data)
        y_le = le.fit_transform(small_df["opr"])
        y = ohe.fit_transform(small_df["opr"].to_numpy().reshape(-1, 1))
        idx = np.arange(len(y))

        x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
            x, y, idx, test_size=0.3, random_state=42, stratify=y
        )

        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        # Make sure images have shape (28, 28, 1)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        print("x_train shape:", x_train.shape)
        print(x_train.shape[0], "train samples")
        print(x_test.shape[0], "test samples")

        model = keras.Sequential(
            [
                keras.Input(shape=input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

        model.summary()

        batch_size = 128
        epochs = 15

        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        model.fit(
            x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1
        )

        score = model.evaluate(x_test, y_test, verbose=0)

        y_pred = model.predict(x_test, verbose=0)

        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

        inv_le = le.inverse_transform(y_le)
        y = ohe.inverse_transform(y).ravel()

        y_pred = ohe.inverse_transform(y_pred).ravel().reshape(-1, 1)
        y_test = ohe.inverse_transform(y_test).ravel().reshape(-1, 1)
        idx_test = idx_test.reshape(-1, 1)

        df_predito = pd.DataFrame(
            np.concatenate([idx_test, y_pred], axis=1), columns=["idx", "pred"]
        ).set_index("idx")

        df_pred = small_df[small_df.index.isin(df_predito.index)]
        df_pred = pd.concat([df_pred, df_predito], axis=1)

        # Tempo de processamento
        fin_time = time.time()
        tempo_proc = fin_time - init_time

        # Informações Gerais
        qnt_imagens = len(small_df)
        qnt_opr = len(small_df["opr"].unique())
        regs_add = small_df.loc[small_df["opr"] == "add"].shape[0]
        regs_sub = small_df.loc[small_df["opr"] == "sub"].shape[0]
        regs_mult = small_df.loc[small_df["opr"] == "mult"].shape[0]
        regs_div = small_df.loc[small_df["opr"] == "div"].shape[0]

        # Informações sobre o treino
        regs_train = len(y_train)
        train_add = small_df[
            (small_df.index.isin(idx_train)) & (small_df["opr"] == "add")
        ].shape[0]
        train_sub = small_df[
            (small_df.index.isin(idx_train)) & (small_df["opr"] == "sub")
        ].shape[0]
        train_mult = small_df[
            (small_df.index.isin(idx_train)) & (small_df["opr"] == "mult")
        ].shape[0]
        train_div = small_df[
            (small_df.index.isin(idx_train)) & (small_df["opr"] == "div")
        ].shape[0]

        # Informações sobre o teste
        regs_test = len(y_test)
        test_add = df_pred[(df_pred["opr"] == "add")].shape[0]
        test_sub = df_pred[(df_pred["opr"] == "sub")].shape[0]
        test_mult = df_pred[(df_pred["opr"] == "mult")].shape[0]
        test_div = df_pred[(df_pred["opr"] == "div")].shape[0]

        # Informações sobre predição
        test_loss = score[0]
        test_accuracy = score[1]
        pred_add = df_pred.loc[df_pred["pred"] == "add"].shape[0]
        pred_sub = df_pred.loc[df_pred["pred"] == "sub"].shape[0]
        pred_mult = df_pred.loc[df_pred["pred"] == "mult"].shape[0]
        pred_div = df_pred.loc[df_pred["pred"] == "div"].shape[0]

        regs = [
            qnt_imagens,
            qnt_opr,
            regs_add,
            regs_sub,
            regs_mult,
            regs_div,
            regs_train,
            train_add,
            train_sub,
            train_mult,
            train_div,
            regs_test,
            test_add,
            test_sub,
            test_mult,
            test_div,
            tempo_proc,
            test_loss,
            test_accuracy,
            pred_add,
            pred_sub,
            pred_mult,
            pred_div,
        ]

        write(regs)

        # Cria a pasta de saída dos dataframes de predição
        if not os.path.exists("./dfs_out/"):
            os.mkdir("./dfs_out/")
            print("Directory './dfs_out/' created")

        file_name = f"Pred_{qnt_imgs}R_"
        for op in oprs:
            file_name += op + "_"

        df_pred.to_csv(f"./dfs_out/{file_name}.csv")
        print("Registros gravados em arquivo")
