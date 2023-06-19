import os
from itertools import combinations_with_replacement as combWR


def write(txt: str, opr: str, img_name: str):
    with open("./SimpleEQ.csv", "a+") as file:
        file.write(f"'{txt}';{opr};{img_name}")
        file.write("\n")


def simpleEQ():
    count = 0
    char = ["a", "b", "c", "x", "y", "z"]
    num = [n for n in range(10)]
    opr = [("-", "sub"), ("+", "add"), ("*", "mult"), ("/", "div")]

    combs = combWR(char + num, 2)

    for c in combs:
        for op in opr:
            if op[0] != "/":
                write(f"{c[0]}{op[0]}{c[1]}", op[1], f"simpEQ_{c[0]}{op[1]}{c[1]}")
                count += 1
            else:
                write(
                    r"\frac{%s}{%s}" % (c[0], c[1]),
                    op[1],
                    f"simpEQ_{c[0]}{op[1]}{c[1]}",
                )
                count += 1

    print(f"{count} SimpleEQ")


def main():
    os.system("cls")
    simpleEQ()


if __name__ == "__main__":
    main()
