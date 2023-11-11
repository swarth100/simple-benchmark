def square(side: int):
    text = ""
    for i in range(side):
        row = ""
        for j in range(side):
            if (i == 0) or (i == side - 1):
                row += "*"
            elif (j == 0) or (j == side - 1):
                row += "*"
            else:
                row += "."

        text += row
        text += "\n"

    print(text)


def triangle(side: int):
    text = ""
    for i in range(side):
        row = ""
        for j in range(side):
            if (i + j + 1) >= side:
                row += "*"
            else:
                row += "."

        text += row
        text += "\n"

    print(text)
