def square(size: int):
    text = ""
    for i in range(size):
        row = ""
        for j in range(size):
            if (i == 0) or (i == size - 1):
                row += "*"
            elif (j == 0) or (j == size - 1):
                row += "*"
            else:
                row += "."

        text += row
        text += "\n"

    print(text)


def triangle(size: int):
    text = ""
    for i in range(size):
        row = ""
        for j in range(size):
            if (i + j + 1) >= size:
                row += "*"
            else:
                row += "."

        text += row
        text += "\n"

    print(text)
