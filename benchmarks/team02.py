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


def rhombus(size: int):
    limit = (size + 1) // 2

    for step in [1, -1]:
        if step == 1:
            start = 0
            end = limit
        else:
            start = limit - 1 - (size % 2)
            end = -1

        for x in range(start, end, step):
            stars = 1 + (x * 2)
            dots = (size - stars) // 2
            row = "." * dots + "*" * stars + "." * dots
            print(row)
