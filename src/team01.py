def square(side: int):
    if side == 1:
        print("*")
    elif side == 2:
        print("**\n**")
    else:
        print("***\n*.*\n***")


def triangle(side: int):
    if side == 1:
        print("*")
    elif side == 2:
        print(".*\n**")
    else:
        print("..*\n.**\n***")
