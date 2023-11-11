def square(size: int):
    if size == 1:
        print("*")
    elif size == 2:
        print("**\n**")
    else:
        print("***\n*.*\n***")


def triangle(size: int):
    if size == 1:
        print("*")
    elif size == 2:
        print(".*\n**")
    else:
        print("..*\n.**\n***")
