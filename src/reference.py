def square(side: int):
    for x in range(side):
        if (x == 0) or (x == side - 1):
            print("*" * side)
        else:
            print("*" + ("." * (side - 2)) + "*")


def triangle(side: int):
    for x in range(1, side + 1):
        n_stars = x
        n_dots = side - n_stars
        stars = n_stars * "*"
        dots = n_dots * "."
        print(dots + stars)
