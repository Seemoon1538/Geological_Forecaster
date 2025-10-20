def add_coords(a, b):
    """Простая функция для примера: складывает два кортежа координат поэлементно."""
    if a is None or b is None:
        raise ValueError("Аргументы не должны быть None")
    if len(a) != len(b):
        raise ValueError("Длины кортежей должны совпадать")
    return tuple(x + y for x, y in zip(a, b))


if __name__ == "__main__":
    print(add_coords((1, 2), (3, 4)))
