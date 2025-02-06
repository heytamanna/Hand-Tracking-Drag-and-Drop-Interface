def is_within_box(box, x, y):
    return (box["x"] < x < box["x"] + box["w"]) and (box["y"] < y < box["y"] + box["h"])
