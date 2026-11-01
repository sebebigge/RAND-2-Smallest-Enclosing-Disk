def MiniDisc(points: list[tuple[float, float]], rng: np.random.Generator):
    n = len(points); counter = 0
    shuffled = rng.permutation(np.asarray(points, dtype=float)).tolist()
    Di = Circle2Points(shuffled[0],shuffled[1])
    for i in range(2,n):
        pi = shuffled[i]
        if distance(pi, Di[0]) > Di[1] * (1 + 1e-12):
            (Di,counter) = MiniDiscWithPoint(shuffled[:i],pi,counter)
        counter += 1
    return (Di,counter)