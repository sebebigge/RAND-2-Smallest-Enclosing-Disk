"""
@author: Sebastian Bigge
@date: 2026-10-31
"""

import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


####################################################################################
################################  IMPLEMENTATION  ##################################
####################################################################################

"""
Calculates the smallest enclosing disc for a collection of points

@param {list[tuple[float, float]]} points:              a set of n points in the plane
@param {np.random.Generator} rng:                       the random number generator used for shuffling
@return {tuple[tuple[tuple[float,float],float], int, int]}:  a tuple of the counter1, counter2 and of the 
    smallest enclosing disc for the given points with q1 and q2 on its boundary returned as a tuple of the 
    center point and the radius squared
"""
def MiniDisc(points: list[tuple[float, float]], rng: np.random.Generator
             ) -> tuple[tuple[tuple[float,float],float], int, int]:
    n = len(points)
    assert n > 1
    # counter1 keeps track of number of circle recomputations
    counter1 = 0
    # counter2 keeps track of iterations as measure of time complexity
    counter2 = 0
    # convert to numpy array, shuffle then convert back to Python list
    shuffled = rng.permutation(np.asarray(points, dtype=float)).tolist()
    # D2 is smallest enclosing disc for p1 and p2
    p1 = shuffled[0]
    p2 = shuffled[1]
    D2 = Circle2Points(p1, p2)
    Di = D2
    #counter1 += 1   -> only in for loop?
    for i in range(2,n):
        pi = shuffled[i]
        # compares distance squared to radius squared ( * (1 + 1e-12) suggest by AI)
        if distance(pi, Di[0]) < Di[1] * (1 + 1e-12):
            pass        # pi within Di -> nothing changes
        else:
            (Di,counter1, counter2) = MiniDiscWithPoint(shuffled[:i], pi, counter1, counter2)
            counter1 += 1
        counter2 += 1
        #DrawSmallestEnclosingDisk(Di, shuffled[:i+1])
    return (Di,counter1, counter2)


"""
Calculates the smallest enclosing disc for a collection of points and a given point of the boundary

The points are already in uniformly random order (a prefix of the shuffled list in MiniDisc), so no
further shuffling and therefore no random number generator is needed here.

@param {list[tuple[float, float]]} points:  a set of n points in the plane
@param {tuple[float, float]} q:             a point belonging to the boundary of the smallest enclosing disc
@param {int} counter1:                      counts the number of circle recomputations
@param {int} counter2:                      counts the number of iterations as measure of time complexity
@return {tuple[tuple[tuple[float,float],float], int, int]}:  a tuple of the counter1, counter2 and of the 
    smallest enclosing disc for the given points with q1 and q2 on its boundary returned as a tuple of the 
    center point and the radius squared
"""
def MiniDiscWithPoint(points: list[tuple[float, float]], q: tuple[float, float], counter1: int, counter2: int
                      ) -> tuple[tuple[tuple[float,float],float], int, int]:
    n = len(points)
    assert n > 0
    # D1 is smallest enclosing disc for p1 and q
    p1 = points[0]
    D1 = Circle2Points(p1, q)
    Dj = D1
    #counter1 += 1   -> only in for loop?
    for j in range(1,n):
        pj = points[j]
        # compares distance squared to radius squared ( * (1 + 1e-12) suggest by AI)
        if distance(pj, Dj[0]) < Dj[1]  * (1 + 1e-12):
            pass        # pj within Dj -> nothing changes
        else:
            (Dj,counter1, counter2) = MiniDiscWith2Points(points[:j], pj, q, counter1, counter2)
            counter1 += 1
        counter2 += 1
    return (Dj,counter1, counter2)


"""
Calculates the smallest enclosing disc for a collection of points and two given points of the boundary

@param {list[tuple[float, float]]} points:  a set of n points in the plane
@param {tuple[float, float]} q1:            a point belonging to the boundary of the smallest enclosing disc
@param {tuple[float, float]} q2:            a point belonging to the boundary of the smallest enclosing disc
@param {int} counter1:                      counts the number of circle recomputations
@param {int} counter2:                      counts the number of iterations as measure of time complexity
@return {tuple[tuple[tuple[float,float],float], int, int]}:  a tuple of the counter1, counter2 and of the 
    smallest enclosing disc for the given points with q1 and q2 on its boundary returned as a tuple of the 
    center point and the radius squared
"""
def MiniDiscWith2Points(points: list[tuple[float, float]], q1: tuple[float, float], q2: tuple[float, float],
                        counter1: int, counter2: int) -> tuple[tuple[tuple[float,float],float], int, int]:
    n = len(points)
    # D0 is smallest enclosing disc for q1 and q2
    D0 = Circle2Points(q1, q2)
    Dk = D0
    #counter1 += 1   -> only in for loop?
    for k in range(0,n):
        # no need to reshuffle
        pk = points[k]
        # compares distance squared to radius squared ( * (1 + 1e-12) suggest by AI)
        if distance(pk, Dk[0]) < Dk[1] * (1 + 1e-12):
            pass        # pk within Dk -> nothing changes
        else:
            Dk = Circle3Points(q1, q2, pk)
            counter1 += 1
        counter2 += 1
    return (Dk, counter1, counter2)


"""
Calculates the circle defined by the two given points where the two points lie opposite each 
other on the edge of the circle

@param {tuple[float,float]} point1:     point defined by x- and y-coordinate
@param {tuple[float,float]} point2:     point defined by x- and y-coordinate
@return {tuple[tuple[float,float],float]}:    circle represented by a tuple of the center coordinates
as well as the radius squared
"""
def Circle2Points(point1: tuple[float,float], point2: tuple[float,float]) -> tuple[tuple[float,float],float]:
    a, b = point1[0], point1[1]
    c, d = point2[0], point2[1]
    # equation of circle is (x−a)(x−c)+(y−b)(y−d)=0 which transformed to (x-xc)^2+(y-yc)^2=r^2 
    # results in:
    xc = (a+c)/2
    yc = (b+d)/2
    r_squared = ((a-c)**2 + (b-d)**2)/4
    return ((xc,yc),r_squared)


"""
Calculates the circle defined by the three given points
see: https://scipython.com/blog/the-equation-of-the-circle-through-three-points/

@param {tuple[float,float]} point1:     point defined by x- and y-coordinate
@param {tuple[float,float]} point2:     point defined by x- and y-coordinate
@param {tuple[float,float]} point3:     point defined by x- and y-coordinate
@return {tuple[tuple[float,float],float]}:    circle represented by a tuple of the center coordinates
as well as the radius squared
"""
def Circle3Points(point1: tuple[float,float], point2: tuple[float,float], point3: tuple[float,float]
            ) -> tuple[tuple[float,float],float]:
    a, b = point1[0], point1[1]
    c, d = point2[0], point2[1]
    x0, y0 = point3[0], point3[1]
    # check for collinearity
    if ((b-d)*x0 - (a-c)*y0 + a*d - b*c) == 0:
        p1 = min(point1, point2, point3)
        p2 = max(point1, point2, point3)
        return Circle2Points(p1, p2)
    alpha = ((x0-a)*(x0-c) + (y0-b)*(y0-d)) / ((b-d)*x0 - (a-c)*y0 + a*d - b*c)
    xc = ((a+c) + alpha*(b-d)) / 2
    yc = ((b+d) - alpha*(a-c)) / 2
    r_squared = alpha * (a*d - b*c) - a*c - b*d + xc**2 + yc**2
    return ((xc,yc),r_squared)


"""
Calculates the Euclidian distance squared between the two given points

@param {tuple[float,float]} point1:     point defined by x- and y-coordinate
@param {tuple[float,float]} point2:     point defined by x- and y-coordinate
@return {float}:                        the Euclidian distance squared
"""
def distance(point1: tuple[float,float], point2: tuple[float,float]) -> float:
    a, b = point1[0], point1[1]
    c, d = point2[0], point2[1]
    return (c-a)**2 + (d-b)**2


####################################################################################
################################  DISTRIBUTION  ####################################
####################################################################################

"""
Multiple distributions to sample from:
- Uniform square
- Uniform disc
- Gaussian
- Points on the edge of a circle
The distribution shouldn't influence the complexity, because it's shuffled

All sampling functions take the same random number generator, so that a single recorded seed
reproduces the complete experiment.
"""

# GenAI
"""
Samples n points uniformly from the unit square [0, 1) x [0, 1)

@param {int} n:                         number of required points
@param {np.random.Generator} rng:       the random number generator to draw from
@return {list[tuple[float, float]]}:    a list of n points (x- and y-coordinates)
"""
def UniformSquare(n: int, rng: np.random.Generator) -> list[tuple[float, float]]:
    return rng.random((n, 2)).tolist()


# GenAI
"""
Samples n points uniformly from the unit the unit circle

@param {int} n:                         number of required points
@param {np.random.Generator} rng:       the random number generator to draw from
@return {list[tuple[float, float]]}:    a list of n points (x- and y-coordinates)
"""
def UniformDisc(n: int, rng: np.random.Generator) -> list[tuple[float, float]]:
    r = np.sqrt(rng.random(n))
    theta = rng.random(n) * 2 * np.pi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return list(zip(x, y))


# GenAI
"""
Samples n points uniformly from a Gaussian distribution with mean 0 and standard deviation 1

@param {int} n:                         number of required points
@param {np.random.Generator} rng:       the random number generator to draw from
@return {list[tuple[float, float]]}:    a list of n points (x- and y-coordinates)
"""
def Gaussian(n: int, rng: np.random.Generator) -> list[tuple[float, float]]:
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    return list(zip(x, y))


# GenAI
"""
Samples n points (located on the edge of the circle - not inside) uniformly from the unit circle 
with radius 1

@param {int} n:                         number of required points
@param {np.random.Generator} rng:       the random number generator to draw from
@return {list[tuple[float, float]]}:    a list of n points (x- and y-coordinates)
"""
def EdgeCircle(n: int, rng: np.random.Generator) -> list[tuple[float, float]]:
    theta = rng.random(n) * 2 * np.pi
    x = np.cos(theta)
    y = np.sin(theta)
    return list(zip(x, y))


####################################################################################
####################################  PLOTS  #######################################
####################################################################################

# GenAI
"""
Draws a disc and its points on a Matplotlib axes.

@param {tuple[tuple[float, float], float]} disc:    disc represented by its center and radius squared
@param {list[tuple[float, float]]} points:          points to draw
@return {None}:                                     draws the smallest enclosing disc and points
"""
def DrawSmallestEnclosingDisk(disc: tuple[tuple[float, float], float], points: list[tuple[float, float]]
                              ) -> None:
    ax = plt.gca()
    center, radius_squared = disc
    radius = math.sqrt(radius_squared)
    ax.add_patch(Circle(center, radius, fill=False, color="red"))
    x_coordinates, y_coordinates = zip(*points)
    ax.scatter(x_coordinates, y_coordinates, color="blue", zorder=2)
    ax.set_aspect("equal")
    # for fixed axes limits, uncomment the following lines:
    """ ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)"""
    plt.show()


# GenAI
"""
Plots the average number of iterations against the input sizes, with the standard error of the mean as
error bars and optionally the fitted line c * n.

@param {np.ndarray} x_points:    the input sizes
@param {list[float]} average:   average iteration count for each input size
@param {float | None} log_c:    offset log(c) of the fitted line c * n; if None, no fit is drawn
@return {None}:                 draws the average number of iterations against the input sizes
"""
def PlotAverage(x_points: np.ndarray, average: list[float], log_c: float | None = None,
                save_path: Path | None = None) -> None:
    plt.figure()
    plt.plot(x_points, average, 'ro', markersize=5, label='data')
    plt.plot(x_points, x_points, label='n')
    if log_c is not None:
        plt.plot(x_points, np.exp(log_c) * x_points, 'g--', label=f'c·n, c = {np.exp(log_c):.2f}')
    plt.grid()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of points (n)')
    plt.ylabel('Average cost (iterations)')
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()



# GenAI
"""
Plots the relative deviation of the average number of iterations from the fitted line.

@param {np.ndarray} x_points:    the input sizes
@param {list[float]} average:   average iteration count for each input size
@param {list[float]} sem:       standard error of the mean (standard deviation of the runs / sqrt(runs))
@param {float} log_c:           offset log(c) of the fitted line c * n
@return {None}:                 draws the relative deviation of the average number of iterations from 
                                the fitted line
"""
def PlotRelativeDeviation(x_points, average, sem, log_c, save_path: Path | None = None):
    n = np.floor(np.asarray(x_points, dtype=float))     # TestMain samples int(n) points
    model = np.exp(log_c) * n
    plt.figure()
    plt.errorbar(n, np.asarray(average) / model, yerr=np.asarray(sem) / model,
                 fmt='bo', markersize=5, capsize=3)
    plt.axhline(1, color='g', linestyle='--', label='fit: c·n')
    plt.xscale('log')
    plt.grid(which='both')
    plt.xlabel('Number of points (n)')
    plt.ylabel('Average cost / (c·n)')
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# GenAI
"""
Fits a straight line to the log-log data (log average = slope * log n + log c) to find the offset log(c).
Two fits are made on the input sizes n >= n_min (smaller sizes are still in the start-up phase, where 
lower-order terms bias the fit):
- a free fit, where the slope should come out close to 1 if the cost is linear
- a fit with the slope fixed to 1 (model: average = c * n), which gives the cleaner estimate of c
Natural logarithms are used, so c = exp(log c).

@param {np.ndarray} x_points:    the input sizes
@param {list[float]} average:   average iteration count for each input size
@param {int} n_min:             smallest input size that is included in the fit
@return {tuple[float, float, float]}:   the slope and offset log(c) of the free fit, and the offset log(c)
    of the fit with slope 1
"""
def FitOffset(x_points: np.ndarray, average: list[float], n_min: int = 1000) -> tuple[float, float, float]:
    # TestMain samples int(x_points[i]) points, so the fit has to use the truncated sizes as well
    n = np.floor(np.asarray(x_points, dtype=float))
    y = np.asarray(average, dtype=float)
    mask = n >= n_min
    if mask.sum() < 2:
        raise ValueError("Fewer than two input sizes with n >= n_min")
    log_n = np.log(n[mask])
    log_y = np.log(y[mask])
    slope, log_c = np.polyfit(log_n, log_y, 1)
    log_c_fixed = float(np.mean(log_y - log_n))
    print(f"free fit (n >= {n_min}):  slope = {slope:.4f},  log(c) = {log_c:.4f},  c = {np.exp(log_c):.4f}")
    print(f"slope fixed to 1:        log(c) = {log_c_fixed:.4f},  c = {np.exp(log_c_fixed):.4f}")
    return (float(slope), float(log_c), log_c_fixed)


####################################################################################
####################################  TESTS  #######################################
####################################################################################

# GenAI: seed, sem
"""
The main test function that measures the average number of iterations (containment tests) for different 
input sizes.

@param {int} n:                 the largest number of points
@param {int} runs:              the number of runs per input size defined by nbN
@param {int} nbN:               the number of different input sizes ranging from log(10) to log(n) equally 
                                spaced on a logarithmic scale
@param {str} distribution:      the distribution to sample from
@param {int | None} seed:       seed of the random number generator; if None, a fresh seed is drawn from 
                                the OS. In both cases the seed is printed, so every run can be reproduced
@return {tuple[np.ndarray, list[float], list[float]]}: the input sizes, the average cost (iterations) and
    the standard error of the mean (sample standard deviation of the runs / sqrt(runs)) for each size
"""
def TestDistribution(n: int, runs: int, nbN: int, distribution: str, seed: int | None = None
             ) -> tuple[np.ndarray, list[float], list[float]]:
    if distribution == "uniform square":
        sample_func = UniformSquare
    elif distribution == "uniform disc":
        sample_func = UniformDisc
    elif distribution == "gaussian":
        sample_func = Gaussian
    elif distribution == "edge circle":
        sample_func = EdgeCircle
    else:
        raise ValueError("Unknown distribution")
    # a single generator for the whole experiment: sampling and shuffling both draw from it
    if seed is None:
        seed = np.random.SeedSequence().entropy     # record the OS entropy so the run can be repeated
    rng = np.random.default_rng(seed)
    if runs < 2:
        raise ValueError("At least 2 runs are needed to estimate the standard deviation")
    average = []
    sem = []
    x_points = np.logspace(np.log10(10), np.log10(n), num=nbN)      # returns float
    for i in range(len(x_points)):
        counts = np.empty(runs)                                     # cost of every single run
        for j in range(runs):
            points = sample_func(int(x_points[i]), rng)             # convert to int
            (disc, counter1, counter2) = MiniDisc(points, rng)
            counts[j] = counter2
            # draw the disc and points
            # DrawSmallestEnclosingDisk(disc, points)
        average.append(counts.mean())
        sem.append(counts.std(ddof=1) / np.sqrt(runs))
    return (x_points, average, sem)


# GenAI: saving to path, seed
"""
Performs the TestDistribution function for all distributions and plots the results

@param {int} n:             the largest number of points
@param {int} runs:          the number of runs per input size defined by nbN
@param {int} nbN:           the number of different input sizes ranging from log(10) to log(n) 
                            equally spaced on a logarithmic scale
@param {int | None} seed:   seed of the random number generator; if None, a fresh seed is drawn from
                            the OS. In both cases the seed is printed, so every run can be reproduced
@return {None}:             performs the tests and plots the results
"""
def TestMain(n: int, runs: int, nbN: int, seed: int | None = None) -> None:
    sample_distributions = ["uniform square", "uniform disc", "gaussian", "edge circle"]
    output_dir = Path("Grafieken/N100000_RUNS1000_NBN25")
    output_dir.mkdir(parents=True, exist_ok=True)
    for distribution in sample_distributions:
        print(f"Testing distribution: {distribution}")
        (x_points, average, sem) = TestDistribution(n, runs, nbN, distribution, seed)
        (slope, log_c, log_c_fixed) = FitOffset(x_points, average, n_min=1000)
        filename = distribution.replace(" ", "_")
        PlotRelativeDeviation(x_points, average, sem, log_c_fixed,
                              output_dir / f"{filename}_relative_deviation.png")
        PlotAverage(x_points, average, log_c_fixed,
                    output_dir / f"{filename}_average.png")


if __name__ == "__main__":
    TestMain(100000, 1000, 25, seed=12345)


"""
Graphs showing relative deviation for n = 1000 are inaccurate
-> file N1000_RUNS10000_NBN100 inaccurate
"""