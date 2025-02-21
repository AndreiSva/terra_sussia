import sys
import random
import numpy as np
from math import pi, cos, sin, floor, sqrt
from PIL import Image

class MapTerrain:
    SEA_LEVEL = 125
    def __init__(self, size):
        self.size = size
        self.terrain = np.zeros((size, size))
        self.biomes = np.zeros((size, size))
        self.colors = {
            "Beach": (225, 227, 130),
            "Mountain": (),
            "Snow": (),
            "Taiga": (),
            "Grassland": (),
        }
        
    def set(self, x, y, value):
        self.terrain[y][x] = value
        
    def get(self, x, y):
        return self.terrain[y][x]
    
    def visualize(self, sea_level = 1):
        # TODO: make this actually good
        normalized = self.terrain.astype(np.uint8)

        image = Image.new("RGB", (self.size, self.size))
        for y in range(self.size):
            for x in range(self.size):
                color = (9, 138, 0)
                if normalized[y][x] > 245:
                    color = (205, 205, 205)
                elif normalized[y][x] > 230:
                    color = (112, 112, 112)
                elif normalized[y][x] < self.SEA_LEVEL:
                    color = (91, 139, 252)
                elif normalized[y][x] < self.SEA_LEVEL + 5:
                    color = (225, 227, 130)
                else:
                    color = (color[0], 255 - normalized[y][x] // 2, color[2])
                image.putpixel((x, y), color)
        
        # image = Image.fromarray(normalized)
        image.show()
        return image
    def set_terrain(self, terrain):
        self.terrain = terrain

class TerrainGenerator:
    NUM_PERMUTATIONS = 256
    def __init__(self, seed = None):
        # this is for pseudo-randomness
        # we use this because seeding random() each time is costly
        if seed is None:
            seed = random.randint(0, sys.maxsize)
        random.seed(seed)
        self._permutations = list(range(self.NUM_PERMUTATIONS))
        random.shuffle(self._permutations)

        # here we have unit vectors for 8 directions
        self._directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1/sqrt(2), 1/sqrt(2)), (-1/sqrt(2), 1/sqrt(2)),
            (1/sqrt(2), -1/sqrt(2)), (-1/sqrt(2), -1/sqrt(2))
        ]

    def _random_grad(self, x, y):
        xw = x % self.NUM_PERMUTATIONS
        yw = y % self.NUM_PERMUTATIONS
        i = self._permutations[(xw + self._permutations[yw]) % self.NUM_PERMUTATIONS]
        return self._directions[i % len(self._directions)]

    def _smooth(self, x):
        return x**3 * (x * (x * 6 - 15) + 10)

    def _lerp(self, v0, v1, t):
        return v0 + t * (v1 - v0)

    def _dot(self, x, y):
        return x[0] * y[0] + x[1] * y[1]

    def _perlin(self, x: float, y: float) -> float:
        xfloor = floor(x)
        yfloor = floor(y)
        # corners of our cell
        cell_corners = (
            (xfloor,     yfloor),     # bottom-left
            (xfloor + 1, yfloor),     # bottom-right
            (xfloor,     yfloor + 1), # top-left
            (xfloor + 1, yfloor + 1)  # top-right
        )

        # this is the relative position of (x, y) to the corner of the cell
        relative_pos = (x - xfloor, y - yfloor)

        # Here we make random unit vectors for each corner of the cell and also calculate distances
        gradients = []
        distances = []
        for corner in cell_corners:
            gradients.append(self._random_grad(corner[0], corner[1]))
            distances.append((x - corner[0], y - corner[1]))

        # doooot producctt between gradient and corresponding distance from point
        dot_products = []
        for i, gradient in enumerate(gradients):
            dot_products.append(self._dot(gradient, distances[i]))

        # these are going to be our interpolation factors (how quickly it is interpolate)
        # we use big brain polynomial from big man Perlin
        horiz_factor = self._smooth(relative_pos[0])
        vert_factor = self._smooth(relative_pos[1])

        # something something bi linear interpolation
        # (we interpolate on both x and y axises)
        value = self._lerp(self._lerp(dot_products[0], dot_products[1], horiz_factor),
                           self._lerp(dot_products[2], dot_products[3], horiz_factor), vert_factor)
        
        return value
    
    def generate(self, size, octaves = 5, persistence = 0.5, freq = 0.005) -> MapTerrain:
        new_map = MapTerrain(size)
        terrain = new_map.terrain
        amp = 1
        for octave in range(octaves):
            # some kewl numpy magic
            grid_x, grid_y = np.meshgrid(np.arange(size) * freq, np.arange(size) * freq)
            noise = np.vectorize(self._perlin)(grid_x, grid_y)
            terrain += noise * amp

            amp *= persistence
            freq *= 2

        # normalize to [0, 255]
        terrain = (terrain - np.min(terrain)) / (np.max(terrain) - np.min(terrain)) * 255
        
        new_map.terrain = terrain
        return new_map
