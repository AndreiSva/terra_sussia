import sys
import random
import numpy as np
from math import pi, cos, sin, floor, sqrt
from PIL import Image
import logging

class MapTerrain:
    SEA_LEVEL = 125
    MOUNTAIN_LEVEL = 230
    MOUNTAIN_SNOW_LEVEL = 245
    def __init__(self, size):
        self.size = size
        self.layers = {}
        self.colours = {
            "beach": (225, 227, 130),
            "mountain": (112, 112, 112),
            "snow": (205, 205, 205),
            "desert": (),
            "grassland": (50, 168, 82),
            "water": (91, 139, 252),
        }

    def _resize_matrix(self, smaller_matrix: np.ndarray) -> np.ndarray:
        """Resize the matrix to the dimensions of the terrain"""

        grid_x, grid_y = np.meshgrid(np.arange(self.size), np.arange(self.size))

        def nearest_neighbour(x: int, y: int):
            """This function assumes smaller_matrix has a range of [0, 255] and is square"""

            size_ratio = smaller_matrix.shape[0] / self.size
            
            # calculate the small coordinates:
            x_adj = min(floor(x * size_ratio), smaller_matrix.shape[1] - 1)
            y_adj = min(floor(y * size_ratio), smaller_matrix.shape[0] - 1)

            return smaller_matrix[y_adj, x_adj]
            
        # we apply nearest neighbor
        result = np.vectorize(nearest_neighbour)(grid_x, grid_y)
        return result

    def add_layer(self, name, matrix: np.ndarray):
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Invalid Matrix Shape")
        if matrix.shape != (self.size, self.size):
            matrix = self._resize_matrix(matrix)
        self.layers[name] = matrix.astype(np.uint8)

    def get_layer(self, name: str) -> np.ndarray:
        return self.layers[name]
        
    def set(self, layer: str, x: int, y: int, value: int):
        self.layers[layer][y, x] = value
        
    def get(self, layer: str, x: int, y: int) -> int:
        if x >= self.size or y >= self.size:
            raise IndexError("Invalid x or y position")
        
        return self.layers[layer][y, x]
    
    def visualize(self):
        # TODO: make this actually good
        terrain = self.get_layer("terrain")

        image = Image.new("RGB", (self.size, self.size))
        for y in range(self.size):
            for x in range(self.size):
                color = self.colours["grassland"]
                if terrain[y, x] > self.MOUNTAIN_SNOW_LEVEL:
                    color = self.colours["snow"]
                elif terrain[y, x] > self.MOUNTAIN_LEVEL:
                    color = self.colours["mountain"]
                elif terrain[y, x] < self.SEA_LEVEL:
                    color = self.colours["water"]
                elif terrain[y, x] < self.SEA_LEVEL + 5:
                    color = self.colours["beach"]
                else:
                    color = (color[0], 255 - terrain[y][x] // 2, color[2])
                    
                image.putpixel((x, y), color)
        
        image.show()
        return image

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

    def _random_grad(self, x: int, y: int, wrap: int):
        x_wrapped = x % wrap
        y_wrapped = y % wrap
        
        xw = x_wrapped % self.NUM_PERMUTATIONS
        yw = y_wrapped % self.NUM_PERMUTATIONS
        i = self._permutations[(xw + self._permutations[yw]) % self.NUM_PERMUTATIONS]
        return self._directions[i % len(self._directions)]

    def _smooth(self, x: float):
        return x**3 * (x * (x * 6 - 15) + 10)

    def _lerp(self, v0: float, v1: float, t: float):
        return v0 + t * (v1 - v0)

    def _dot(self, x: tuple, y: tuple):
        return x[0] * y[0] + x[1] * y[1]

    def _perlin(self, x: float, y: float, wrap: int) -> float:
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
            gradients.append(self._random_grad(corner[0], corner[1], wrap))
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

    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        return ((matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix)) * 255).astype(np.uint8)
    
    def _generate_feature(self, size: int, octaves: int = 5, persistence: float = 0.5, freq: float = 0.005) -> np.ndarray:
        feature = np.zeros((size, size))
        amp = 1
        for octave in range(octaves):
            # some kewl numpy magic
            grid_x, grid_y = np.meshgrid(np.arange(size) * freq, np.arange(size) * freq)

            # we need to recalculate the wrap period for each octave because the frequency
            # changes
            current_wrap = int(np.floor(size * freq))
            
            noise = np.vectorize(self._perlin, excluded = {2})(grid_x, grid_y, current_wrap)
            feature += noise * amp

            amp *= persistence
            freq *= 2

        # normalize to [0, 255]
        feature = self._normalize(feature)
        return feature

    def _generate_terrain(self, size: int):
        return self._generate_feature(size, 5, 0.5, 0.005)

    def _generate_biome(self, size: int):
        pass
    
    def _generate_ore(self, size: int):
        return self._generate_feature(size, 5, 0.9, 0.01)

    def _display_grayscale(self, matrix: np.ndarray):
        """Useful for debugging or tweaking generation params"""
        matrix = self._normalize(matrix)
        image = Image.fromarray(matrix)
        image.show()
        return image

    def generate(self, size: int):
        logging.debug("Generating New Map...")        
        new_map = MapTerrain(size)

        logging.debug("Generating Terrain...")
        terrain = self._generate_terrain(size)
        logging.debug("Generating Biome...")
        biome = self._generate_biome(size)
        logging.debug("Generating Ore...")        
        ore = self._generate_ore(size // 5)

        new_map.add_layer("terrain", terrain)
        new_map.add_layer("ore", ore)
        
        return new_map
        
