# TensorFlow and tf.keras
import math
from time import sleep, time_ns
import tensorflow as tf
# Helper libraries
import numpy as np
import car
import pygame
import threading

import multiprocessing as mp

rng = np.random.default_rng()

class GeneticAlgorithm:
    def return_none():
        return None

    def __init__(self, create_model=return_none, population=1, fitness=None, sim=None):
        self.population = population
        self.create_model = create_model
        self.models = [create_model() for _ in range(population)]
        self.scores = np.zeros(population) # numpy array
        self.fitness = fitness
        self.sim = sim
    
    def generation(self, count = 1):
        '''
        simulate one generation
        '''
        top_10p = int(np.ceil(self.population / 10.0))
        for gen in range(count):
            # calculate scores
            if self.sim:
                self.scores = self.sim(self.models)
            else:
                for i in range(self.population):
                    self.scores[i] = self.fitness(self.models[i])

            print(np.max(self.scores))

            # calculate weights for the next generation
            best_models = self.best(2)
            p1 = best_models[0].get_weights()
            p2 = best_models[1].get_weights()

            # keep top two for next gen
            self.models[0].set_weights(p1)
            self.models[1].set_weights(p2)

            for i in range(2, self.population):
                weights = self.mutate(self.crossover(p1, p2))
                self.models[i].set_weights(weights)

    def best(self, count = 1) -> list[tf.keras.Model]:
        '''
        return the best model from current generation
        '''
        indices = np.argpartition(self.scores, -count)[-count:]
        
        return [self.models[i] for i in indices]

    def mutate(self, w: list[np.ndarray]):
        for i in range(len(w)):
            w[i] += rng.normal(scale=0.01,size=w[i].shape)

        return w
    
    def crossover(self, w1: list[np.ndarray], w2: list[np.ndarray]):
        w = [None for i in range(len(w1))]
        for i in range(len(w1)):
            select = np.random.randint(2, size=w1[i].shape)
            w[i] = w1[i] * select + w2[i] * (1 - select)
        return w
    
    def save(self, folder):
        '''
        save all models
        '''
        for i in range(self.population):
            self.models[i].save(f"{folder}/{i}")
        
    
    def load(self, folder):
        self.models = []
        for i in range(self.population):
            model = tf.keras.models.load_model(f"{folder}/{i}")
            self.models.append(model)
        return self

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(3, input_shape=(3,), activation='relu'),
        tf.keras.layers.Dense(10, input_shape=(3,), activation='relu'),
        tf.keras.layers.Dense(2, input_shape=(10,), activation='sigmoid')
    ])

    model.compile(loss='mse',optimizer='adam')
    return model

def predict(model, inputs):
    return model(np.atleast_2d(np.asarray(inputs)))[0]

r = car.Road().generate_rand(50)

def crash(sens):
    return sens[1] == math.inf or sens[1] <= 1

def fitness(model):
    c = car.Car(r.start, r.dir)
    for i in range(200):
        sens = car.sensors(c, r)

        if (crash(sens)): # check crash
            break

        a = predict(model, sens)
        c.drive(a[0], a[1])

    return i

draw_surface = None
pool = None

# multi threading for collision calculations
def getSensor(c):
    # if (scores[index] != 0):
    #     return None
    return r.sensors(c)

def sim(models):
    length = len(models)
    cars = [car.Car(r.start, r.dir) for i in range(length)]
    scores = np.zeros(length)

    time_last = time_ns()

    def draw():
        draw_surface.fill('#000000')
        r.draw(draw_surface, '#ffffff')
        for m in range(length):
            cars[m].draw(draw_surface, '#00ff00')

    for i in range(1, 150):
        # intersection calculation are costly, multi threading to increase performance
        sensors = pool.map(getSensor, cars)
        
        for m in range(length):
            if (scores[m] != 0):
                continue
            if (crash(sensors[m])): # check crash
                scores[m] = i
                continue

            a = predict(models[m], sensors[m])
            cars[m].drive(a[0], a[1])

        draw()
        
        if np.all(scores):
            break
    
    for m in range(length):
        if (scores[m] == 0):
            scores[m] = 150

    return scores

def ml_main():
    x = GeneticAlgorithm(create_model, population=30, fitness=fitness, sim=sim).load("models")

    for i in range(5):
        x.generation()

    x.save("models")

    print("waiting for pygame")


def main():
    global pygame_is_running
    global draw_surface
    global pool

    pool = mp.Pool(8)

    pygame.init()
    pygame.display.set_caption('Quick Start')
    window_surface = pygame.display.set_mode((800, 600))
    background = pygame.Surface((800, 600))
    pygame_is_running = True
    draw_surface = background

    # start ml thread
    ml_thread = threading.Thread(target=ml_main)
    ml_thread.start()

    while pygame_is_running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame_is_running = False

        window_surface.blit(background, (0, 0))
        pygame.display.update()

    ml_thread.join()

if __name__ == "__main__":
    main()
