import sys
import time
import string
import random
import logging

from optparse import OptionParser

logger = logging.getLogger('GA')

GENES = "".join(map(lambda x, y: x+y, string.ascii_uppercase, string.ascii_lowercase)) + \
        string.punctuation + " "

TARGET = "Hello world! Genetic Algorithm in action."

def sample_wr(population, k):
    n = len(population)
    _random, _int = random.random, int
    result = [None] * k
    for i in range(k):
        j = _int(_random() * n)
        result[i] = population[j]
    return result

def fitness(dnk, target):
    f = 0
    for index, gene in enumerate(dnk):
        if gene != target[index]:
            f -= 1
    return f

class GeneticCode:
    def __init__(self, dnk="", target=TARGET):
        if dnk == "":
            self.dnk = "".join(sample_wr(GENES, len(target)))
        else:
            self.dnk = dnk
        self.target = target

    def get(self):
        return self.dnk

    def fitness(self):
        return fitness(self.dnk, self.target)
    
    def mutate(self, turns=5):
        _dnk = list(self.dnk)
        for item in range(turns):
            rnd_elem_index = random.randint(0, len(_dnk)-1)
            if _dnk[rnd_elem_index] == self.target[rnd_elem_index]:
                pass
            else:
                _dnk[rnd_elem_index] = random.choice(GENES)
        self.dnk = "".join(_dnk)

    def replicate(self, another_dnk):
        part = random.randint(0, len(self.dnk)-1)
        return "".join(self.dnk[0:part] + another_dnk.get()[part:])


class GenePopulation():
    population_size = 1000
    
    def __init__(self, target=TARGET):
        self.population = [GeneticCode(target=target) for item in range(self.population_size)]
        self.target = target

    def _print(self):
        for item in self.population:
            print(item.get() + " - " + str(item.fitness()))
            

    def get_random(self):
        return self.population[random.randint(0, len(self.population)-1)]

    def darvin(self, winners=0.1):
        all_fitness = [(item.fitness(), item) for item in self.population]
        new_population = [item[1] for item in
                    sorted(all_fitness, key=lambda x: x[0], reverse=True)]
        self.population = new_population[:int(round(self.population_size * winners))]

        while len(self.population) < self.population_size:
            new_life = self.get_random().replicate(self.get_random())
            new_gc = GeneticCode(dnk=new_life, target=self.target)
            self.population.append(new_gc)

    def evolution(self, turns=1000):
        iterations = 0
        while (iterations < turns) and (self.population[0].get() != self.target):
            for index, item in enumerate(self.population):
                self.population[index].mutate()
            self.darvin()
            logger.info(self.population[0].get() + str(item.fitness())) 
            #xb = input('')
            time.sleep(0.1)
            iterations += 1
            
        return iterations


def main():
    usage = '{} [options] [text]'.format(sys.argv[0])
    parser = OptionParser(usage)
    parser.add_option('-l', '--log', default='-',
                      help='redirect logs to file')
    opts, args = parser.parse_args()

    if opts.log == '-':
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    else:
        logging.basicConfig(filename="helloworld.log", level=logging.INFO)

    if args:
        text = args[0]
    else:
        text = TARGET

    gp = GenePopulation(target=text)
    steps = gp.evolution()
    logger.info("Steps: {}".format(steps))
    

start_time = time.time()
main()
print()
print("Estimatied time:\t{}".format(time.time() - start_time))