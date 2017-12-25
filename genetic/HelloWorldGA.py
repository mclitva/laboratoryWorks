import sys
import time
import string
import random
import logging

from optparse import OptionParser

__version__ = "1.0"

logger = logging.getLogger('GA')

# form genetic codes
GENES = "".join(map(lambda x, y: x+y, string.ascii_uppercase, string.ascii_lowercase)) + \
        string.punctuation + " "

# target string C.O.
TARGET = "Hello world! Genetic Algorithm in action."

def fitness(dnk, target):
    """
    calculates how close is dnk to target <<< this called "fitness function"
    f = 0 if conditions are satisfied
    """
    f = 0
    for index, gene in enumerate(dnk):
        if gene != target[index]:
            f -= 1
    return f

def sample_wr(population, k):
    "Chooses k random elements (with replacement) from a population"
    n = len(population)
    _random, _int = random.random, int  # speed hack 
    result = [None] * k
    for i in range(k):
        j = _int(_random() * n)
        result[i] = population[j]
    return result


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
        """
        mutate dnk sequence "on place"
        turns - how much elements will be changed
        """
        _dnk = list(self.dnk)
        for item in range(turns):
            rnd_elem_index = random.randint(0, len(_dnk)-1)
            if _dnk[rnd_elem_index] == self.target[rnd_elem_index]:
                pass
            else:
                _dnk[rnd_elem_index] = random.choice(GENES)
        self.dnk = "".join(_dnk)

    def replicate(self, another_dnk):
        """
        breed 2 dnk sequences
        cut one, cut two and mix it together
        return offspring dnk string
        """
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
        "Get random element from population"
        return self.population[random.randint(0, len(self.population)-1)]

    def darvin(self, winners=0.1):
        """
        choose only good dnk sequences
        winners - part of population to breed
        """
        all_fitness = [(item.fitness(), item) for item in self.population]
        new_population = [item[1] for item in
                    sorted(all_fitness, key=lambda x: x[0], reverse=True)]
        self.population = new_population[:int(round(self.population_size * winners))]

        while len(self.population) < self.population_size:
            new_life = self.get_random().replicate(self.get_random())
            new_gc = GeneticCode(dnk=new_life, target=self.target)
            self.population.append(new_gc)

    def evolution(self, turns=1000):
        """Evalute population"""
        iterations = 0
        while (iterations < turns) and (self.population[0].get() != self.target):
            for index, item in enumerate(self.population):
                self.population[index].mutate()
            self.darvin()
            logger.info(self.population[0].get() + str(item.fitness()))
            time.sleep(0.1)
            iterations += 1
            
        return iterations


def main():
    usage = '%s [options] [text]' % sys.argv[0]
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
    logger.info("Steps: %d" % steps)
    

start_time = time.time()
main()
print()
print("Estimatied time:\t%s" % (time.time() - start_time))