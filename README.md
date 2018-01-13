# Laboratory Works
Laboratory works on the theory of machine learning.
Implementation of basic algorithms of machine learning, using all known libraries.
Отчет по практическим заданиям
Дисциплина: Теория машинного обучения
Студент: Литовченко Дмитрий Юрьевич
Преподаватель: Анафиев Айдер Сератович

Задание 1. Knn-классификатор
Для классификации каждого из объектов тестовой выборки необходимо последовательно выполнить следующие операции:
Вычислить расстояние до каждого из объектов обучающей выборки
Отобрать k объектов обучающей выборки, расстояние до которых минимально
Класс классифицируемого объекта — это класс, наиболее часто встречающийся среди k ближайших соседей
Для применения алгоритма был выбран датасет Wine из базы USI.
Листинг 1.1. Формирование датасета опущено
def getEuclideanDistance(instance_to, instance_from, length):
    distance = 0
    for i in range(length):
        distance += pow((instance_to[i] - instance_from[i]), 2)
    return math.sqrt(distance)


def getNeighbors(training_set, test_set_instance, k):
    distances = []
    length = len(test_set_instance) - 1
    for i in range(len(training_set)):
        dist = getEuclideanDistance(test_set_instance, training_set[i], length)
        distances.append((training_set[i], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])      
    return neighbors


def getClassVoteResult(neighbors):
    class_votes = {}
    for i in range(len(neighbors)):
        response = neighbors[i][0]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1
    sorted_votes = sorted(class_votes.items(),
                          key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]


def calculateAccuracy(test_set, predicted_values):
    correct = 0
    for i in range(len(test_set)):
        if test_set[i][0] == predicted_values[i]:
            correct += 1
    return (correct / float(len(test_set))) * 100.0

Результат выполнения скрипта

Задание 2. Наивный Байесовский классификатор
Был использован алгоритм из библиотеки nltk
В качестве датасета для реализации был выбран датасет из nltk.corpus, представленный в виде обзоров на фильмы.
Задачей классификатора было научиться определять хороший фильм или плохой, исходя из наиболее часто встречающихся слов в обзорах. Для тренировки использовалась выборка в 1900 документов, для теста – 1100. 
Листинг 2.1
all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]
        
training_set = featuresets[:1900]
testing_set =  featuresets[1900:]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Naive bayes accuracy:", (nltk.classify.accuracy(classifier, testing_set)) * 100)
classifier.show_most_informative_features(15)
Результат выполнения скрипта:

На данных мы видим какие признаки (в нашем случае слова) наиболее сильно характеризуют класс объекта (в нашем случае позитивную или негативную оценку).

Задание 3. Линейная регрессия
Задачей линейной регресии является определение зависимости одной переменной от другой или других с линейной функцией зависимости.
Для наилучшего представления линейной зависимости был подобран датасет зависимости цены дома от его площади и нахождения. Датасет взят на сайте https://wiki.csc.calpoly.edu  
Этапы решения задачи:
Нахождения дисперсий зависящих векторов
Вычисление ковариации векторов (в нашем случае цены, и квадратуры)
Зависимая переменная восстанавливается по полученным весам и заданным значениям свободной переменной
Нахождения среднеквадратичной ошибки по имеющимся данным для оценки результатов.

Листинг 3.1. Вспомогательные функции
def average(readings):
    readings_total = sum(readings)
    number_of_readings = len(readings)
    average = readings_total / float(number_of_readings)
    return average
 
 
def varianceOf(readings):
    readings_average = average(readings)
    mean_difference_squared_readings = [pow((reading - readings_average), 2) for reading in readings]
    variance = sum(mean_difference_squared_readings)
    return variance / float(len(readings) - 1)
 
 
def covarianceBetween(readings_1, readings_2):
    readings_1_average = average(readings_1)
    readings_2_average = average(readings_2)
    readings_size = len(readings_1)
    covariance = 0.0
    for i in xrange(0, readings_size):
        covariance += (readings_1[i] - readings_1_average) * (readings_2[i] - readings_2_average)
    return covariance / float(readings_size - 1)
 
 
def simpleLinearRegressionCoefficients(x_readings, y_readings):
    b1 = covarianceBetween(x_readings, y_readings) / float(varianceOf(x_readings))
    b0 = average(y_readings) - (b1 * average(x_readings))
    return b0, b1
 
 
def predict_target_value(x, b0, b1):
    return b0 + b1 * x
 
 
def rootMeanSquareError(actual_readings, predicted_readings):
    square_error_total = 0.0
    total_readings = len(actual_readings)
    for i in range(0, total_readings):
        error = predicted_readings[i] - actual_readings[i]
        square_error_total += pow(error, 2)
    rmse = square_error_total / float(total_readings)
    return rmse
Листинг 3.2. Главная функция
def simple_linear_regression(dataset):
    dataset_headers = get_headers(dataset)

    square_feet_average = average(dataset[dataset_headers[0]])
    price_average = average(dataset[dataset_headers[1]])
 
    square_feet_variance = varianceOf(dataset[dataset_headers[0]])
    price_variance = varianceOf(dataset[dataset_headers[1]])
 
    covariance_of_price_and_square_feet = dataset.cov()[dataset_headers[0]][dataset_headers[1]]
    w1 = covariance_of_price_and_square_feet / float(square_feet_variance)
 
    w0 = price_average - (w1 * square_feet_average)
    dataset['PredictedPrice'] = w0 + w1 * dataset[dataset_headers[0]]
    print(dataset[['PredictedPrice','Size']])

    print(rootMeanSquareError(dataset[dataset_headers[1]],dataset['PredictedPrice']))
    return dataset
Задание 4. Метод обратного распространения ошибки
Целью было обучить НС класифицировать ирисы по 4-ем признакам методом backpropagation. Был использован датасет iris.data из библиотеки sklearn. В качестве активационной функции использовалась формула гиперболического тангенса. НС обучается по следующему алгоритму:
Инициализировать синаптические веса маленькими случайными значениями.
Выбрать очередную обучающую пару из обучающего множества; подать входной вектор на вход сети.
Вычислить выход сети.
Вычислить разность между выходом сети и требуемым выходом (целевым вектором обучающей пары).
Подкорректировать веса сети для минимизации ошибки (как см. ниже).
Повторять шаги с 2 по 5 для каждого вектора обучающего множества до тех пор, пока ошибка на всем множестве не достигнет приемлемого уровня.
Листинг 4.1 класс НС
class MultiLayerPerceptron:
    int_num_epochs = 1000
    int_num_input_neurons = 0
    int_num_output_neurons = 0
    int_num_hidden_layers = 0
    int_num_hidden_neurons = 0
    dbl_mse_threshold = 0.001
    dbl_eta = 0.0001
    dbl_bias = 0.0002
    dbl_w0 = 0.0002

    arr_num_neurons_in_hidden_layers = []

    # Веса
    wo = []
    wh = []

    # Среднеквадратичная ошибка
    arr_mse = []


    def __init__(self, _int_num_input_neurons, _int_num_output_neurons, _int_num_hidden_layers, _int_num_epochs,
                 _int_num_hidden_neurons, _dbl_eta):
        self.int_num_input_neurons = _int_num_input_neurons
        self.int_num_output_neurons = _int_num_output_neurons
        self.int_num_hidden_layers = _int_num_hidden_layers
        self.int_num_epochs = _int_num_epochs
        self.int_num_hidden_neurons = _int_num_hidden_neurons
        self.dbl_eta = _dbl_eta

        
        self.wo = [[self.dbl_w0 for x in range(self.int_num_hidden_neurons + 1)] for y in
                   range(self.int_num_output_neurons)]  ## bias +1
        self.wh = [[self.dbl_w0 for x in range(self.int_num_input_neurons + 1)] for y in
                   range(self.int_num_hidden_neurons)]  ## bias +1
        return

    ## алгоритм
    def train(self, training_set):
        ## цикл поколений
        mse = []
        for e in range(0, self.int_num_epochs):
            ## сет для тренировки в цикле
            errors = []
            for t in range(0, len(training_set)):
                # Входные нейроны
                x = training_set[t][0]

                # Ответ
                d = training_set[t][1]

                # Путь вперед
                actual_hidden_output = self.hyberb(numpy.inner(self.wh, x))

                actual_hidden_output_plus_rshp = numpy.reshape(actual_hidden_output, (self.int_num_hidden_neurons))
                actual_hidden_output_plus_bias = numpy.append(actual_hidden_output_plus_rshp, self.dbl_bias)

                actual_output = self.hyberb(numpy.inner(self.wo, actual_hidden_output_plus_bias))

                error = d - actual_output

                ## Ошибки потом дополнительно оцениваются МНК
                errors = numpy.append(errors, error)

                ## Путь назад
                error_signal_output = error * self.derivhyberb(numpy.inner(self.wo, actual_hidden_output_plus_bias))

                error_signal_output_rshp = numpy.reshape(error_signal_output, (self.int_num_hidden_neurons))                
                error_signal_output_dump_bias = numpy.append(error_signal_output_rshp, self.dbl_bias)
                error_signal_hidden = self.derivhyberb(numpy.inner(self.wh, x)) * numpy.inner(self.wo,
                                                                                              error_signal_output_dump_bias)
                ## обновление весов
                tmp_wh = numpy.transpose(self.wh)
                counter = 0
                for x_ele in x:
                    delta_wh = self.dbl_eta * error_signal_hidden * x_ele
                    tmp_wh[counter] = delta_wh + tmp_wh[counter]  ## непосредственно оно
                    counter = counter + 1
                self.wh = tmp_wh.transpose()

                ## обновление выходных нейронов
                delta_wo = self.dbl_eta * error_signal_output * actual_hidden_output
                counter = 0
                for delta_wo_ele in delta_wo:
                    self.wo[counter] = self.wo[counter] + delta_wo_ele
                    counter = counter + 1
            self.arr_mse = numpy.append(self.arr_mse, numpy.mean(numpy.sum(numpy.square(errors))))
        
        return

    ## гиперболическая функция
    def hyberb(self, V):
        """
        :rtype: VECTOR SAME DIMENSIONS AS V
        """
        return (numpy.exp(V * 2) - 1) / (numpy.exp(V * 2) + 1)

    def derivhyberb(self, V):
        """
        :rtype: VECTOR SAME DIMENSIONS AS V
        """
        # PHI.arange().reshape
        return 4 * numpy.exp(V * 2) / numpy.square(1 + numpy.exp(V * 2))

результат выполнения для датасета iris.data представлен зависимостью среднеквадратичной ошибки от номера эпохи. Результаты очевидно коррелируют с этим показателем.
При этта(скорости изменения весов) равном 0.001:

Видно, что сети понадобилось всего несколько поколений при такой скорости обучения, что бы оптимизировать параметры.
При этта(скорости изменения весов) равном 0.0001:

Т.К. сила изменения весов уменьшилась, понадобилось больше эпох для решения задачи
В данной задаче важным является четко сбалансировать количество элементов сети и ее параметры, иначе результат может оказаться не актуальным, а скорость обучения может сильно упасть.






Задание 5. Генетический алгоритм
Генетический алгоритм представляет собой искусственный аналог действия эволюции. Генетический алгоритм проходит в 4 этапа.
Генерация начальной популяции
Селекция
Скрещивание и мутация
Проверка результата (повторение шагов 2, 3)
Следующий листинг приводит пример реализации генетического алгоритма.
Листинг 5.1 функции селекции, скрещивания и мутации
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
Листинг 5.2 Фитнесс-функция, штрафующая популяцию за несовпадения с целью эксперимента
def fitness(dnk, target):
    f = 0
    for index, gene in enumerate(dnk):
        if gene != target[index]:
            f -= 1
    return f
Результат выполнения скрипта
