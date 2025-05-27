import random
from itertools import product

objetos = []
CAPACIDAD_MAXIMA = 0
TAMANO_POBLACION = 30
TASA_MUTACION = 0.05
GENERACIONES_LIMITE = 1000

def generar_individuo():
    return [random.randint(0, 1) for _ in range(len(objetos))]

def generar_poblacion():
    return [generar_individuo() for _ in range(TAMANO_POBLACION)]

def calcular_fitness(individuo):
    peso_total = sum(objetos[i]["peso"] for i, bit in enumerate(individuo) if bit)
    valor_total = sum(objetos[i]["valor"] for i, bit in enumerate(individuo) if bit)
    return valor_total if peso_total <= CAPACIDAD_MAXIMA else 0

def seleccion_ruleta(poblacion, fitnesses):
    total = sum(fitnesses)
    if total == 0:
        return random.choice(poblacion)
    pick = random.uniform(0, total)
    acumulado = 0
    for ind, fit in zip(poblacion, fitnesses):
        acumulado += fit
        if acumulado >= pick:
            return ind

def crossover(p1, p2):
    punto = random.randint(1, len(p1)-1)
    return p1[:punto] + p2[punto:]

def mutar(individuo):
    for i in range(len(individuo)):
        if random.random() < TASA_MUTACION:
            individuo[i] = 1 - individuo[i]
    return individuo

def obtener_valor_optimo():
    max_val = 0
    for comb in product([0,1], repeat=len(objetos)):
        peso = sum(objetos[i]["peso"] for i, b in enumerate(comb) if b)
        valor = sum(objetos[i]["valor"] for i, b in enumerate(comb) if b)
        if peso <= CAPACIDAD_MAXIMA and valor > max_val:
            max_val = valor
    return max_val

def ejecutar_genetico():
    poblacion = generar_poblacion()
    mejor_global = (None, 0)
    valor_optimo = obtener_valor_optimo()
    historia = []

    for gen in range(1, GENERACIONES_LIMITE + 1):
        fitnesses = [calcular_fitness(ind) for ind in poblacion]
        mejor = max(zip(poblacion, fitnesses), key=lambda x: x[1])
        if mejor[1] > mejor_global[1]:
            mejor_global = mejor
        historia.append(mejor[1])
        if mejor_global[1] == valor_optimo:
            break
        nueva = []
        while len(nueva) < TAMANO_POBLACION:
            p1 = seleccion_ruleta(poblacion, fitnesses)
            p2 = seleccion_ruleta(poblacion, fitnesses)
            h = mutar(crossover(p1, p2))
            nueva.append(h)
        poblacion = nueva

    bits, val = mejor_global
    seleccion = [objetos[i] for i, b in enumerate(bits) if b]
    peso = sum(o["peso"] for o in seleccion)
    return seleccion, peso, val, historia
