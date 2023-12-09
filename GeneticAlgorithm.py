import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from random import randint

def initialization_of_population(size, n_feat):
    population = []
    for _ in range(size):
        chromosome = np.ones(n_feat, dtype=bool)
        chromosome[:int(0.3 * n_feat)] = False
        np.random.shuffle(chromosome)
        population.append(chromosome)
    return population

def fitness_score(population, X_train, X_test, Y_train, Y_test):
    scores = []
    for chromosome in population:
        logmodel.fit(X_train.iloc[:, chromosome], Y_train)
        predictions = logmodel.predict(X_test.iloc[:, chromosome])
        scores.append(accuracy_score(Y_test, predictions))
    scores, population = np.array(scores), np.array(population)
    inds = np.argsort(scores)
    return list(scores[inds][::-1]), list(population[inds, :][::-1])

def selection(pop_after_fit, n_parents):
    population_nextgen = []
    for i in range(n_parents):
        population_nextgen.append(pop_after_fit[i])
    return population_nextgen

def crossover(pop_after_sel):
    pop_nextgen = pop_after_sel.copy()
    for i in range(0, len(pop_after_sel), 2):
        new_par = []
        child_1, child_2 = pop_nextgen[i], pop_nextgen[i + 1]
        new_par = np.concatenate((child_1[:len(child_1)//2], child_2[len(child_1)//2:]))
        pop_nextgen.append(new_par)
    return pop_nextgen

def mutation(pop_after_cross, mutation_rate, n_feat):
    mutation_range = int(mutation_rate * n_feat)
    pop_next_gen = []
    for chromo in pop_after_cross:
        rand_posi = np.random.choice(n_feat, mutation_range, replace=False)
        chromo[rand_posi] = ~chromo[rand_posi]
        pop_next_gen.append(chromo)
    return pop_next_gen

def generations(data, label, size, n_feat, n_parents, mutation_rate, n_gen, X_train, X_test, Y_train, Y_test):
    best_chromo = []
    best_score = []
    population_nextgen = initialization_of_population(size, n_feat)
    for i in range(n_gen):
        scores, pop_after_fit = fitness_score(population_nextgen, X_train, X_test, Y_train, Y_test)
        print('Best score in generation', i + 1, ':', scores[:1])
        pop_after_sel = selection(pop_after_fit, n_parents)
        pop_after_cross = crossover(pop_after_sel)
        population_nextgen = mutation(pop_after_cross, mutation_rate, n_feat)
        best_chromo.append(pop_after_fit[0])
        best_score.append(scores[0])
    return best_chromo, best_score

# Assuming you have a split function, if not, you need to implement it
def split(data, label):
    X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.2, random_state=42)
    return X_train, X_test, Y_train, Y_test

data_pcos = pd.read_csv("F:\Semester 5\Artificial Intelligence\sampel.csv")
label_pcos = data_pcos["PCOS (Y/N)"]
data_pcos.drop(["Sl. No","Patient File No.","PCOS (Y/N)","Unnamed: 44","II    beta-HCG(mIU/mL)","AMH(ng/mL)"],axis = 1,inplace = True)
data_pcos["Marraige Status (Yrs)"].fillna(data_pcos['Marraige Status (Yrs)'].describe().loc[['50%']][0], inplace = True)
data_pcos["Fast food (Y/N)"].fillna(1, inplace = True)

# Initialize logmodel using RandomForestClassifier
logmodel = RandomForestClassifier(n_estimators=200, random_state=0)

# Split the data
X_train, X_test, Y_train, Y_test = split(data_pcos, label_pcos)

# Run the genetic algorithm
best_chromosome, score_pcos = generations(data_pcos, label_pcos, size=80, n_feat=data_pcos.shape[1], n_parents=64,
                                           mutation_rate=0.20, n_gen=10, X_train=X_train, X_test=X_test, Y_train=Y_train,
                                           Y_test=Y_test)

print("==================================")
print("")
# Extract the best chromosome from the last generation
yaya = best_chromosome[-1]

# Filter the training and test data based on the selected features
X_train_selected = X_train.iloc[:, yaya]
X_test_selected = X_test.iloc[:, yaya]

# Train the RandomForestClassifier using the selected features
logmodel.fit(X_train_selected, Y_train)

# Prediksi 20 data 
Y_pred = logmodel.predict(X_test_selected.iloc[:20, :])

# Evaluasi 20 data 
accuracy = accuracy_score(Y_test.iloc[:20], Y_pred)

# Print the PCOS labels in the specified format for the first 20 data
for actual, predicted in zip(Y_test.iloc[:20], Y_pred):
    print(f"Actual PCOS (Y/N) = {actual}, Predicted PCOS (Y/N) = {predicted}")

