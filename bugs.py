import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import time
import os

#---------------------------Parametry Symulacji------------------------------
GRID_WIDTH = 80
GRID_HEIGHT = 60
SIMULATION_STEPS = 800
ANIMATION_INTERVAL = 150 

#---------------------------Parametry (Wspólne i Wizualizacja)------------------------------
BUG_SIZE = 3

#---------------------------Parametry Ofiar (typ 'prey')------------------------------
NUM_PREY_INITIAL = 40
PREY_REPRODUCTION_RATE = 0.01
SMELL_EMISSION_AMOUNT = 1.0
PREY_SOUND_EMISSION_AMOUNT = 1.0 #: Ilość emitowanego dźwięku przez ofiarę


#---------------------------Parametry Drapieżników (typ 'predator')------------------------------
NUM_PREDATORS_INITIAL = 10
PREDATOR_INITIAL_ENERGY = 150
PREDATOR_ENERGY_PER_MOVE = 1.5
PREDATOR_ENERGY_GAIN_PER_PREY = 100
PREDATOR_MAX_ENERGY = 300
PREDATOR_MAX_AGE = 500
PREDATOR_INITIAL_AGE = 0
PREDATOR_MIN_ENERGY_REPRODUCTION = 200
PREDATOR_MIN_AGE_REPRODUCTION = 100
PREDATOR_MAX_AGE_REPRODUCTION = 400
PREDATOR_MUTATION_RATE = 0.1
#  Początkowy genotyp drapieżnika z parametrami słuchu
PREDATOR_INITIAL_GENOTYPE = {
    'smell_threshold': 0.1,      # próg czułości węchu
    'hearing_threshold': 0.05,   # próg czułości słuchu
    'damping_exponent_n': 2.5    # wykładnik tłumienia dźwięku (n > 2 dla nieidealnych warunków)
}


#---------------------------Parametry Zapachu------------------------------
SMELL_DIFFUSION_FACTOR = 0.375
SMELL_TRANSFER_FACTOR = 0.125
SMELL_DECAY_FACTOR = 0.125

#---------------------------Definicje ruchu i sąsiedztwa------------------------------
MOORE_NEIGHBORHOOD = [
    (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)
]
VON_NEUMAN_NEIGHBORHOOD = [
    (0, 1), (1, 0), (0, -1), (-1, 0)
]
TURNS = ['F', 'R', 'HR', 'RV', 'HL', 'L'] # Dla ofiar (ruch)
TURN_INDEX_CHANGE = {
    'F': 0, 'R': 1, 'HR': 2, 'RV': 3, 'HL': -2, 'L': -1
}
DIRECTIONS = [(0, 2), (2, 1), (2, -1), (0, -2), (-2, -1), (-2, 1)] # Dla ofiar (ruch)


#---------------------------Klasa Zwierzę (Ofiara/Drapieżnik)------------------------------
class Animal:
    def __init__(self, id, x, y, animal_type, initial_genotype=None):
        self.id = id
        self.x = x
        self.y = y
        self.animal_type = animal_type # 'prey' or 'predator'
        self.age = 0

        # --- Atrybuty i logika Ofiar (typ 'prey') ---
        if self.animal_type == 'prey':
             self.genotype = initial_genotype if initial_genotype is not None else {turn: 0 for turn in TURNS}
             self._turn_probabilities = self._calculate_turn_probabilities_prey()
             self._direction_idx = random.randint(0, len(DIRECTIONS) - 1)
             self._current_direction = DIRECTIONS[self._direction_idx]

        # --- Atrybuty i logika Drapieżników (typ 'predator') ---
        elif self.animal_type == 'predator':
             self.energy = PREDATOR_INITIAL_ENERGY
             self.energy_per_move = PREDATOR_ENERGY_PER_MOVE
             self.energy_gain_per_prey = PREDATOR_ENERGY_GAIN_PER_PREY
             self.max_energy = PREDATOR_MAX_ENERGY
             self.max_age = PREDATOR_MAX_AGE
             self.initial_age = PREDATOR_INITIAL_AGE
             self.min_energy_reproduction = PREDATOR_MIN_ENERGY_REPRODUCTION
             self.min_age_reproduction = PREDATOR_MIN_AGE_REPRODUCTION
             self.max_age_reproduction = PREDATOR_MAX_AGE_REPRODUCTION
             self.mutation_rate = PREDATOR_MUTATION_RATE
             # Genotyp drapieżnika kontroluje parametry zmysłu węchu i słuchu
             self.genotype = initial_genotype if initial_genotype is not None else PREDATOR_INITIAL_GENOTYPE.copy()
             self._current_direction = random.choice(MOORE_NEIGHBORHOOD) # Kierunek dla logiki ruchu drapieżnika

    def _calculate_turn_probabilities_prey(self):
        # Probabilistyka skrętów ofiary 
        gene_strengths = {turn: 2**self.genotype[turn] for turn in TURNS}
        total_strength = sum(gene_strengths.values())
        epsilon = 1e-9
        probabilities = {turn: strength / (total_strength + epsilon) for turn, strength in gene_strengths.items()}
        return probabilities

    def emit_smell(self):
        #---------------------------Ofiara emituje zapach------------------------------
        if self.animal_type == 'prey':
            return SMELL_EMISSION_AMOUNT
        return 0 # Inne zwierzęta nie emitują zapachu 

    def emit_sound(self):
        #---------------------------Ofiara emituje dźwięk------------------------------
        if self.animal_type == 'prey':
            return PREY_SOUND_EMISSION_AMOUNT 
        return 0 # Inne zwierzęta nie emitują dźwięku 

    def move(self, grid_width, grid_height, smell_grid=None, all_animals_list=None):
        #---------------------------Ruch zwierzęcia------------------------------
        if self.animal_type == 'prey':
             #---------------------------Ruch ofiary ------------------------------
             chosen_turn = random.choices(list(self._turn_probabilities.keys()), weights=list(self._turn_probabilities.values()), k=1)[0]
             turn_change = TURN_INDEX_CHANGE[chosen_turn]
             self._direction_idx = (self._direction_idx + turn_change) % len(DIRECTIONS)
             dx, dy = DIRECTIONS[self._direction_idx]
             self._current_direction = (dx, dy)

             self.x = (self.x + dx) % grid_width
             self.y = (self.y + dy) % grid_height


        elif self.animal_type == 'predator' and smell_grid is not None and all_animals_list is not None:
             #---------------------------Ruch drapieżnika (oparty na zapachu i słuchu)------------------------------
             # 1. Analiza zmysłu węchu
             smell_direction_info = self._sense_smell(smell_grid, grid_width, grid_height) # (best_direction_vector, max_smell)

             # 2. Analiza zmysłu słuchu
             sound_direction_info = self._sense_hearing(all_animals_list, grid_width, grid_height) # (best_direction_vector, loudest_sound)

             # 3. Rozstrzygnięcie konfliktu i wybór kierunku
             chosen_direction_vector = self._resolve_sensory_conflict(smell_direction_info, sound_direction_info)

             if chosen_direction_vector is not None:
                 self._current_direction = chosen_direction_vector
                 dx, dy = self._current_direction
                 self.x = (self.x + dx) % grid_width
                 self.y = (self.y + dy) % grid_height

                 if (dx, dy) != (0,0):
                      self.energy -= self.energy_per_move

             self.age += 1 # Drapieżnik starzeje się w każdym kroku


    def _sense_smell(self, smell_grid, grid_width, grid_height):
        #---------------------------Predator: Zmysł węchu------------------------------
        best_direction_vector = None
        max_smell = self.genotype['smell_threshold'] # Próg czułości węchu

        analyzed_coords_with_vectors = [(dx, dy) for dx, dy in MOORE_NEIGHBORHOOD]
        analyzed_coords_with_vectors.append((0,0)) # obecna pozycja

        for dx, dy in analyzed_coords_with_vectors:
             nx, ny = self.x + dx, self.y + dy
             wrapped_x, wrapped_y = nx % grid_width, ny % grid_height
             smell_intensity = smell_grid[wrapped_y, wrapped_x]

             if smell_intensity > max_smell:
                 max_smell = smell_intensity
                 best_direction_vector = (dx, dy) # wektor kierunku

        if best_direction_vector is None: # Jeśli nie ma zapachu powyżej progu
            best_direction_vector = (0,0) # Pozostań w miejscu (domyślnie)
            # return random.choice(MOORE_NEIGHBORHOOD + [(0,0)]) # lub losowy ruch
        return best_direction_vector, max_smell # Zwróć kierunek i najwyższą intensywność


    def _calculate_sound_intensity(self, distance, source_volume):
        #---------------------------Oblicza intensywność dźwięku na danej odległości------------------------------
        # g = g0 * l^(-n)
        if distance == 0: # W tej samej komórce, dźwięk jest maksymalny (źródło)
            return source_volume
        
        damping_exponent = self.genotype['damping_exponent_n'] # Wykładnik tłumienia z genotypu
        intensity = source_volume * (distance ** (-damping_exponent))
        return intensity


    def _sense_hearing(self, all_animals_list, grid_width, grid_height):
        #---------------------------Predator: Zmysł słuchu------------------------------
        best_direction_vector = None
        loudest_sound_intensity = self.genotype['hearing_threshold'] # Próg czułości słuchu

        # Przeszukaj wszystkie ofiary, aby znaleźć najbliższą i najgłośniejszą
        for target in all_animals_list:
             if target.animal_type == 'prey' and not target.is_dead(): # Słyszymy tylko żywe ofiary
                 distance = np.sqrt((self.x - target.x)**2 + (self.y - target.y)**2) # Odległość euklidesowa
                 # Zasięg słuchu: l_imax = g_i ^ (-1/n)
                 max_hearing_distance = self.genotype['hearing_threshold'] ** (-1.0 / self.genotype['damping_exponent_n'])
                 
                 if distance <= max_hearing_distance: # Tylko jeśli cel jest w maksymalnym zasięgu słuchu
                     source_volume = target.emit_sound() # Głośność źródła
                     sound_at_predator = self._calculate_sound_intensity(distance, source_volume)

                     if sound_at_predator > loudest_sound_intensity: # Jeśli dźwięk jest powyżej progu
                         loudest_sound_intensity = sound_at_predator
                         # Wyznacz kierunek do najgłośniejszej ofiary
                         dx = target.x - self.x
                         dy = target.y - self.y
                         best_direction_vector = (np.sign(dx) if dx != 0 else 0, np.sign(dy) if dy != 0 else 0)

                         if best_direction_vector == (0,0): # Jeśli ofiara jest w tej samej komórce
                              best_direction_vector = random.choice(MOORE_NEIGHBORHOOD) # Rusz się losowo


        if best_direction_vector is None: # Jeśli nie słyszano nic powyżej progu
            best_direction_vector = (0,0) # Pozostań w miejscu (domyślnie)
        return best_direction_vector, loudest_sound_intensity #  kierunek i najgłośniejszy dźwięk


    def _resolve_sensory_conflict(self, smell_info, sound_info):
        #---------------------------Rozstrzygnięcie konfliktu ------------------------------
        # Algorytm rozstrzygania konfliktu kierunków zmysłu węchu i słuchu
        smell_direction, max_smell = smell_info
        sound_direction, loudest_sound = sound_info

        # Algorytm:
        # 1. Jeśli oba zmysły nic nie wykryły, ruch losowy/pozostań w miejscu.
        if max_smell == self.genotype['smell_threshold'] and loudest_sound == self.genotype['hearing_threshold']:
            return random.choice(MOORE_NEIGHBORHOOD + [(0,0)])

        # 2. Jeśli tylko jeden zmysł coś wykrył, podążaj za nim.
        if max_smell > self.genotype['smell_threshold'] and loudest_sound <= self.genotype['hearing_threshold']:
            return smell_direction
        if loudest_sound > self.genotype['hearing_threshold'] and max_smell <= self.genotype['smell_threshold']:
            return sound_direction

        # 3. Jeśli oba coś wykryły, ale wskazują ten sam kierunek (lub podobny) - podążaj za nim.
        if smell_direction == sound_direction:
            return smell_direction

        # 4. Jeśli oba coś wykryły i wskazują różne kierunki - rozwiąż konflikt.
        smell_strength_above_threshold = max_smell - self.genotype['smell_threshold']
        sound_strength_above_threshold = loudest_sound - self.genotype['hearing_threshold']

        if smell_strength_above_threshold >= sound_strength_above_threshold:
            return smell_direction
        else:
            return sound_direction


    #---------------------------Metody interakcji ------------------------------

    def hunt_prey(self, all_animals_list):
        #---------------------------Drapieżnik poluje na ofiarę------------------------------
        if self.animal_type == 'predator':
            prey_eaten = None
            for target in all_animals_list:
                 if target.animal_type == 'prey' and target.x == self.x and target.y == self.y:
                     self.energy = min(self.energy + self.energy_gain_per_prey, self.max_energy)
                     prey_eaten = target
                     break

            return prey_eaten


    def is_dead(self):
        #---------------------------Sprawdza warunki śmierci------------------------------
        if self.animal_type == 'prey':
            return False
        elif self.animal_type == 'predator':
             return self.energy <= 0 or self.age >= self.max_age


    def can_reproduce(self):
        #---------------------------Sprawdza warunki rozmnażania------------------------------
        if self.animal_type == 'prey':
            return False
        elif self.animal_type == 'predator':
            return (self.energy > self.min_energy_reproduction and
                    self.age > self.min_age_reproduction and
                    self.age < self.max_age_reproduction)


    def reproduce(self, next_id):
        #---------------------------Tworzy potomka------------------------------
        if self.animal_type == 'predator' and self.can_reproduce():
            child_genotype = self.genotype.copy()
            child_type = self.animal_type

            if random.random() < self.mutation_rate:
                # Mutacja dla drapieżnika (genotyp węchu i słuchu)
                param_to_mutate = random.choice(list(child_genotype.keys()))
                change_factor = random.uniform(-0.2, 0.2) # Zmiana o +/- 20%
                child_genotype[param_to_mutate] *= (1 + change_factor)
                if 'threshold' in param_to_mutate: # Dla progów czułości
                     child_genotype[param_to_mutate] = max(0.0, child_genotype[param_to_mutate])
                if 'exponent' in param_to_mutate: # Dla wykładnika tłumienia
                     child_genotype[param_to_mutate] = max(2.0, child_genotype[param_to_mutate]) # n > 2


            child_energy = self.energy / 2.0
            self.energy /= 2.0
            child_age = 0

            return Animal(next_id, self.x, self.y, child_type, child_genotype)
        return None


#---------------------------Klasa Simulation------------------------------
class PredatorPreySimulation:
    def __init__(self, width, height, num_prey, num_predators, simulation_steps):
        self.width = width
        self.height = height
        self.simulation_steps = simulation_steps
        self.smell_grid = np.zeros((height, width), dtype=float)

        self.animals = []

        self._next_animal_id = 0

        #---------------------------Matplotlib Setup------------------------------
        self.fig, self.ax = plt.subplots(figsize=(width/8, height/8))
        self.ax.set_xlim(-0.5, self.width - 0.5)
        self.ax.set_ylim(-0.5, self.height - 0.5)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_title("Symulacja Drapieżnik-Ofiara (Spacja=Pauza/Wznów, Strzałka Prawa=Krok, P=Pokaż/Ukryj Zapach)")
        self.ax.set_xlabel("x - axis")
        self.ax.set_ylabel("y -axis")

        self.ax.set_xticks(np.arange(-0.5, self.width, 1), minor=False)
        self.ax.set_yticks(np.arange(-0.5, self.height, 1), minor=False)
        self.ax.grid(which='major', color='gray', linestyle='-', linewidth=0.5)
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])

        #---------------------------Obiekty do wizualizacji------------------------------
        self.smell_image = self.ax.imshow(self.smell_grid, cmap='hot', origin='lower', extent=[-0.5, self.width-0.5, -0.5, self.height-0.5], vmin=0, vmax=SMELL_EMISSION_AMOUNT * 5, alpha=0.0)

        self.prey_scatter = self.ax.scatter([], [], color='blue', label='Ofiary', s=50)
        self.predator_scatter = self.ax.scatter([], [], color='red', label='Drapieżniki', s=70, marker='X')

        self.ax.legend()

        self._ani = None
        self._paused = False
        self._current_step = 0
        self._show_smell = False

    def on_key_press(self, event):
        #---------------------------Obsługa klawiszy------------------------------
        if event.key == ' ':
            if self._paused:
                self._ani.event_source.start()
                print(f"Symulacja wznowiona. Krok: {self._current_step}")
            else:
                self._ani.event_source.stop()
                print(f"Symulacja zatrzymana. Krok: {self._current_step}")
            self._paused = not self._paused

        elif event.key == 'right' and self._paused:
            if self._current_step < self.simulation_steps:
                self._current_step += 1
                self.run_step()
                self.update_visualization()
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                print(f"Wykonano krok naprzód. Krok: {self._current_step}")
            else:
                print("Osiągnięto maksymalną liczbę kroków symulacji.")

        elif event.key == 'p':
             self._show_smell = not self._show_smell
             self.smell_image.set_alpha(0.7 if self._show_smell else 0.0)
             self.fig.canvas.draw_idle()


    def initialize(self, num_prey, num_predators):
        #---------------------------Inicjalizacja zwierząt------------------------------
        for i in range(num_prey):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            initial_genotype = {turn: 0 for turn in TURNS}
            self.animals.append(Animal(self._next_animal_id, x, y, 'prey', initial_genotype))
            self._next_animal_id += 1

        for i in range(num_predators):
             x = random.randint(0, self.width - 1)
             y = random.randint(0, self.height - 1)
             initial_genotype = PREDATOR_INITIAL_GENOTYPE.copy()
             self.animals.append(Animal(self._next_animal_id, x, y, 'predator', initial_genotype))
             self._next_animal_id += 1


    def diffuse_smell(self):
        #---------------------------Moduluje dyfuzję zapachu------------------------------
        new_smell_grid = np.zeros_like(self.smell_grid)

        for animal in self.animals:
             if animal.animal_type == 'prey':
                 if 0 <= animal.y < self.height and 0 <= animal.x < self.width:
                      new_smell_grid[animal.y, animal.x] += animal.emit_smell()

        temp_smell_grid = self.smell_grid.copy()

        for r in range(self.height):
             for c in range(self.width):
                smell_here = temp_smell_grid[r, c]
                if smell_here > 1e-6:
                     new_smell_grid[r, c] += smell_here * SMELL_DIFFUSION_FACTOR
                     transfer_amount = smell_here * SMELL_TRANSFER_FACTOR
                     for dr, dc in VON_NEUMAN_NEIGHBORHOOD:
                         nr, nc = (r + dr) % self.height, (c + dc) % self.width
                         new_smell_grid[nr, nc] += transfer_amount

        self.smell_grid = new_smell_grid


    def add_prey_randomly(self):
        #---------------------------Losowe pojawianie się ofiar------------------------------
        if random.random() < PREY_REPRODUCTION_RATE:
             x = random.randint(0, self.width - 1)
             y = random.randint(0, self.height - 1)
             new_prey = Animal(self._next_animal_id, x, y, 'prey', {turn: 0 for turn in TURNS})
             self.animals.append(new_prey)
             self._next_animal_id += 1


    def update_visualization(self):
        #---------------------------Aktualizuje wizualizację------------------------------
        prey_x = []
        prey_y = []
        predator_x = []
        predator_y = []

        for animal in self.animals:
             if animal.animal_type == 'prey':
                 prey_x.append(animal.x)
                 prey_y.append(animal.y)
             elif animal.animal_type == 'predator':
                 predator_x.append(animal.x)
                 predator_y.append(animal.y)

        self.prey_scatter.set_offsets(np.c_[prey_x, prey_y])
        self.predator_scatter.set_offsets(np.c_[predator_x, predator_y])

        self.smell_image.set_data(self.smell_grid)

        return (self.prey_scatter, self.predator_scatter, self.smell_image)


    def run_step(self):
        #---------------------------Wykonuje jeden krok symulacji------------------------------
        animals_to_remove = []
        new_animals = []

        # --- Ruch i interakcje ---
        for animal in self.animals:
             if not animal.is_dead(): 
                 animal.move(self.width, self.height, self.smell_grid, self.animals) 

                 if animal.animal_type == 'predator':
                     eaten_prey = animal.hunt_prey(self.animals)
                     if eaten_prey:
                         animals_to_remove.append(eaten_prey)

                 if animal.animal_type == 'predator': # Tylko drapieżniki się rozmnażają
                      child = animal.reproduce(self._next_animal_id)
                      if child:
                          new_animals.append(child)
                          self._next_animal_id += 1

             if animal.animal_type == 'predator' and animal.is_dead(): # Tylko drapieżniki umierają naturalnie
                  animals_to_remove.append(animal)


        # --- Usunięcie/Dodanie zwierząt ---
        self.animals = [animal for animal in self.animals if animal not in animals_to_remove]
        self.animals.extend(new_animals)
        self.add_prey_randomly() # Losowe pojawianie się ofiar

        # --- Dyfuzja Zapachu ---
        self.diffuse_smell()

        # --- Aktualizacja kroku i wizualizacji ---
        self._current_step += 1
        return self.update_visualization()

    def animate(self, i):
         #---------------------------Funkcja animacji------------------------------
         self.run_step()
         return self.update_visualization()


    def run_with_animation(self, num_steps):
        #---------------------------Uruchomienie symulacji------------------------------
        self.initialize(NUM_PREY_INITIAL, NUM_PREDATORS_INITIAL)
        print(f"Uruchamianie symulacji Drapieżnik-Ofiara dla {num_steps} kroków...")
        print("Sterowanie: Spacja = Pauza/Wznów, Strzałka Prawa = Krok naprzód (tylko w pauzie), P = Pokaż/Ukryj Zapach")

        self._ani = animation.FuncAnimation(self.fig, self.animate, frames=num_steps, interval=ANIMATION_INTERVAL, blit=False, repeat=False)

        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        plt.show()
        print("Symulacja zakończona.")
        final_prey_count = sum(1 for a in self.animals if a.animal_type == 'prey')
        final_predator_count = sum(1 for a in self.animals if a.animal_type == 'predator')
        print(f"Pozostało ofiar: {final_prey_count}")
        print(f"Pozostało drapieżników: {final_predator_count}")


#---------------------------Uruchomienie główne------------------------------
if __name__ == "__main__":
    sim = PredatorPreySimulation(GRID_WIDTH, GRID_HEIGHT, NUM_PREY_INITIAL, NUM_PREDATORS_INITIAL, SIMULATION_STEPS)
    sim.run_with_animation(SIMULATION_STEPS)