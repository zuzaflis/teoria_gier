import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon, Circle
import math

# ---------------------------------- Global parameters -------------------------------
num_boids = 50
x_limit = 50
y_limit = 50

max_speed = 2.0
max_acceleration = 0.5
neighborhood_radius = 10
min_distance = 3
separation_weight = 0.1
alignment_weight = 0.05
cohesion_weight = 0.05
attraction_weight = 0.05
goal_position = np.array([x_limit / 2, y_limit / 2], dtype=float)
repulsion_weight = 0.1
repulsion_position = np.array([x_limit * 0.25, y_limit * 0.75], dtype=float)
limit_walls = True # Flag to enable/disable limited walls


# ------------------------Predator-prey parameters------------------------
predator_prey_enabled = True # Flag to enable/disable predator-prey interaction
predator_ratio = 0.1 # Ratio of predators to total boids
predator_perception_radius = 15
prey_flee_radius = 15
predator_pursuit_weight = 0.1
prey_flee_weight = 0.2

# ------------------------Obstacle avoidance parameters------------------------
obstacle_avoidance_enabled = True # Flag to enable/disable obstacles 
obstacles = [np.array([15, 15]), np.array([35, 35])] # List of obstacle positions
obstacle_radius = 5
obstacle_avoidance_distance = 10
obstacle_avoidance_weight = 0.5

# ------------------------Resting and feeding parameters------------------------
resting_enabled = False # Flag to enable/disable resting
resting_probability = 0.01
resting_duration_range = [50, 100] # Range of resting duration in frames
feeding_enabled = False # Flag to enable/disable feeding


# --------------------------------Feeding -------------------------------
food_sources = [np.array([10, 40]), np.array([40, 10])] # List food source positions
food_radius = 3
hunger_increase_rate = 0.1
max_hunger = 1.0
feeding_satisfaction = 0.5
feeding_range = 5

class Boid:
    def __init__(self, position, velocity, boid_type="normal"):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.size = 1.0  # triangle size
        self.boid_type = boid_type
        self.state = "flying"  # "flying", "resting", "feeding"
        self.rest_timer = 0
        self.rest_duration = 0
        self.hunger = 0.0
        self.food_target = None

    def update(self, boids):
        global separation_weight, alignment_weight, cohesion_weight, neighborhood_radius, min_distance, goal_position, attraction_weight, repulsion_position, repulsion_weight, max_speed, max_acceleration, predator_prey_enabled, predator_perception_radius, prey_flee_radius, predator_pursuit_weight, prey_flee_weight, obstacle_avoidance_enabled, obstacles, obstacle_radius, obstacle_avoidance_distance, obstacle_avoidance_weight, resting_enabled, resting_probability, resting_duration_range, feeding_enabled, food_sources, food_radius, hunger_increase_rate, max_hunger, feeding_satisfaction, feeding_range
        
        if resting_enabled and self.state == "flying" and np.random.rand() < resting_probability:
            self.state = "resting"
            self.velocity = np.zeros_like(self.velocity)
            self.rest_duration = np.random.randint(resting_duration_range[0], resting_duration_range[1])
            self.rest_timer = 0
        elif self.state == "resting":
            self.rest_timer += 1
            if self.rest_timer >= self.rest_duration:
                self.state = "flying"
                
        if feeding_enabled:
            self.hunger = min(max_hunger, self.hunger + hunger_increase_rate)
            if self.state == "flying" and self.hunger > max_hunger * 0.8:
                self.food_target = self.find_nearest_food()
                if self.food_target is not None:
                    distance_to_food = np.linalg.norm(self.position - self.food_target)
                    if distance_to_food < feeding_range:
                        self.state = "feeding"
                        self.velocity = np.zeros_like(self.velocity)
                    else:
                        self.velocity += self.move_towards(self.food_target) * attraction_weight # attraction weight to move towards food
            elif self.state == "feeding":
                self.hunger -= feeding_satisfaction
                if self.hunger <= 0:
                    self.state = "flying"
                    self.food_target = None

        if self.state == "flying":
            v1 = self.rule1(boids)
            v2 = self.rule2(boids)
            v3 = self.rule3(boids)
            v4 = self.rule4_attraction()
            v5 = self.rule5_repulsion()
            v6 = np.zeros_like(self.velocity)
            if predator_prey_enabled:
                if self.boid_type == "prey":
                    v6 = self.rule_flee(boids)
                elif self.boid_type == "predator":
                    v6 = self.rule_pursue(boids)
            v7 = np.zeros_like(self.velocity)
            if obstacle_avoidance_enabled:
                v7 = self.rule_avoid_obstacles()

            acceleration = v1 + v2 + v3 + v4 + v5 + v6 + v7
            accel_magnitude = np.linalg.norm(acceleration)

            if accel_magnitude > max_acceleration:
                acceleration = acceleration / accel_magnitude * max_acceleration

            self.velocity += acceleration

            # Limit speed
            speed = np.linalg.norm(self.velocity)
            if speed > max_speed:
                self.velocity = self.velocity / speed * max_speed

            self.position += self.velocity

    def move_towards(self, target):
        desired_velocity = target - self.position
        distance = np.linalg.norm(desired_velocity)
        if distance > 0:
            return desired_velocity / distance * max_speed
        return np.zeros_like(self.velocity)

    def find_nearest_food(self):
        nearest_food = None
        min_distance = float('inf')
        for food in food_sources:
            distance = np.linalg.norm(self.position - food)
            if distance < min_distance:
                min_distance = distance
                nearest_food = food
        return nearest_food

    def rule1(self, boids):
        sep_velocity = np.zeros_like(self.velocity)
        close_boids = 0
        for boid in boids:
            if boid != self:
                distance = np.linalg.norm(self.position - boid.position)
                if distance < min_distance:
                    sep_velocity -= (boid.position - self.position) / distance
                    close_boids += 1
        if close_boids > 0:
            sep_velocity /= close_boids
        return sep_velocity * separation_weight

    def rule2(self, boids):
        ali_velocity = np.zeros_like(self.velocity)
        nearby_boids = 0
        for boid in boids:
            if boid != self:
                distance = np.linalg.norm(self.position - boid.position)
                if distance < neighborhood_radius:
                    ali_velocity += boid.velocity
                    nearby_boids += 1
        if nearby_boids > 0:
            ali_velocity /= nearby_boids
            return (ali_velocity - self.velocity) * alignment_weight
        return ali_velocity

    def rule3(self, boids):
        coh_velocity = np.zeros_like(self.velocity)
        nearby_boids = 0
        center_of_mass = np.zeros_like(self.position)
        for boid in boids:
            if boid != self:
                distance = np.linalg.norm(self.position - boid.position)
                if distance < neighborhood_radius:
                    center_of_mass += boid.position
                    nearby_boids += 1
        if nearby_boids > 0:
            center_of_mass /= nearby_boids
            return (center_of_mass - self.position) * cohesion_weight
        return coh_velocity

    def rule4_attraction(self):
        desired_velocity = goal_position - self.position
        distance_to_goal = np.linalg.norm(desired_velocity)
        if distance_to_goal > 0:
            desired_velocity = desired_velocity / distance_to_goal * max_speed
            return (desired_velocity - self.velocity) * attraction_weight
        return np.zeros_like(self.velocity)

    def rule5_repulsion(self):
        vector_to_repulsion = self.position - repulsion_position
        distance = np.linalg.norm(vector_to_repulsion)
        if distance < 10:  # Only apply repulsion within  range
            if distance > 0:
                repulsion_velocity = vector_to_repulsion / distance * max_speed * repulsion_weight
                return repulsion_velocity - self.velocity
            else:
                return np.random.rand(2) * max_speed * repulsion_weight
        return np.zeros_like(self.velocity)
 
    def rule_avoid_obstacles(self):
        avoid_velocity = np.zeros_like(self.velocity)
        for obstacle in obstacles:
            distance = np.linalg.norm(self.position - obstacle)
            if distance < obstacle_avoidance_distance + obstacle_radius:
                direction_away = (self.position - obstacle)
                norm = np.linalg.norm(direction_away)
                if norm > 0:
                    avoid_velocity += direction_away / norm * (obstacle_avoidance_distance + obstacle_radius - distance)
        return avoid_velocity * obstacle_avoidance_weight   

    def rule_flee(self, boids):
        flee_velocity = np.zeros_like(self.velocity)
        nearby_predators = 0
        for boid in boids:
            if boid != self and boid.boid_type == "predator":
                distance = np.linalg.norm(self.position - boid.position)
                if distance < prey_flee_radius:
                    flee_velocity += (self.position - boid.position) / distance
                    nearby_predators += 1
        if nearby_predators > 0:
            flee_velocity /= nearby_predators
        return flee_velocity * prey_flee_weight

    def rule_pursue(self, boids):
        pursue_velocity = np.zeros_like(self.velocity)
        nearby_prey = []
        for boid in boids:
            if boid != self and boid.boid_type == "prey":
                distance = np.linalg.norm(self.position - boid.position)
                if distance < predator_perception_radius:
                    nearby_prey.append(boid)

        if nearby_prey:
            closest_prey = min(nearby_prey, key=lambda b: np.linalg.norm(self.position - b.position))
            direction_to_prey = closest_prey.position - self.position
            norm = np.linalg.norm(direction_to_prey)
            if norm > 0:
                pursue_velocity = direction_to_prey / norm * max_speed
                return (pursue_velocity - self.velocity) * predator_pursuit_weight
        return pursue_velocity

def initialize_boids(num_boids, x_limit, y_limit, predator_ratio):
    boids = []
    num_predators = int(num_boids * predator_ratio)
    for i in range(num_boids):
        position = [np.random.uniform(0, x_limit), np.random.uniform(0, y_limit)]
        velocity = [np.random.uniform(-max_speed, max_speed), np.random.uniform(-max_speed, max_speed)]
        boid_type = "normal"
        if i < num_predators:
            boid_type = "predator"
        elif i >= num_predators and i < num_boids - int(num_boids * predator_ratio * 0.5) and predator_ratio > 0: 
            boid_type = "prey"
        boids.append(Boid(position, velocity, boid_type))
    return boids

def update_boids(frame, boids, boid_patches, goal_point, repulsion_point, obstacle_patches, food_patches):
    global limit_walls
    for i, boid in enumerate(boids):
        boid.update(boids)

        # Boundary conditions:wrap around (toroidal)
        if not limit_walls:
            if boid.position[0] > x_limit:
                boid.position[0] -= x_limit
            elif boid.position[0] < 0:
                boid.position[0] += x_limit
            if boid.position[1] > y_limit:
                boid.position[1] -= y_limit
            elif boid.position[1] < 0:
                boid.position[1] += y_limit
        # Boundary conditions: limited walls (bouncing)
        else:
            if boid.position[0] > x_limit:
                boid.position[0] = x_limit
                boid.velocity[0] *= -1
            elif boid.position[0] < 0:
                boid.position[0] = 0
                boid.velocity[0] *= -1
            if boid.position[1] > y_limit:
                boid.position[1] = y_limit
                boid.velocity[1] *= -1
            elif boid.position[1] < 0:
                boid.position[1] = 0
                boid.velocity[1] *= -1

        # Update boid triangle color based on type and state
        if boid.state == "resting":
            boid_patches[i].set_facecolor('lightgray')
        elif boid.state == "feeding":
            boid_patches[i].set_facecolor('lightgreen')
        elif boid.boid_type == "predator":
            boid_patches[i].set_facecolor('red')
        elif boid.boid_type == "prey":
            boid_patches[i].set_facecolor('blue')
        else:
            boid_patches[i].set_facecolor('blue')

        # Update boids position and orientation
        angle = np.arctan2(boid.velocity[1], boid.velocity[0])
        size = boid.size
        tip = boid.position + size * np.array([np.cos(angle), np.sin(angle)])
        left_base = boid.position + size * np.array([np.cos(angle + 2 * np.pi / 3), np.sin(angle + 2 * np.pi / 3)]) * 0.7
        right_base = boid.position + size * np.array([np.cos(angle - 2 * np.pi / 3), np.sin(angle - 2 * np.pi / 3)]) * 0.7
        vertices = np.array([tip, left_base, right_base])
        boid_patches[i].set_xy(vertices)

    # Update goal and repulsion point positions
    goal_point.set_data([goal_position[0]], [goal_position[1]])
    repulsion_point.set_data([repulsion_position[0]], [repulsion_position[1]])

    return [*boid_patches, goal_point, repulsion_point, *obstacle_patches, *food_patches]

def on_key_press(event):
    global max_speed, max_acceleration, separation_weight, alignment_weight, cohesion_weight, attraction_weight, repulsion_weight, neighborhood_radius, min_distance, goal_position, repulsion_position, limit_walls, predator_prey_enabled, predator_ratio, predator_perception_radius, prey_flee_radius, predator_pursuit_weight, prey_flee_weight, obstacle_avoidance_enabled, resting_enabled, feeding_enabled, food_sources
    if event.key == 'w':
        max_speed += 0.1
        print(f"Max speed increased to: {max_speed:.1f}")
    elif event.key == 's':
        max_speed = max(0.1, max_speed - 0.1)
        print(f"Max speed decreased to: {max_speed:.1f}")
    elif event.key == 'a':
        max_acceleration += 0.05
        print(f"Max acceleration increased to: {max_acceleration:.2f}")
    elif event.key == 'd':
        max_acceleration = max(0.01, max_acceleration - 0.05)
        print(f"Max acceleration decreased to: {max_acceleration:.2f}")
    elif event.key == '1':
        separation_weight += 0.01
        print(f"Separation weight increased to: {separation_weight:.2f}")
    elif event.key == '2':
        separation_weight = max(0.0, separation_weight - 0.01)
        print(f"Separation weight decreased to: {separation_weight:.2f}")
    elif event.key == '3':
        alignment_weight += 0.01
        print(f"Alignment weight increased to: {alignment_weight:.2f}")
    elif event.key == '4':
        alignment_weight = max(0.0, alignment_weight - 0.01)
        print(f"Alignment weight decreased to: {alignment_weight:.2f}")
    elif event.key == '5':
        cohesion_weight += 0.01
        print(f"Cohesion weight increased to: {cohesion_weight:.2f}")
    elif event.key == '6':
        cohesion_weight = max(0.0, cohesion_weight - 0.1)
        print(f"Cohesion weight decreased to: {cohesion_weight:.2f}")
    elif event.key == '7':
        attraction_weight += 0.01
        print(f"Attraction weight increased to: {attraction_weight:.2f}")
    elif event.key == '8':
        attraction_weight = max(0.0, attraction_weight - 0.01)
        print(f"Attraction weight decreased to: {attraction_weight:.2f}")
    elif event.key == '9':
        repulsion_weight += 0.01
        print(f"Repulsion weight increased to: {repulsion_weight:.2f}")
    elif event.key == '0':
        repulsion_weight = max(0.0, repulsion_weight - 0.01)
        print(f"Repulsion weight decreased to: {repulsion_weight:.2f}")
    elif event.key == 'g':
        goal_position[0] += 1
        print(f"Goal position moved to: {goal_position}")
    elif event.key == 'h':
        goal_position[0] -= 1
        print(f"Goal position moved to: {goal_position}")
    elif event.key == 'j':
        goal_position[1] += 1
        print(f"Goal position moved to: {goal_position}")
    elif event.key == 'k':
        goal_position[1] -= 1
        print(f"Goal position moved to: {goal_position}")
    elif event.key == 'r':
        repulsion_position[0] += 1
        print(f"Repulsion position moved to: {repulsion_position}")
    elif event.key == 't':
        repulsion_position[0] -= 1
        print(f"Repulsion position moved to: {repulsion_position}")
    elif event.key == 'y':
        repulsion_position[1] += 1
        print(f"Repulsion position moved to: {repulsion_position}")
    elif event.key == 'u':
        repulsion_position[1] -= 1
        print(f"Repulsion position moved to: {repulsion_position}")
    elif event.key == 'n':
        neighborhood_radius += 1
        print(f"Neighborhood radius increased to: {neighborhood_radius}")
    elif event.key == 'm':
        neighborhood_radius = max(1, neighborhood_radius - 1)
        print(f"Neighborhood radius decreased to: {neighborhood_radius:.2f}")
    elif event.key == ',':
        min_distance += 0.1
        print(f"Minimum distance increased to: {min_distance:.1f}")
    elif event.key == '.':
        min_distance = max(0.1, min_distance - 0.1)
        print(f"Minimum distance decreased to: {min_distance:.1f}")
    elif event.key == 'b':
        limit_walls = not limit_walls
        print(f"Limited walls toggled to: {limit_walls}")
    elif event.key == 'p':
        predator_prey_enabled = not predator_prey_enabled
        print(f"Predator-Prey mode toggled to: {predator_prey_enabled}")
    # elif event.key == 'v':
    #     predator_ratio = min(1.0, predator_ratio + 0.05)
    #     print(f"Predator ratio increased to: {predator_ratio:.2f}")
    elif event.key == 'c':
        predator_ratio = max(0.0, predator_ratio - 0.05)
        print(f"Predator ratio decreased to: {predator_ratio:.2f}")
    elif event.key == 'o':
        predator_perception_radius += 1
        print(f"Predator perception radius increased to: {predator_perception_radius}")
    elif event.key == 'i':
        predator_perception_radius = max(1, predator_perception_radius - 1)
        print(f"Predator perception radius decreased to: {predator_perception_radius}")
    elif event.key == 'l':
        prey_flee_radius += 1
        print(f"Prey flee radius increased to: {prey_flee_radius}")
    elif event.key == 'k':
        prey_flee_radius = max(1, prey_flee_radius - 1)
        print(f"Prey flee radius decreased to: {prey_flee_radius:.2f}")
    elif event.key == ';':
        predator_pursuit_weight += 0.01
        print(f"Predator pursuit weight increased to: {predator_pursuit_weight:.2f}")
    elif event.key == "'":
        predator_pursuit_weight = max(0.0, predator_pursuit_weight - 0.01)
        print(f"Predator pursuit weight decreased to: {predator_pursuit_weight:.2f}")
    # elif event.key == 'z':
    #     prey_flee_weight += 0.01
    #     print(f"Prey flee weight increased to: {prey_flee_weight:.2f}")
    # elif event.key == 'x':
    #     prey_flee_weight = max(0.0, prey_flee_weight - 0.01)
    #     print(f"Prey flee weight decreased to: {prey_flee_weight:.2f}")
    # elif event.key == 'v':
    #     obstacle_avoidance_enabled = not obstacle_avoidance_enabled
    #     print(f"Obstacle Avoidance toggled to: {obstacle_avoidance_enabled}")
    elif event.key == '[':
        global obstacles
        obstacles.append(np.array([np.random.uniform(0, x_limit), np.random.uniform(0, y_limit)]))
        print(f"Added obstacle at: {obstacles[-1]}")
    elif event.key == ']':
        if obstacles:
            obstacles.pop()
            print(f"Removed last obstacle.")
    elif event.key == 'z':
        resting_enabled = not resting_enabled
        print(f"Resting toggled to: {resting_enabled}")
    elif event.key == 'v':
        feeding_enabled = not feeding_enabled
        print(f"Feeding toggled to: {feeding_enabled}")
    elif event.key == '+':
        food_sources.append(np.array([np.random.uniform(0, x_limit), np.random.uniform(0, y_limit)]))
        print(f"Added food source at: {food_sources[-1]}")
    elif event.key == '-':
        if food_sources:
            food_sources.pop()
            print(f"Removed last food source.")


# Initialize boids
boids = initialize_boids(num_boids, x_limit, y_limit, predator_ratio)

# --------------------------Set up the figure and axes----------------------------
fig, ax = plt.subplots()
ax.set_xlim(0, x_limit)
ax.set_ylim(0, y_limit)
repulsion_point, = ax.plot(repulsion_position[0], repulsion_position[1], 'rx', markersize=10, label='Repulsion Point')
goal_point, = ax.plot(goal_position[0], goal_position[1], 'go', markersize=8, label='Goal Point')
obstacle_patches = [Circle(obstacle, obstacle_radius, color='gray') for obstacle in obstacles]
for patch in obstacle_patches:
    ax.add_patch(patch)
food_patches = [Circle(food, food_radius, color='lightgreen') for food in food_sources]
for patch in food_patches:
    ax.add_patch(patch)
ax.legend()

# -----------------------------Create boids------------------------
boid_patches = []
for boid in boids:
    angle = np.arctan2(boid.velocity[1], boid.velocity[0])
    size = boid.size
    tip = boid.position + size * np.array([np.cos(angle), np.sin(angle)])
    left_base = boid.position + size * np.array([np.cos(angle + 2 * np.pi / 3), np.sin(angle + 2 * np.pi / 3)]) * 0.7
    right_base = boid.position + size * np.array([np.cos(angle - 2 * np.pi / 3), np.sin(angle - 2 * np.pi / 3)]) * 0.7
    vertices = np.array([tip, left_base, right_base])
    patch = Polygon(vertices, facecolor='blue', edgecolor='black')
    ax.add_patch(patch)
    boid_patches.append(patch)

# -------------------------- animation-------------------------
ani = animation.FuncAnimation(fig, update_boids, fargs=(boids, boid_patches, goal_point, repulsion_point, obstacle_patches, food_patches), interval=50, blit=True)

fig.canvas.mpl_connect('key_press_event', on_key_press)

plt.title("Boids Simulation")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.show()

print("\nInteractive Controls:")
print("W/S: Increase/Decrease Max Speed")
print("A/D: Increase/Decrease Max Acceleration")
print("1/2: Increase/Decrease Separation Weight")
print("3/4: Increase/Decrease Alignment Weight")
print("5/6: Increase/Decrease Cohesion Weight")
print("7/8: Increase/Decrease Attraction Weight")
print("9/0: Increase/Decrease Repulsion Weight")
print("G/H: Move Goal Position (X-axis)")
print("J/K: Move Goal Position (Y-axis)")
print("R/T: Move Repulsion Position (X-axis)")
print("Y/U: Move Repulsion Position (Y-axis)")
print("N/M: Increase/Decrease Neighborhood Radius")
print(",/.: Increase/Decrease Minimum Distance")
print("B: Toggle Limited Walls (Bouncing)")
print("P: Toggle Predator-Prey mode")
print("V/C: Increase/Decrease Predator Ratio")
print("O/I: Increase/Decrease Predator Perception Radius")
print("L/K: Increase/Decrease Prey Flee Radius")
print(";/' : Increase/Decrease Predator Pursuit Weight")
# print("Z/X: Increase/Decrease Prey Flee Weight")
# print("V: Toggle Obstacle Avoidance")
print("[ : Add Random Obstacle")
print("] : Remove Last Obstacle")
print("Z: Toggle Resting")
print("V: Toggle Feeding")
print("+: Add Random Food Source")
print("-: Remove Last Food Source")