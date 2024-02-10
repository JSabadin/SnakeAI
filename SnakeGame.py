import numpy as np
from matplotlib import pyplot as plt

class SnakeGame:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        self.snake = [(self.width // 2, self.height // 2)]
        self.snake_set = set(self.snake)  # For faster collision checking
        self.direction = (0, -1)  # Initially moving upwards
        self.score = 0
        self.food = None
        self._place_food()
        self.game_over = False

    def _place_food(self):
        while self.food is None or self.food in self.snake:
            self.food = (np.random.randint(0, self.width), np.random.randint(0, self.height))

    def is_collision(self, x, y):
        # Check if the point collides with the snake or the walls
        if x < 0 or x >= self.width or y < 0 or y >= self.height or (x, y) in self.snake_set:
            return True
        return False


    def play_step(self, action):
        # Convert action into a direction
        if action == 0:    # Left
            self.direction = (-1, 0)
        elif action == 1:  # Right
            self.direction = (1, 0)
        elif action == 2:  # Up
            self.direction = (0, -1)
        elif action == 3:  # Down
            self.direction = (0, 1)


        # Move the snake
        next_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])

        # Check for game over conditions
        if self.is_collision(next_head[0], next_head[1]):
            self.game_over = True
            return self.score, self.game_over

        self.snake.insert(0, next_head)
        self.snake_set.add(next_head)

        # Check for food consumption
        if next_head == self.food:
            self.score += 1
            self.food = None
            self._place_food()
        else:
            tail = self.snake.pop()
            self.snake_set.remove(tail)
  
        return self.score, self.game_over

    
    def get_state(self):
        head_x, head_y = self.snake[0]
        point_l = (head_x - 1, head_y)
        point_r = (head_x + 1, head_y)
        point_u = (head_x, head_y - 1)
        point_d = (head_x, head_y + 1)
        
        dir_l = self.direction == (-1, 0)
        dir_r = self.direction == (1, 0)
        dir_u = self.direction == (0, -1)
        dir_d = self.direction == (0, 1)

        state = [
            # Danger straight
            (dir_r and self.is_collision(*point_r)) or 
            (dir_l and self.is_collision(*point_l)) or 
            (dir_u and self.is_collision(*point_u)) or 
            (dir_d and self.is_collision(*point_d)),

            # Danger right
            (dir_u and self.is_collision(*point_r)) or 
            (dir_d and self.is_collision(*point_l)) or 
            (dir_l and self.is_collision(*point_u)) or 
            (dir_r and self.is_collision(*point_d)),

            # Danger left
            (dir_d and self.is_collision(*point_r)) or 
            (dir_u and self.is_collision(*point_l)) or 
            (dir_r and self.is_collision(*point_u)) or 
            (dir_l and self.is_collision(*point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            self.food[0] < head_x,  # food left
            self.food[0] > head_x,  # food right
            self.food[1] < head_y,  # food up
            self.food[1] > head_y,  # food down
        ]

        return np.array(state, dtype=int)

    def plot_state(self):
        # Create a grid initialized to zeros
        grid = np.zeros((self.height, self.width))
        
        # Mark the snake on the grid
        for segment in self.snake:
            grid[segment[1], segment[0]] = 1
        
        # Mark the food
        if self.food:
            grid[self.food[1], self.food[0]] = 2
        
        # Plotting
        plt.imshow(grid, cmap='hot', interpolation='nearest')
        plt.title(f'Score: {self.score}')
        plt.draw()
        plt.pause(0.05)  # Pause to update the plot
        plt.clf()  # Clear the current figure for the next state's plot