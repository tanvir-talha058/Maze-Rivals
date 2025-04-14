import pygame
import random
import numpy as np
import time
from enum import Enum
from collections import deque

# Initialize pygame
pygame.init()

# Constants
CELL_SIZE = 30
GRID_WIDTH = 25
GRID_HEIGHT = 20
SCREEN_WIDTH = CELL_SIZE * GRID_WIDTH
SCREEN_HEIGHT = CELL_SIZE * GRID_HEIGHT + 60  # Extra space for UI
FPS = 30
MAX_DEPTH = 3  # Max depth for min-max search

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)
LIGHT_BLUE = (173, 216, 230)
PINK = (255, 182, 193)

# Cell types
class CellType(Enum):
    EMPTY = 0
    WALL = 1
    PLAYER = 2
    AI = 3
    RESOURCE_LOW = 4
    RESOURCE_MED = 5
    RESOURCE_HIGH = 6
    EXIT = 7

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Maze Rivals")
        self.clock = pygame.time.Clock()
        self.running = True
        self.grid = None
        self.player_pos = None
        self.ai_pos = None
        self.exit_pos = None
        self.resources = []
        self.player_score = 0
        self.ai_score = 0
        self.ai_path = []
        self.ai_eval_states = []  # For visualizing min-max evaluation
        self.game_over = False
        
        # UI elements
        self.font = pygame.font.SysFont('Arial', 24)
        self.small_font = pygame.font.SysFont('Arial', 16)
        
        # Initialize game
        self.generate_maze()
        
    def generate_maze(self):
        """Generate a random maze using a randomized DFS algorithm"""
        # Initialize grid with walls
        self.grid = np.ones((GRID_HEIGHT, GRID_WIDTH), dtype=int)
        
        # Choose random starting point (must be odd coordinates)
        start_x = random.randrange(1, GRID_WIDTH-1, 2)
        start_y = random.randrange(1, GRID_HEIGHT-1, 2)
        self.grid[start_y][start_x] = CellType.EMPTY.value
        
        # Stack for backtracking
        stack = [(start_x, start_y)]
        
        # Directions: right, down, left, up
        directions = [(2, 0), (0, 2), (-2, 0), (0, -2)]
        
        while stack:
            current_x, current_y = stack[-1]
            
            # Find unvisited neighbors
            neighbors = []
            random.shuffle(directions)
            
            for dx, dy in directions:
                nx, ny = current_x + dx, current_y + dy
                if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and self.grid[ny][nx] == CellType.WALL.value:
                    neighbors.append((nx, ny, dx, dy))
            
            if neighbors:
                # Choose a random neighbor
                next_x, next_y, dx, dy = random.choice(neighbors)
                
                # Remove the wall between the current cell and the chosen neighbor
                self.grid[current_y + dy//2][current_x + dx//2] = CellType.EMPTY.value
                
                # Mark the chosen neighbor as visited
                self.grid[next_y][next_x] = CellType.EMPTY.value
                
                # Add the neighbor to the stack
                stack.append((next_x, next_y))
            else:
                # Backtrack
                stack.pop()
        
        # Add some random passages to make the maze less perfect
        for _ in range(GRID_WIDTH * GRID_HEIGHT // 10):
            x = random.randint(1, GRID_WIDTH-2)
            y = random.randint(1, GRID_HEIGHT-2)
            if self.grid[y][x] == CellType.WALL.value:
                self.grid[y][x] = CellType.EMPTY.value
        
        # Place player, AI, resources, and exit
        self.place_entities()
        self.game_over = False
    
    def place_entities(self):
        """Place player, AI, resources, and exit in the maze"""
        empty_cells = [(x, y) for y in range(GRID_HEIGHT) for x in range(GRID_WIDTH) 
                      if self.grid[y][x] == CellType.EMPTY.value]
        
        if not empty_cells:
            return  # No empty cells available
        
        # Place player and AI at opposite corners when possible
        corners = []
        for x, y in empty_cells:
            if (x < GRID_WIDTH // 3 and y < GRID_HEIGHT // 3) or \
               (x < GRID_WIDTH // 3 and y > 2 * GRID_HEIGHT // 3) or \
               (x > 2 * GRID_WIDTH // 3 and y < GRID_HEIGHT // 3) or \
               (x > 2 * GRID_WIDTH // 3 and y > 2 * GRID_HEIGHT // 3):
                corners.append((x, y))
        
        if len(corners) >= 2:
            player_idx = random.randint(0, len(corners) - 1)
            self.player_pos = corners[player_idx]
            corners.pop(player_idx)
            self.ai_pos = corners[random.randint(0, len(corners) - 1)]
        else:
            # Fallback if not enough corner positions
            random.shuffle(empty_cells)
            self.player_pos = empty_cells.pop()
            self.ai_pos = empty_cells.pop() if empty_cells else (1, 1)
        
        # Update grid with player and AI positions
        x, y = self.player_pos
        self.grid[y][x] = CellType.PLAYER.value
        x, y = self.ai_pos
        self.grid[y][x] = CellType.AI.value
        
        # Place exit
        if empty_cells:
            self.exit_pos = empty_cells.pop()
            x, y = self.exit_pos
            self.grid[y][x] = CellType.EXIT.value
        
        # Place resources
        resource_types = [CellType.RESOURCE_LOW, CellType.RESOURCE_MED, CellType.RESOURCE_HIGH]
        resource_counts = [12, 6, 3]  # Number of each resource type
        
        self.resources = []
        for resource_type, count in zip(resource_types, resource_counts):
            for _ in range(count):
                if empty_cells:
                    pos = empty_cells.pop()
                    x, y = pos
                    self.grid[y][x] = resource_type.value
                    self.resources.append((pos, resource_type))
    
    def evaluate_state(self, ai_pos, player_pos, resources, exit_pos):
        """Evaluate the game state for min-max algorithm"""
        score = 0
        
        # Value resources based on their type and distance
        for (rx, ry), resource_type in resources:
            # Calculate distance from AI to resource
            ai_dist = abs(ai_pos[0] - rx) + abs(ai_pos[1] - ry)
            # Calculate distance from player to resource
            player_dist = abs(player_pos[0] - rx) + abs(player_pos[1] - ry)
            
            # Value based on resource type
            if resource_type == CellType.RESOURCE_LOW:
                value = 5
            elif resource_type == CellType.RESOURCE_MED:
                value = 10
            else:  # RESOURCE_HIGH
                value = 20
            
            # Resources that AI can reach faster are more valuable
            if ai_dist < player_dist:
                score += value * (1 + (player_dist - ai_dist) / 10)
            else:
                score -= value * (1 + (ai_dist - player_dist) / 10)
        
        # Value exit point if it exists
        if exit_pos:
            ai_dist_to_exit = abs(ai_pos[0] - exit_pos[0]) + abs(ai_pos[1] - exit_pos[1])
            player_dist_to_exit = abs(player_pos[0] - exit_pos[0]) + abs(player_pos[1] - exit_pos[1])
            
            if ai_dist_to_exit < player_dist_to_exit:
                score += 50 * (1 + (player_dist_to_exit - ai_dist_to_exit) / 10)
            else:
                score -= 50 * (1 + (ai_dist_to_exit - player_dist_to_exit) / 10)
        
        return score
    
    def get_valid_moves(self, position):
        """Get valid moves from a position"""
        x, y = position
        valid_moves = []
        
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            if (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and 
                self.grid[ny][nx] != CellType.WALL.value and
                self.grid[ny][nx] != CellType.PLAYER.value and
                self.grid[ny][nx] != CellType.AI.value):
                valid_moves.append((nx, ny))
        
        return valid_moves
    
    def simulate_move(self, position, new_position, resources, exit_pos):
        """Simulate a move and return updated resources and exit"""
        x, y = new_position
        new_resources = resources.copy()
        new_exit = exit_pos
        
        # Check if moving to a resource
        for i, ((rx, ry), resource_type) in enumerate(resources):
            if (rx, ry) == new_position:
                new_resources.pop(i)
                break
        
        # Check if moving to exit
        if exit_pos == new_position:
            new_exit = None
        
        return new_resources, new_exit
    
    def minimax(self, depth, is_maximizing, ai_pos, player_pos, resources, exit_pos, alpha, beta):
        """Min-max algorithm with alpha-beta pruning"""
        # Store state for visualization
        self.ai_eval_states.append((ai_pos, player_pos, depth, is_maximizing))
        
        # Base case: depth limit reached or game over
        if depth == 0 or not resources and not exit_pos:
            return self.evaluate_state(ai_pos, player_pos, resources, exit_pos), None
        
        if is_maximizing:
            # AI's turn (maximizing)
            max_eval = float('-inf')
            best_move = None
            
            for move in self.get_valid_moves(ai_pos):
                # Simulate the move
                new_resources, new_exit = self.simulate_move(ai_pos, move, resources, exit_pos)
                
                # Recursive call
                eval_score, _ = self.minimax(depth - 1, False, move, player_pos, new_resources, new_exit, alpha, beta)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                # Alpha-beta pruning
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            
            return max_eval, best_move
        else:
            # Player's turn (minimizing)
            min_eval = float('inf')
            best_move = None
            
            for move in self.get_valid_moves(player_pos):
                # Simulate the move
                new_resources, new_exit = self.simulate_move(player_pos, move, resources, exit_pos)
                
                # Recursive call
                eval_score, _ = self.minimax(depth - 1, True, ai_pos, move, new_resources, new_exit, alpha, beta)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                
                # Alpha-beta pruning
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            
            return min_eval, best_move
    
    def ai_move(self):
        """AI decision making and movement using minimax"""
        # Reset visualization data
        self.ai_path = []
        self.ai_eval_states = []
        
        # Use min-max with alpha-beta pruning
        _, best_move = self.minimax(
            MAX_DEPTH, True, self.ai_pos, self.player_pos, 
            self.resources, self.exit_pos, float('-inf'), float('inf')
        )
        
        if best_move:
            # Create a path for visualization
            self.ai_path = [self.ai_pos, best_move]
            
            # Move AI to best position
            x, y = self.ai_pos
            self.grid[y][x] = CellType.EMPTY.value
            
            new_x, new_y = best_move
            cell_type = self.grid[new_y][new_x]
            
            # Collect resource if present
            if cell_type == CellType.RESOURCE_LOW.value:
                self.ai_score += 5
                self.resources = [(pos, res_type) for (pos, res_type) in self.resources 
                                 if pos != best_move]
            elif cell_type == CellType.RESOURCE_MED.value:
                self.ai_score += 10
                self.resources = [(pos, res_type) for (pos, res_type) in self.resources 
                                 if pos != best_move]
            elif cell_type == CellType.RESOURCE_HIGH.value:
                self.ai_score += 20
                self.resources = [(pos, res_type) for (pos, res_type) in self.resources 
                                 if pos != best_move]
            elif cell_type == CellType.EXIT.value:
                self.ai_score += 50
                self.exit_pos = None
            
            # Update AI position
            self.grid[new_y][new_x] = CellType.AI.value
            self.ai_pos = best_move
        else:
            # Fallback if no valid move found
            # Perform random valid move
            valid_moves = self.get_valid_moves(self.ai_pos)
            if valid_moves:
                best_move = random.choice(valid_moves)
                
                # Move AI to random position
                x, y = self.ai_pos
                self.grid[y][x] = CellType.EMPTY.value
                
                new_x, new_y = best_move
                cell_type = self.grid[new_y][new_x]
                
                # Collect resource if present
                if cell_type == CellType.RESOURCE_LOW.value:
                    self.ai_score += 5
                    self.resources = [(pos, res_type) for (pos, res_type) in self.resources 
                                    if pos != best_move]
                elif cell_type == CellType.RESOURCE_MED.value:
                    self.ai_score += 10
                    self.resources = [(pos, res_type) for (pos, res_type) in self.resources 
                                    if pos != best_move]
                elif cell_type == CellType.RESOURCE_HIGH.value:
                    self.ai_score += 20
                    self.resources = [(pos, res_type) for (pos, res_type) in self.resources 
                                    if pos != best_move]
                elif cell_type == CellType.EXIT.value:
                    self.ai_score += 50
                    self.exit_pos = None
                
                # Update AI position
                self.grid[new_y][new_x] = CellType.AI.value
                self.ai_pos = best_move
    
    def handle_player_movement(self, dx, dy):
        """Handle player movement and resource collection"""
        if self.game_over:
            return
            
        x, y = self.player_pos
        new_x, new_y = x + dx, y + dy
        
        # Check if the move is valid
        if (0 <= new_x < GRID_WIDTH and 0 <= new_y < GRID_HEIGHT and 
            self.grid[new_y][new_x] != CellType.WALL.value and
            self.grid[new_y][new_x] != CellType.AI.value):
            
            # Check what's at the new position
            cell_type = self.grid[new_y][new_x]
            
            # Clear current position
            self.grid[y][x] = CellType.EMPTY.value
            
            # Collect resource if present
            if cell_type == CellType.RESOURCE_LOW.value:
                self.player_score += 5
                self.resources = [(pos, res_type) for (pos, res_type) in self.resources 
                                 if pos != (new_x, new_y)]
            elif cell_type == CellType.RESOURCE_MED.value:
                self.player_score += 10
                self.resources = [(pos, res_type) for (pos, res_type) in self.resources 
                                 if pos != (new_x, new_y)]
            elif cell_type == CellType.RESOURCE_HIGH.value:
                self.player_score += 20
                self.resources = [(pos, res_type) for (pos, res_type) in self.resources 
                                 if pos != (new_x, new_y)]
            elif cell_type == CellType.EXIT.value:
                self.player_score += 50
                self.exit_pos = None
            
            # Update player position
            self.grid[new_y][new_x] = CellType.PLAYER.value
            self.player_pos = (new_x, new_y)
            
            # Check if game is over
            if not self.resources and not self.exit_pos:
                self.game_over = True
            else:
                # AI moves after player
                self.ai_move()
                
                # Check if game is over after AI move
                if not self.resources and not self.exit_pos:
                    self.game_over = True
    
    def draw(self):
        """Draw the game state"""
        self.screen.fill(BLACK)
        
        # Draw grid
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                cell_type = self.grid[y][x]
                
                if cell_type == CellType.WALL.value:
                    pygame.draw.rect(self.screen, GRAY, rect)
                elif cell_type == CellType.EMPTY.value:
                    pygame.draw.rect(self.screen, BLACK, rect)
                    pygame.draw.rect(self.screen, DARK_GRAY, rect, 1)
                elif cell_type == CellType.PLAYER.value:
                    pygame.draw.rect(self.screen, BLACK, rect)
                    pygame.draw.rect(self.screen, DARK_GRAY, rect, 1)
                    pygame.draw.circle(self.screen, GREEN, rect.center, CELL_SIZE // 2 - 4)
                elif cell_type == CellType.AI.value:
                    pygame.draw.rect(self.screen, BLACK, rect)
                    pygame.draw.rect(self.screen, DARK_GRAY, rect, 1)
                    pygame.draw.circle(self.screen, RED, rect.center, CELL_SIZE // 2 - 4)
                elif cell_type == CellType.RESOURCE_LOW.value:
                    pygame.draw.rect(self.screen, BLACK, rect)
                    pygame.draw.rect(self.screen, DARK_GRAY, rect, 1)
                    pygame.draw.circle(self.screen, YELLOW, rect.center, CELL_SIZE // 3)
                elif cell_type == CellType.RESOURCE_MED.value:
                    pygame.draw.rect(self.screen, BLACK, rect)
                    pygame.draw.rect(self.screen, DARK_GRAY, rect, 1)
                    pygame.draw.circle(self.screen, ORANGE, rect.center, CELL_SIZE // 3)
                elif cell_type == CellType.RESOURCE_HIGH.value:
                    pygame.draw.rect(self.screen, BLACK, rect)
                    pygame.draw.rect(self.screen, DARK_GRAY, rect, 1)
                    pygame.draw.circle(self.screen, PURPLE, rect.center, CELL_SIZE // 3)
                elif cell_type == CellType.EXIT.value:
                    pygame.draw.rect(self.screen, BLACK, rect)
                    pygame.draw.rect(self.screen, DARK_GRAY, rect, 1)
                    pygame.draw.rect(self.screen, CYAN, 
                                   (x * CELL_SIZE + 5, y * CELL_SIZE + 5, 
                                    CELL_SIZE - 10, CELL_SIZE - 10))
        
        # Draw AI's planned path
        for i, (x, y) in enumerate(self.ai_path):
            if i > 0:  # Skip starting position
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                alpha_surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                alpha_surface.fill((255, 0, 0, 100))  # Semi-transparent red
                self.screen.blit(alpha_surface, rect)
        
        # Draw min-max evaluation states
        alpha = 100  # Transparency level
        for pos, _, depth, is_max in self.ai_eval_states:
            if pos != self.ai_pos:
                x, y = pos
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                color = (255, 192, 203, alpha) if is_max else (173, 216, 230, alpha)  # Pink for max, light blue for min
                alpha_surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                alpha_surface.fill(color)
                self.screen.blit(alpha_surface, rect)
        
        # Draw UI
        # Algorithm information area
        pygame.draw.rect(self.screen, DARK_GRAY, (0, SCREEN_HEIGHT - 60, SCREEN_WIDTH, 60))
        
        # Draw scores
        player_text = self.font.render(f"Player: {self.player_score}", True, GREEN)
        ai_text = self.font.render(f"AI: {self.ai_score}", True, RED)
        self.screen.blit(player_text, (10, 10))
        self.screen.blit(ai_text, (SCREEN_WIDTH - ai_text.get_width() - 10, 10))
        
        # Draw algorithm description
        algo_desc = "(Strategic decision making with ai Made by Tanvir " + str(MAX_DEPTH) + ")"
        algo_text = self.small_font.render(algo_desc, True, WHITE)
        self.screen.blit(algo_text, (SCREEN_WIDTH // 2 - algo_text.get_width() // 2, SCREEN_HEIGHT - 40))
        
        # Draw controls information
        controls_text = self.small_font.render("Use arrow keys to move. Press R to restart.", True, WHITE)
        self.screen.blit(controls_text, (SCREEN_WIDTH // 2 - controls_text.get_width() // 2, SCREEN_HEIGHT - 20))
        
        # Check for game end
        if self.game_over:
            winner_text = None
            if self.player_score > self.ai_score:
                winner_text = self.font.render("Player Wins!", True, GREEN)
            elif self.ai_score > self.player_score:
                winner_text = self.font.render("AI Wins!", True, RED)
            else:
                winner_text = self.font.render("It's a Tie!", True, WHITE)
            
            if winner_text:
                # Draw semi-transparent overlay
                overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT - 60))
                overlay.set_alpha(180)
                overlay.fill(BLACK)
                self.screen.blit(overlay, (0, 0))
                
                text_rect = winner_text.get_rect(center=(SCREEN_WIDTH // 2, (SCREEN_HEIGHT - 60) // 2))
                self.screen.blit(winner_text, text_rect)
                
                restart_text = self.font.render("Press R to restart", True, WHITE)
                restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH // 2, (SCREEN_HEIGHT - 60) // 2 + 40))
                self.screen.blit(restart_text, restart_rect)
        
        pygame.display.flip()
    
    def run(self):
        
        while self.running:
            self.clock.tick(FPS)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                
                if event.type == pygame.KEYDOWN:
                    # Player movement
                    if event.key == pygame.K_UP:
                        self.handle_player_movement(0, -1)
                    elif event.key == pygame.K_RIGHT:
                        self.handle_player_movement(1, 0)
                    elif event.key == pygame.K_DOWN:
                        self.handle_player_movement(0, 1)
                    elif event.key == pygame.K_LEFT:
                        self.handle_player_movement(-1, 0)
                    
                    # Restart 
                    elif event.key == pygame.K_r:
                        self.generate_maze()
                        self.player_score = 0
                        self.ai_score = 0
                        self.ai_path = []
                        self.ai_eval_states = []
                        self.game_over = False
            
            self.draw()
        
        pygame.quit()

if __name__ == "__main__":
    game = Game()
    game.run()