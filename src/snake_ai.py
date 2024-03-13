import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
from conf import C_SPEED

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

class Direction(Enum):
        RIGHT = 1
        LEFT = 2
        UP = 3
        DOWN = 4

Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)

GREEN = (0, 255, 0)
GREEN2 = (100, 255, 0)

RED = (200,0,0)

BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)

BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = C_SPEED

class SnakeP:
    def __init__(self, w, d1, h, d2, player_s=False):
        self.player_s = player_s
        if self.player_s == True:
            self.head = Point(d1, d2)
            print("Player:", self.head)
        else:
            self.head = Point(random.randint(0, (w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE, 
                             random.randint(0, (h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE)
        self.score = 0
        self.snake = [self.head]
        self.direction = Direction.RIGHT
        self.famine = 0
        

class SnakeGameAI:
    def __init__(self, w=640, h=480, snake=None, food_count=1, player=False, cordinates=None, famine=True):
        self.n = food_count
        self.famine_glob = famine
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        self.snakes = []

        #player 
        self.player = player
        self.cordinates = cordinates

        self.reset(snake)


    def reset(self, snake):
        self.snakes = []
        for w, d1, h, d2 in snake:
            self.snakes.append(SnakeP(w, d1, h, d2))

        if self.player:
            self.snakes.append(SnakeP(w, self.cordinates[0], h, self.cordinates[1], True))

        self.food_list = []
        self._place_n_food()
        self.frame_iteration = 0
    
    def _place_n_food(self):
        while len(self.food_list) < self.n:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            food_pt = Point(x, y)
            food_in_snake = any(food_pt in snake.snake for snake in self.snakes)
            if not food_in_snake and food_pt not in self.food_list:
                self.food_list.append(food_pt)

    def get_player_action(self):
        action = [0, 1, 0] 
        reset_g = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
        
              current_direction = self.snakes[-1].direction

              if event.key == pygame.K_RIGHT or event.key == pygame.K_d: 
                if current_direction == Direction.RIGHT:
                    action = [0,1,0]
                elif current_direction == Direction.LEFT:
                    pass
                elif current_direction == Direction.UP:
                    action = [0,0,1]
                elif current_direction == Direction.DOWN:
                    action = [1,0,0]    

              elif event.key == pygame.K_LEFT or event.key == pygame.K_a:   
                if current_direction == Direction.RIGHT:  
                  pass
                elif current_direction == Direction.LEFT:  
                  action = [0,1,0]
                elif current_direction == Direction.UP:  
                  action = [1,0,0]
                elif current_direction == Direction.DOWN:  
                  action= [0,0,1]

              elif event.key == pygame.K_UP or event.key == pygame.K_w:
                if current_direction == Direction.RIGHT:  
                  action= [1,0,0]
                elif current_direction == Direction.LEFT:  
                  action = [0,0,1]
                elif current_direction == Direction.UP:  
                  action = [0,1,0]
                elif current_direction == Direction.DOWN:  
                  pass

              elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                if current_direction == Direction.RIGHT:  
                 action = [0,0,1]
                elif current_direction == Direction.LEFT:  
                  action = [1,0,0]
                elif current_direction == Direction.UP:  
                  pass
                elif current_direction == Direction.DOWN:  
                  action = [0,1,0]
              elif event.key == pygame.K_r:
                  reset_g = True
              else:
                  action = [0,1,0]
        return action, reset_g

    def play_step(self, actions):
        rewards, game_overs, scores = [],[],[]
        if len(actions) != len(self.snakes):
            print("Less actions inserted")
            exit()       
        self.frame_iteration +=1
        for event in pygame.event.get():
           if event.type == pygame.QUIT:
               pygame.quit()
               quit()

            

        for s, action in zip(self.snakes, actions):
            self._move(action, s)
            s.snake.insert(0, s.head)
        
        for s in self.snakes:
            reward = 0
            game_over = False
            s.famine = 100*len(s.snake)
            if self.is_collision(s) or self.frame_iteration > s.famine and self.famine_glob == True:
                print(f"Snake {self.snakes.index(s)}: has died of famine")
                if self.frame_iteration > s.famine and s.player_s == True:
                    print(f"Player {self.snakes.index(s)}: has died of famine")
                game_over = True
                reward = -10
                
                rewards.append(reward) 
                game_overs.append(game_over)
                scores.append(s.score)

            if s.head in self.food_list:
                s.score += 1
                reward = 10
                self.food_list.remove(s.head)

                rewards.append(reward) 
                game_overs.append(game_over)
                scores.append(s.score) 

                self._place_n_food()
            else:
                s.snake.pop()
                rewards.append(0) 
                game_overs.append(False)
                scores.append(s.score) 

            #print(f"Action taken: {actions[0]}, Current direction: {self.snakes[0].direction}")

        self._update_ui()
        self.clock.tick(SPEED)
        #print("Rew:", rewards)
        #print("Ov:", game_overs)
        #print("Sca:", scores)
        
        return rewards, game_overs, scores
    
    #done
    def is_collision(self, snake, pt=None):
        if pt is None:
            pt = snake.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            #if snake.player_s == True:
            #   print(f"Snake {self.snakes.index(snake)} died of wall collision on x:{pt.x} y:{pt.y}")
            return True
        if pt in snake.snake[1:]:
            #if snake.player_s == True:
            #   print(f"Snake {self.snakes.index(snake)} died of self collision on x:{pt.x} y:{pt.y}")
            return True
        for s in self.snakes:
            if s != snake and pt in s.snake or pt in s.head:
                #if snake.player_s == True:
                #    print(f"Snake {self.snakes.index(snake)} died of other snake collision on x:{pt.x} y:{pt.y}")
                return True
        
        return False
    
    #done
    def _move(self, action, snake):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(snake.direction)
        if np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 0, 1]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] #right turn r -> d -> l -> u
        else: # [1,0,0]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] #left turn r -> u -> l -> d
        
        snake.direction = new_dir
        x = snake.head.x
        y = snake.head.y
        if snake.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif snake.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif snake.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif snake.direction == Direction.UP:
            y -= BLOCK_SIZE
        snake.head = Point(x, y)
    
    def _update_ui(self):
        self.display.fill(BLACK)

        #done
        for s in self.snakes:
            if s.player_s == True:
                main_COLOR = GREEN
                additional_COLOR = GREEN2
            else:
                main_COLOR = BLUE1
                additional_COLOR = BLUE2
            for pt in s.snake:
                    pygame.draw.rect(self.display, WHITE, pygame.Rect(s.head.x, s.head.y, BLOCK_SIZE, BLOCK_SIZE))
                    pygame.draw.rect(self.display, additional_COLOR, pygame.Rect(s.head.x+4, s.head.y+4, 12, 12))
                    pygame.draw.rect(self.display, main_COLOR, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                    pygame.draw.rect(self.display, additional_COLOR, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        
        for food in self.food_list:
            pygame.draw.rect(self.display, RED, pygame.Rect(food.x, food.y, BLOCK_SIZE, BLOCK_SIZE))
        

        
        for i, s in enumerate(self.snakes):
            text = font.render("" + str(s.score), True, WHITE)
            self.display.blit(text, [0, (i+1)*20])
            if s.player_s == True and self.famine_glob == True:
                text = font.render("Famine: " + str(self.snakes[-1].famine - self.frame_iteration), True, WHITE)
                self.display.blit(text, [0, self.h - 22])
            

        pygame.display.flip()
