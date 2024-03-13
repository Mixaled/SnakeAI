import torch
import random
import numpy as np
from collections import deque
from snake_ai import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer, device
import argparse
from conf import C_MAX_MEMORY,C_BATCH_SIZE,C_EPSILON,C_GAMMA,C_LR


MAX_MEMORY = C_MAX_MEMORY
BATCH_SIZE = C_BATCH_SIZE
LR = C_LR
EPSILON = C_EPSILON
GAMMA = C_GAMMA



class Agent:

    def __init__(self, pl_save=None, index=None):
        self.index = index
        self.n_games = 0
        self.epsilon = EPSILON
        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3).to(device)
        self.exploration = True
        if pl_save != None:
            self.exploration = False
            self.model.load(pl_save)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
    def find_food(self, snake, food_list):
        pos_y, pos_x = snake.head
        lst = food_list
        results = []
        for i in lst:
            diff = pos_y - i[0]
            diff2 = pos_x - i[1]
            dist= abs(diff)+abs(diff2)
            results.append([dist, i[0], i[1]])
        return min(results)


    def get_state(self, game, snake): 
        head = snake.snake[0]
        #print("Head:", head)
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = snake.direction == Direction.LEFT
        dir_r = snake.direction == Direction.RIGHT
        dir_u = snake.direction == Direction.UP
        dir_d = snake.direction == Direction.DOWN

        nearest_index = 0
        info = self.find_food(snake, game.food_list)
        for i, food in enumerate(game.food_list):
            if food.x == info[1] and food.y == info[2]:
                nearest_index = i
        state = [   
            (dir_r and game.is_collision(snake, point_r)) or 
            (dir_l and game.is_collision(snake, point_l)) or 
            (dir_u and game.is_collision(snake, point_u)) or 
            (dir_d and game.is_collision(snake, point_d)),

            (dir_u and game.is_collision(snake, point_r)) or 
            (dir_d and game.is_collision(snake, point_l)) or 
            (dir_l and game.is_collision(snake, point_u)) or 
            (dir_r and game.is_collision(snake, point_d)),

            (dir_d and game.is_collision(snake, point_r)) or 
            (dir_u and game.is_collision(snake, point_l)) or 
            (dir_r and game.is_collision(snake, point_u)) or 
            (dir_l and game.is_collision(snake, point_d)),
            
            dir_l,
            dir_r,
            dir_u,
            dir_d,           
            game.food_list[nearest_index].x < snake.head.x,  # food left
            game.food_list[nearest_index].x > snake.head.x,  # food right
            game.food_list[nearest_index].y < snake.head.y,  # food up
            game.food_list[nearest_index].y > snake.head.y  # food down
            ]
        #print("State:")
        #print(np.array(state, dtype=int))
        #print("end of state")

        return np.array(state, dtype=int)


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        #print("Remembered:", state, action, reward, next_state, done)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        #print("Training long memory with mini sample:", mini_sample)
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
        #print("Training short memory with:", state, action, reward, next_state, done)
    
    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon and self.exploration == True:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(device)
            prediction = self.model(state0)
            #rint(f"Predicted Q-values: {prediction}")
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
    
    def load_snake(self):
        try:
            self.model.load()
        except:
            pass

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def train(h, w, snk_count, food_count, player_=False, cordinates_=None, 
          optimization=False, famine_c=True, load_model="", save_name="", winnable=False):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    if player_:
        counter = 1
    else:
        counter = 0
    agent =[Agent(load_model, i) for i in range(0, snk_count+counter)] 
    game = SnakeGameAI(h,w,[(h, 0, w, 0) for _ in range(snk_count)], food_count, 
                       player=player_, cordinates=cordinates_, famine=famine_c)
    
    while True:

        actions = []
        states_old = []
        final_moves = []
        reset_status = False
        
        for i, snake in enumerate(game.snakes):
            #print(i)
            states_old.append(agent[i].get_state(game, snake))
            final_moves.append(agent[i].get_action(states_old[i])) 
            actions.append(final_moves[i])

        if player_== True: # player handler
            player_action, reset_status = game.get_player_action()
            if reset_status == True:
                #print("whatever")
                game.reset([(h, 0, w, 0) for _ in range(snk_count)])
            actions[-1] = player_action
        
        if reset_status == False:
            states_new = []
            reward, done, score = game.play_step(actions)
            for i, snake in enumerate(game.snakes):

                states_new.append(agent[i].get_state(game, snake))

                #print("info:",states_old[i], final_moves[i], reward[i], states_new[i], done[i])

                agent[i].train_short_memory(states_old[i], final_moves[i], reward[i], states_new[i], done[i])
                agent[i].remember(states_old[i], final_moves[i], reward[i], states_new[i], done[i])

                if len(game.snakes) == 1 and done[i] == True and optimization == True:
                   
                    #if game.snakes[i].player_s == True:
                    #    print(f"Removed player: {i}")
                    #print(f"Removed snake: {i}")
                    game.snakes.remove(game.snakes[i])
                    

                    agent[i].n_games += 1
                    agent[i].train_long_memory()
                    if score[i] > record:
                        record = score[i]
                        agent[i].model.save()
                    #print('Game', agent[i].n_games, 'Score', score[i], 'Record:', record)
                elif optimization == False and done[i] == True:
                    #print(f"Removed snake: {i}")
                    game.snakes.remove(game.snakes[i])

                    agent[i].n_games += 1
                    agent[i].train_long_memory()
                    if score[i] > record:
                        record = score[i]
                        agent[i].model.save(save_name)
                    #print('Game', agent[i].n_games, 'Score', score[i], 'Record:', record)
                elif done[i] == True:
                    #game.reset([(h, 2, w, 2)])\
                    #print("Done status:", done[i])
                    #print("Final move:", final_moves[i])
                    #print("len: ", len(game.snakes))
                    #if game.snakes[i].player_s == True:
                    #    print(f"Removed player: {i}")
                    #print(f"Removed snake: {i}")

                    game.snakes.remove(game.snakes[i])   
                if len(game.snakes) == 1 and game.snakes[0].player_s == True:
                    exit(0)
                if len(game.snakes) == 0:
                    game.reset([(h, 0, w, 0) for _ in range(snk_count)])
                    if optimization == True:
                        for i in range(snk_count):
                            agent[i].load_snake()
                
                    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Snake Game Training Program')
    parser.add_argument('--player', type=str2bool, help='Specify if player is present')
    parser.add_argument('--cordinates', type=int, nargs='+', help='Specify the coordinates')
    parser.add_argument('--optimization', type=str2bool, help='Specify if optimization is enabled')
    parser.add_argument('--famine', type=str2bool, help='Specify if famine is enabled')
    parser.add_argument('--load_model', type=str, help='Specify the model to load')
    parser.add_argument('--save_name', type=str, help='Specify the name to save the model')
    parser.add_argument('--h', type=int, help='Height of the screen')
    parser.add_argument('--w', type=int, help='Width of the screen')
    parser.add_argument('--snake_count', type=int, help='Number of snakes')
    parser.add_argument('--food_count', type=int, help='Number of food items')
    parser.add_argument('--win', type=str2bool, help='Winnable game')

    args = parser.parse_args()
    train(args.h, args.w, args.snake_count, args.food_count, 
          player_=args.player, cordinates_=args.cordinates, 
          optimization=args.optimization, famine_c=args.famine, 
          load_model=args.load_model, save_name=args.save_name, winnable=args.win)