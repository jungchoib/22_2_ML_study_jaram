from torch.optim import SGD #Default optimizer. you may use other optimizers
from torch.utils.data import DataLoader
import argparse
import torch
import torch.nn as nn
import gym
from utils import get_explore_rate, select_action
from model import Q_net

def simulate(model, args): #model: the neural network
    #optimizer and loss for the neural network
    optimizer = SGD(model.parameters(), lr = 0.01)
    criterion = nn.MSELoss()
    ## Instantiating the learning related parameters
    explore_rate = get_explore_rate(0, args.decay_constant, args.min_explore_rate)

    memory = list()
    num_streaks = 0
    for episode in range(args.num_episodes):
        # Reset the environment
        state_0 = env.reset()
        
        for t in range(args.max_timestep):
            # env.render()#you may want to comment this line out, to run code silently without rendering
            
            # Selecting an action. the action MUST be choosed from the neural network's output.
            with torch.no_grad():
                actiontable = model(torch.Tensor(state_0).unsqueeze(0))
                action = select_action(actiontable.squeeze(0), explore_rate, env)

            # Execute the action then collect outputs
            state, reward, done, _ = env.step(action)

            #Update the memory
            '''
                ####################################################
                TODO:Implement your memory
                ####################################################
            '''
            
            # Update the Q-net parameters
            replay(model, memory, args, criterion, optimizer, iteration = 1)
            
            state_0 = state

            #done: the cart failed to maintain balance
            if done == True:
                break

        # Update parameters
        explore_rate = get_explore_rate(episode, args.decay_constant, args.min_explore_rate)
        
        print("Episode %d finished after %f time steps. Streak: %d" % (episode, t, num_streaks))

        #The episode is considered as a success if timestep >SOLVED_TIMESTEP 
        if (t >= args.solved_timestep):
            num_streaks += 1
        else:
            num_streaks = 0
            
        #  when the agent 'solves' the environment: steak over 120 times consecutively
        if num_streaks > args.streak_to_end:
            print("The Environment is solved")
            torch.save(model.state_dict(), 'modelparam.pt')
            break

    env.close()#closes window

def replay(model, memory, args, criterion, optimizer, iteration = 1):
    if len(memory) < args.batchsize:
        return None
    d_loader = DataLoader(memory, args.batchsize ,shuffle = True, drop_last= True)

    
    for i, batch in enumerate(d_loader):
        '''
            ####################################################
            TODO: Implement your replay function.
            ####################################################
        '''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model!')
    parser.add_argument('--num_episodes', type = int, default= 10000)
    parser.add_argument('--max_timestep', type = int, default= 250)
    parser.add_argument('--solved_timestep', type = int, default= 199)
    parser.add_argument('--streak_to_end', type = int, default= 120)
    parser.add_argument('--batchsize', type = int, default= 64)
    parser.add_argument('--min_explore_rate', type = float, default= 0.01)
    parser.add_argument('--discount_factor', type = float, default= 0.99)
    parser.add_argument('--decay_constant', type = int, default= 25)
    parser.add_argument('--max_memory', type = int, default = 1000)
    train_args = parser.parse_args()

    env = gym.make('CartPole-v1')
    qnet = Q_net()
    simulate(qnet, train_args)