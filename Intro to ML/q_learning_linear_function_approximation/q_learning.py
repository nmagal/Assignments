import sys
from environment import MountainCar
import numpy as np
import random
import copy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class QLearning:
    
    def init_weights(self, game_mode):
        
        if game_mode == "tile":
            weights = np.zeros((2048, 3)) 
        if game_mode == "raw":
            weights = np.zeros((2,3))
        
        bias = 0    
        return(weights,bias) 
    
    #Predicts the best action to take
    def predict_action_and_q_value(self, state_vector, weights, bias, epsilion, choose_random):
        
        #So 
        choice_for_expliotation_vs_exploration = random.random()
        
        #If our random choice is greater then epsilion we exploit
        if choice_for_expliotation_vs_exploration >= epsilion or choose_random == False:
            
            #Must calculate for all different actions and choose the highest one 
            action_values = []     
            for action in range(3):
                
                action_weight_vector = weights[:, action]
                predicted_q_value = np.dot(state_vector, action_weight_vector) + bias 
                action_values.append(predicted_q_value)
            
            #Finding the max value 
            max_predicted_q_value, optimal_predicted_action = action_values[0], 0
            
            if action_values[1] > max_predicted_q_value:
                
                max_predicted_q_value = action_values[1]
                optimal_predicted_action = 1
            
            if action_values[2] > max_predicted_q_value: 
                
                max_predicted_q_value = action_values[2]
                optimal_predicted_action = 2 
            
        #Otherwise we explore
        if choice_for_expliotation_vs_exploration < epsilion and choose_random == True:
            
            #Our random action
            optimal_predicted_action = random.randint(0,2)
            
            #Prediciting q value 
            action_weight_vector = weights[:, optimal_predicted_action]
            max_predicted_q_value = np.dot(state_vector, action_weight_vector) + bias
     
        
        return max_predicted_q_value, optimal_predicted_action
    
    #Trains the bias and the weights given our world is represented rawly
    def train_raw(self, episodes, max_iterations, weight_matrix, learning_rate, epsilion, env, bias, gamma):
        weight_matrix = copy.deepcopy(weight_matrix)
        rewards_per_episode = []
        
        for episode in range(episodes): 
            
            #Eventually we will write out the total amount of reward per episode
            total_reward = 0
            
            #Must start off initilizing the q value
            initial_state = env.reset()
            initial_state = np.array(list(initial_state.values()))
                
            #Getting the current q value and the action
            q_value, current_action = self.predict_action_and_q_value(initial_state, 
                                                               weight_matrix,
                                                               bias,
                                                               epsilion, True)
            
            gradient_wr_weight = initial_state

            #Must incorporate when the car is at the finished state
            for iteration in range(max_iterations):
                
                #Getting the info needed to find our next state
                s_prime, reward, complete_state = env.step(current_action)
                total_reward = total_reward + reward 
                
                if complete_state == True:
                    break
            
                #Getting our new state in nice format 
                s_prime = np.array(list(s_prime.values()))
                
                #Now finding our q prime
                q_prime, a_prime = self.predict_action_and_q_value(s_prime, 
                                                                   weight_matrix,
                                                                   bias,
                                                                   epsilion, False)
                                
                #Inner portion
                inner_portion = (q_value - (reward + gamma*(q_prime)))
                weight_matrix[:, current_action] = weight_matrix[:, current_action] - (learning_rate * inner_portion  *  gradient_wr_weight)
                
                #Updating Bias
                bias = bias - learning_rate * (q_value - (reward + gamma*(q_prime)))
                
                #Setting our q
                q_value, current_action = self.predict_action_and_q_value(s_prime, weight_matrix, bias, epsilion, True)
                #what about our reward
                gradient_wr_weight = s_prime


                
            rewards_per_episode.append(total_reward)
            print(total_reward)

        return(weight_matrix, bias, rewards_per_episode)
    

    #Trains the bias and the weights given our world is represented rawly
    def train_tile(self, episodes, max_iterations, weight_matrix, learning_rate, epsilion, env, bias, gamma):
        weight_matrix = copy.deepcopy(weight_matrix)
        rewards_per_episode = []
        
        for episode in range(episodes): 
            
            #Eventually we will write out the total amount of reward per episode
            total_reward = 0
            
            #Must start off initilizing the q value
            initial_state = env.reset()
            initial_state = self.one_hot_tile(initial_state)
            
                
            #Getting the current q value and the action
            q_value, current_action = self.predict_action_and_q_value(initial_state, 
                                                               weight_matrix,
                                                               bias,
                                                               epsilion, True)
            
            gradient_wr_weight = initial_state

            #Must incorporate when the car is at the finished state
            for iteration in range(max_iterations):
                
                #Getting the info needed to find our next state
                s_prime, reward, complete_state = env.step(current_action)
                total_reward = total_reward + reward 
            
                #Getting our new state in nice format
                s_prime = self.one_hot_tile(s_prime)
                
                #Now finding our q prime
                q_prime, a_prime = self.predict_action_and_q_value(s_prime, 
                                                                   weight_matrix,
                                                                   bias,
                                                                   epsilion, False)

                
                #Updating Weight Matrix
                weight_matrix[:, current_action] = weight_matrix[:, current_action] - (learning_rate * (q_value - (reward + gamma*(q_prime))) * gradient_wr_weight)
                
                #Updating Bias
                bias = bias - learning_rate * (q_value - (reward + gamma*(q_prime)))
                
                if complete_state == True:
                    break
                #Setting our q
                q_value, current_action = self.predict_action_and_q_value(s_prime, weight_matrix, bias, epsilion, True) 

                gradient_wr_weight = s_prime
                env.render()
                
                

            
            rewards_per_episode.append(total_reward)

            print(total_reward)

        return(weight_matrix, bias, rewards_per_episode)
        
    #Creates a one hot encoded vector of our current state     
    def one_hot_tile(self, given_states):
        
        one_hot_encoded = {}
        #Create a dictionary with all 0s
        for key in range(2048):
            one_hot_encoded.update({key : 0})
        
        #Now let's merge our two together!
        one_hot_encoded.update(given_states)
        
        one_hot_encoded =np.array(list(one_hot_encoded.values()))
        
        return one_hot_encoded

#%%

user_input = sys.argv
q_learning  = QLearning()
env = MountainCar(mode = user_input[1])
print(env.state)

#Initilizing our game base on the world we are living in
if user_input[1] == "raw":
    

    initial_weights, bias = q_learning.init_weights("raw")
    
    trained_weights, bias, rewards = q_learning.train_raw(int(user_input[4]), 
                                       int(user_input[5]), 
                                       initial_weights,
                                       float(user_input[8]),
                                       float(user_input[6]),
                                       env, bias,
                                       float(user_input[7]))
    
        #Outputting our results
    with open(user_input[3], 'w') as f:
        for reward in range(len(rewards)):
            f.write(str(rewards[reward]) + "\n")
            
        #Now outputting our weights and biases 
    with open(user_input[2], "w") as f: 
        f.write(str(bias)+"\n")
        
        for row in trained_weights:
            for entry in row:
                f.write(str(entry) + "\n")

if user_input[1] == "tile":
    
    #Training our model
    initial_weights, bias = q_learning.init_weights("tile")
    trained_weights, bias, rewards = q_learning.train_tile(int(user_input[4]), 
                                   int(user_input[5]), 
                                   initial_weights,
                                   float(user_input[8]),
                                   float(user_input[6]),
                                   env, bias,
                                   float(user_input[7]))
    
    #Outputting our results starting with rewards
    with open(user_input[3], 'w') as f:
        for reward in range(len(rewards)):
            f.write(str(rewards[reward]) + "\n")
    
    #Now outputting our weights and biases 
    with open(user_input[2], "w") as f: 
        f.write(str(bias)+"\n")
        
        for row in trained_weights:
            for entry in row:
                f.write(str(entry) + "\n")

#%% Plotting this data

#Convert so we can use rolling method from pandas 
rewards_series = pd.Series(rewards)
rolling_rewards_25 = rewards_series.rolling(25).mean()
index_of_rewards = np.arange(len(rewards))

    
#plt.plot(index_of_rewards, rewards, label = "Rewards per episode")
sns.lineplot(data = rewards_series, label = "Rewards Per Episode")
sns.lineplot(data = rolling_rewards_25, label = "Rolling Mean")
plt.xlabel("Number of episodes")
plt.ylabel("Total Reward")


    
