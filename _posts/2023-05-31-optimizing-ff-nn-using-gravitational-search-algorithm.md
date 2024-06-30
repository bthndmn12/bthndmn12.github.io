---
layout: post
title:  "Optimizing FF-NN using Gravitational Search Algorithm"
date:   2023-05-31 12:45:00 +0100
categories: jekyll update
author: 'Mehmet Batuhan Duman'
---

> _In the context of neural networks, this article gives a thorough analysis and use of the Gravitational Search Algorithm (GSA). The GSA is an optimization method that borrows principles from nature and imitates the effects of mass and gravity. To maximize the weights of the network, the technique is combined with a feed-forward neural network. A Python implementation of the GSA for optimizing the weights of a neural network with a single hidden layer is shown in the provided code. Using matplotlib, the evaluation of the algorithm’s performance is done over a number of epochs. Scholarly studies exploring the use of GSA in neural networks and its benefits over conventional optimization techniques are among the sources considered for this study._

#### **INTRODUCTION**

The Gravitational Search Algorithm (GSA) is a nature-inspired optimization algorithm that mimics the law of gravity and mass interactions. In 2009, Rashedi, Nezamabadi-pour, and Saryazdi first presented it. The GSA has been applied to a variety of domains, including machine learning, to improve the weights of neural networks. In order to optimize the weights of a neural network with a single hidden layer, the GSA is presented in this article with a Python implementation.

#### PROBLEM DEFINITION

The primary challenge in training neural networks is the optimization of the weights. Traditional optimization methods, such as gradient descent, can often get stuck in local minima, leading to suboptimal performance. The GSA, being a global optimization algorithm, can potentially overcome this issue by exploring the solution space more thoroughly. The problem at hand is to implement the GSA in Python and use it to optimize the weights of a neural network.

#### LITERATURE ANALYSIS

Several studies have been conducted on the application of the GSA in neural networks. For instance, a study by Fedorovici et al. (2012) presented a method of embedding GSA in CNNs for OCR systems. The study demonstrated that the GSA, in combination with the Back Propagation BP algorithm, could improve performance by avoiding the algorithms’ traps in local minima. Another study by George and Huerta (2018) introduced Deep Filtering, a method for end-to-end time-series signal processing, based on a system of two deep convolutional neural networks, designed for classification and regression to rapidly detect and estimate parameters of signals in highly noisy time-series data streams. The study showcased the application of this method for the detection and parameter estimation of gravitational waves from binary black hole mergers.

#### METHODS AND TECHNOLOGIES TO BE USED

The method used in this study is the Gravitational Search Algorithm, a nature-inspired optimization algorithm. The GSA is implemented in Python and used to optimize the weights of a neural network with a single hidden layer. The neural network is a simple feed-forward network implemented using the numpy library. Since the GSA is a global optimization technique, it might locate the global minimum of the error function rather than a local minimum. Given that the error function of a neural network is frequently non-convex and may have numerous local minima, this makes it an excellent option for optimizing the weights of a neural network. The literature has looked into the application of GSA for neural network weight optimization. For instance, a way of embedding GSA in CNNs for OCR systems was provided in a study by Fedorovici et al. (2012). The study showed that by avoiding the algorithms’ traps in local minima, the GSA and the Back Propagation method might increase performance. Deep Filtering is a technique for end-to-end time-series signal processing developed by George and Huerta (2018). It is based on a system of two deep convolutional neural networks and was created for classification and regression in order to quickly identify and estimate the parameters of signals in time-series data streams with high levels of noise. This method’s use for the detection and parameter determination of gravitational waves from binary black hole mergers was demonstrated in the study.

#### STUDIES

The neural network is a simple feed-forward network with a single hidden layer. The weights of the network are initialized randomly, and the GSA is used to optimize these weights over a series of epochs. The performance of the algorithm is evaluated by calculating the error between the network’s output and the desired output.

The parameters for the GSA and the neural network are set. This includes the number of agents , the learning rate, the number of iterations , the number of input features, and the number of neurons in the hidden layer. The agents, which represent the weights of the neural network, are initialized randomly. Each agent is a vector of weights, and the size of this vector is determined by the number of input features and the number of neurons in the hidden layer. The input data and the desired output are set. In this case, the input data is a vector of 0.2s, and the goal prediction is 0.9. A function is defined to calculate the fitness of an agent.

The fitness is calculated as the sum of the squared differences between the output of the network (using the agent’s weights) and the goal prediction.
```python
 def calculate_fitness(agent):  
    # Reshape the agent's weights to match the network's structure  
    weights = agent.reshape((n_inputs, n_hidden))  
    biases = np.zeros(n_hidden)  # You could also optimize the biases  
  
    # Forward pass through the network  
    hidden_layer = np.maximum(0, np.dot(input_data, weights) + biases)  
    output_layer = np.dot(hidden_layer, output_weights)  
  
    # Calculate error  
    error = np.sum((output_layer - goal_pred) ** 2)  
    return error
```
A function is defined to calculate the gravitational forces between agents. The force between two agents is proportional to the product of their masses and inversely proportional to the distance between them. The direction of the force is from the agent with lesser mass to the agent with greater mass.

```python
 def calculate_forces(i):  
    forces_i = np.zeros(agents.shape[1])  
    for j in range(n_agents):  
        if i != j:  
            # Distance between agents  
            distance = np.abs(agents[i] - agents[j])  
  
            # Gravitational force  
            force = (mass[i] * mass[j]) / (distance + 1e-5)  # Add a small constant to avoid division by zero  
  
            # Direction  
            direction = (agents[j] - agents[i]) / (distance + 1e-5)  # Add a small constant to avoid division by zero  
  
            # Update force  
            forces_i += force * direction  
    return forces_i
```

The algorithm starts by initializing a population of agents. Each agent represents a potential solution to the optimization problem. In the context of neural networks, each agent is a vector of weights. The fitness of each agent is evaluated. The fitness function depends on the specific problem being solved. In the context of neural networks, the fitness function could be the error between the network’s output (using the agent’s weights) and the desired output.   
The mass of each agent is calculated based on its fitness. Agents with better fitness have higher mass. The mass of an agent i at iteration t is calculated as follows:

![](https://cdn-images-1.medium.com/max/800/1*jAiwNw4opJR3Amif_a-R8w.png)

The gravitational force between two agents is calculated. The force between agent i and agent j at iteration t is calculated as follows:

![](https://cdn-images-1.medium.com/max/800/1*UmTbmedzG4Iu6-j3dnb1aQ.png)

The positions of the agents are updated based on the forces. The new position of agent i at iteration t+1 is calculated as follows:

![](https://cdn-images-1.medium.com/max/800/1*E9Yix07y3H248_kP7on6Dg.png)

The weights of the neural network, which are represented by the agents in the GSA, are updated in each epoch based on the calculated forces and a random velocity. The formula used in the code to update the weights is as follows:

![](https://cdn-images-1.medium.com/max/800/1*ttGVAKfX0CiBV7AJEQ8-jA.png)


In this example parameters are in the below

```python
# GSA parameters  
n_agents = 100  # Number of agents (solutions)  
# lr = 0.001  # Learning rate  
lr = np.float32(0.001)   
  
epochs = 1000  # Number of iterations  
n_inputs = 20  # The number of input features  
n_hidden = 10  # The number of neurons in the hidden layer  
  
# Initialize agents (weights)  
# agents = np.random.uniform(low=-1, high=1, size=(n_agents, n_inputs * n_hidden))  
agents = np.random.uniform(low=-1, high=1, size=(n_agents, n_inputs * n_hidden)).astype(np.float32)  
  
# Training data  
# goal_pred = 0.9  
goal_pred = np.float32(0.9)  
# input_data = np.full(n_inputs, 0.2)  
input_data = np.full(n_inputs, 0.2, dtype=np.float32)
```


#### REFERENCES

1. Fedorovici, L. O., Precup, R. E., Dragan, F., David, R. C., & Purcaru, C. (2012). Embedding Gravitational SearchAlgorithms in Convolutional Neural Networks for OCR applications. 2012 7th IEEE International Symposium on Applied Computational Intelligence and Informatics (SACI). ([https://ieeexplore.ieee.org/document/6249989](https://ieeexplore.ieee.org/document/6249989))
2. 2. George, D. J., & Huerta, E. A. (2018). Deep neural networks to enable real-time multimessenger astrophysics. Physical Review D, 97(4)([https://doi.org/10.1103/physrevd.97.044039](https://doi.org/10.1103/physrevd.97.044039))  
3. Poma, Y., Melin, P., González, C. M., & Martínez, G. (2019). Optimization of Convolutional Neural Networks Using the Fuzzy Gravitational Search Algorithm. Journal of Applied Mathematics and Robotics. ([https://doi.org/10.14313/jamris/1-2020/12](https://doi.org/10.14313/jamris/1-2020/12))  
4. Wang, J., & Han, S. (2015). Feed-Forward Neural Network Soft-Sensor Modeling of Flotation Process Based on Particle Swarm Optimization and Gravitational Search Algorithm. Mathematical Problems in Engineering, 2015. ([https://doi.org/10.1155/2015/147843](https://doi.org/10.1155/2015/147843))  
5. Nagra, A. A., Alyas, T., Abdul Hamid, M. A., Tabassum, N., & Ahmad, A. (2022). Training a Feedforward Neural Network Using Hybrid Gravitational Search Algorithm with Dynamic Multiswarm Particle Swarm Optimization. Complexity, 2022.([https://doi.org/10.1155/2022/2636515](https://doi.org/10.1155/2022/2636515))  
6. Ezzat, D., Hassanien, A. E., & Ella, H. A. (2021). An optimized deep learning architecture for the diagnosis of COVID-19 disease based on gravitational search optimization. Applied Soft Computing, 98. ([https://doi.org/10.1016/j.asoc.2020.106742](https://doi.org/10.1016/j.asoc.2020.106742))