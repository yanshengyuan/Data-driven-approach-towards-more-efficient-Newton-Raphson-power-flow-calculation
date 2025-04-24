import pandapower as pp
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from SimpleTwoBus import SimpleTwoBus
import matplotlib.pyplot as plt

class GridEnv(gym.Env):
    def __init__(self,V_ext = 1.0, G = 100, B = 10, k_limit = 3, max_iteration=50, termination_counter=10):


        self.observation_space = spaces.Box(low = np.array([0.5,-90, 0]), high = np.array([2, 90, max_iteration+1]), dtype=np.float32) #[V_init, theta_init, number_iterations]
        
        self.action_space = spaces.Box(low=np.array([-0.5, -50]), high=np.array([0.5, 50]), dtype=np.float32)

        self.k_limit = k_limit
        self.termination_counter = termination_counter
        self.max_iteration = max_iteration


        self.G = G
        self.B = B
        self.V_ext = V_ext

        #initialize network
        self.state, info = self.reset()

    def create_feasible_Ybusnet(self):

        YbusNet = SimpleTwoBus(self.V_ext,self.P,self.Q,self.G,self.B,self.V_bus1,self.theta_bus1, 0.98, 0.5) #just to create a sparse Ybus matrix
        net = YbusNet.net

        return net


    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None):
        if seed is not None:
            self.seed(seed)

        self.counter = 0
        self.done = False
        self.terminated = False
        self.state = np.zeros(3)

        self.P = 0.9 #np.random.uniform(low= 0, high=10)
        self.Q = 0.6 #np.random.uniform(low = 0, high =10)

        self.V_bus1 = 1.0#np.random.uniform(low = 0.85, high = 1.15, size=1)
        self.theta_bus1 = 0#np.random.uniform(low = -45, high = 45, size=1)
        self.V = np.random.uniform(low = 0.5, high = 2, size=1) # initial guess
        self.theta = np.random.uniform(low = -90, high = 90, size=1) # initial guess


        Net = SimpleTwoBus(self.V_ext,self.P,self.Q,self.G,self.B,self.V,self.theta, self.V_bus1, self.theta_bus1)
        self.net = Net.net

        self.Ybus = self.calculate_Ybus()

        iterations = self.perform_NR_step()

        
        self.update_state(iterations)


        return self.state, {}




    def calculate_Ybus(self):


        Ybusnet = self.create_feasible_Ybusnet()
        pp.runpp(Ybusnet, max_iteration = self.max_iteration, tolerance_mva=1e-5)
        Ybus = Ybusnet._ppc["internal"]["Ybus"]



        return Ybus


    def calculate_complex_V(self, V, theta):
        complex_V = V*np.exp(1j*theta) #rectangular form

        return complex_V
    
    def update_V(self, action):


        new_V = self.V - action[0]
        new_theta = self.theta - action[1]


        # maybe try different way of scaling the actions back when they exceed the limits?
        #defines the action constraints -> this might not be the correct way to do this!
        if new_V[0] > 2:
            new_V[0] = 2 
        if new_V[0] < 0.5:
            new_V[0] = 0.5
        if new_theta[0] > 90:
            new_theta[0] = 90
        if new_theta[0] < -90:
            new_theta[0] = -90
        


        complete_V = np.zeros(2)
        complete_theta = np.zeros(2)

        complete_V[0] = self.V_bus1
        complete_V[1] = new_V[0]
        complete_theta[0] = self.theta_bus1
        complete_theta[1] = new_theta[0]


        self.complex_V = self.calculate_complex_V(complete_V, complete_theta)



        self.V = new_V

        self.theta = new_theta





    def calculate_residual(self, action):

        # net = self.net.deepcopy()  # Keep the network unchanged

        self.update_V(action) #rectangular form
        # print(f"{self.complex_V=}")

        # print(f"{self.Ybus[0,1]=}")

        term2 = self.Ybus@self.complex_V
        term2_complex_conj = np.conj(term2)

        term1 = self.complex_V[1:]@term2_complex_conj[1:] #without the external bus!
        # term1 = self.complex_V@term2_complex_conj

        F = self.P + 1j*self.Q - term1

        delta_P = np.real(F)
        delta_Q = np.imag(F)

        residual = np.array([delta_P, delta_Q])



        return residual


    def perform_NR_step(self):

        net = self.net.deepcopy()  # Keep the network unchanged

        try:
            pp.runpp(net, max_iteration = self.max_iteration, tolerance_mva = 1e-5, init_vm_pu=self.V,init_va_degree=self.theta)
            # print(f"{net.res_bus[['va_degree']].values=}")
            # print(f"{net.res_bus[['vm_pu']].values=}")
        

            iterations = net._ppc["iterations"]
        except:
            iterations = 50
            # print(f"{net.res_bus[['va_degree']].values=}")
            # print(f"{net.res_bus[['vm_pu']].values=}")

        return iterations
        

    

    def calculate_reward(self, residual):


        reward = np.linalg.norm(residual)

        return -reward
    

    def update_state(self, iterations):
        
        self.state[0] = self.V[0]
        self.state[1] = self.theta[0]
        self.state[2] = iterations


    def step(self, action):


        self.counter += 1
        # action = [delta_V, delta_theta]

        # perform action
        residual = self.calculate_residual(action)
        # print(f"{residual=}")

        # calcualate reward
        reward = self.calculate_reward(residual)

        iterations = self.perform_NR_step()

        #update state:
        self.update_state(iterations)

        if iterations <= self.k_limit:
            self.done = True
            return self.state, reward, self.done, self.terminated, {}

        

        if self.counter == self.termination_counter:
            self.terminated = True
            return self.state, reward, self.done, self.terminated, {}

        return self.state, reward, self.done, self.terminated, {}


  

    def render(self):
        pass




if __name__=="__main__":
    # Test run


    env = GridEnv()

    k_list = []
    for i in range(int(100)):

        state, info = env.reset()
        k = state[-1]
        k_list.append(k)


    #view the distribution of hard and easy cases
    plt.figure()
    plt.hist(k_list)
    plt.xlabel("k")
    plt.ylabel("counts")
    plt.show()

    state,info = env.reset()

    print("Initial State:", state)
    # env.render()

    # Define a sample action within the specified ranges
    action = np.array([0.03, 15.0], dtype=np.float32)

    # Take a step in the environment using the sample action
    next_state, reward, done, terminated, info = env.step(action)

    # Print the results
    print("\nAction Taken:", action)
    print("Next State:", next_state)
    # env.render()
    print("Reward:", reward)
    print("Done:", done)