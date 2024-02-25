import random
import graphviz

class TreeNode:
    def __init__(self, state, depth=0, r=0):
        self.state = state
        self.depth = depth
        self.r = r    # discounted reward of transition from parent to node
        self.u = None    # used to store the value u of node in S and u(n) of node in T
        self.b = None
        self.in_S = False    # indicator if node is in S
        self.in_T = False    # indicator if node is in T
        self.children = []

    def add_child(self, child):
        self.children.append(child)

class Tree:
    def __init__(self, depth, branching_factor, n, discount_factor=0.9, make_viz=False, sampling="uniform", randomize="True",
                        reward_bound=1, initial_state=None, action_list=[], transition=None, watch_states=False):
        
        # for custom generation
        self.initial_state = initial_state
        self.action_list = action_list
        self.transition = transition
        # keeping track of states
        self.watch_states = watch_states
        self.state_list = []

        self.depth = depth
        self.branching_factor = branching_factor
        self.discount_factor = discount_factor
        self.n = n
        self.sampling = sampling
        self.reward_bound = reward_bound
        self.randomize = randomize
        self.root = self.initialize_root(depth)
        self.make_viz = make_viz
        self.S = [self.root]
        self.T = []

    def initialize_root(self, depth):
        if self.sampling == "from_transition":
            root =  self.generate_tree_from_transition(self.initial_state, 0, self.transition, depth, 0)
        else:
            root = self.generate_tree(depth, current_depth=0, reward_bound=self.reward_bound)
        
        root.in_S = True
        root.u = 0
        root.b = 1
        return root
    
    def generate_tree(self, depth, current_depth=0, reward_bound=1):
        if current_depth == depth:
            return None
        
        # assign reward
        discounted_reward = 0
        if current_depth > 0:
            if self.randomize:
                discounted_reward = (self.discount_factor ** current_depth) * random.uniform(0, reward_bound)
            else:
                discounted_reward = (self.discount_factor ** current_depth) * reward_bound
        
        root = TreeNode(random.randint(0, 100), current_depth, discounted_reward)
        for i in range(self.branching_factor):
            if self.sampling == "uniform":
                child = self.generate_tree(depth, current_depth + 1, reward_bound)
            elif self.sampling == "asymetric":
                if i < self.branching_factor-1:
                    child = self.generate_tree(depth, current_depth + 1, 0)    # 0 value
                else:
                    child = self.generate_tree(depth, current_depth + 1, reward_bound * (self.branching_factor - 1))
            if child:
                root.add_child(child)
        return root
    
    def generate_tree_from_transition(self, state, reward, transition, depth, current_depth=0):
        if current_depth == depth:
            return None
        
        # assign reward
        discounted_reward = (self.discount_factor ** current_depth) * reward
        
        root = TreeNode(state, current_depth, discounted_reward)

        for action in self.action_list:
            next_state, next_reward = transition(state, action)
            child = self.generate_tree_from_transition(next_state, next_reward, transition, depth, current_depth + 1)
            if child:
                root.add_child(child)

        return root
    
    def uniform_search(self):
        for i in range(self.n):
            self.expand_uniform()
        self.update_u_values(self.root)

        # plot after updating u values
        if self.make_viz:
            graph = self.to_graphviz()
            graph.render(f'viz/tree_final', format='png', cleanup=True)

        # return max([node.u for node in self.root.children])
        return self.root.u
    
    def optimistic_search(self):
        for i in range(self.n):
            self.expand_optimistic()
        self.update_u_values(self.root)

        return self.root.u

    def expand_uniform(self):

        if len(self.S) == 0:
            pass

        top_node = self.S.pop(0)

        # expand children and compute u value
        # shuffling children (no tie-breaking rule)
        for idx in random.sample(list(range(self.branching_factor)), self.branching_factor):
            node = top_node.children[idx]
            node.u = top_node.u + node.r
            node.in_S = True
            self.S.append(node)

        if self.watch_states:
            self.state_list.append(top_node.state)

        # move expanded node to T
        top_node.in_S = False
        top_node.in_T = True
        self.T.append(top_node)

    def expand_optimistic(self):

        if len(self.S) == 0:
            pass

        top_node = self.S.pop(0)

        # expand children and compute u value
        for node in top_node.children:
            node.u = top_node.u + node.r
            node.b = node.u + (self.discount_factor)**(node.depth) / (1 - self.discount_factor)
            node.in_S = True
            # push new node and sort (complexity could be reduced with priority queue)
            self.S.append(node)
            self.S.sort(key=lambda x: x.b, reverse=True)

        if self.watch_states:
            self.state_list.append(top_node.state)
        
        # move expanded node to T
        top_node.in_S = False
        top_node.in_T = True
        self.T.append(top_node)


    def update_u_values(self, node):
        # update u values in T
        if node.in_S or len(node.children) == 0:
            return

        u_children = []
        for child_node in node.children:
            self.update_u_values(child_node)
            u_children.append(child_node.u)
        node.u = max(u_children)

    def compute_max_reward(self, node, max_depth, curr_depth):
        if curr_depth == max_depth - 1:
            return node.r
        child_rewards = []
        for child_node in node.children:
            child_rewards.append(self.compute_max_reward(child_node, max_depth, curr_depth+1))
        return node.r + max(child_rewards)

    def compute_regret(self, reward):
        # computes regret with respect to obtained reward
        max_reward = self.compute_max_reward(self.root, self.depth, 0)
        return max_reward - reward
    
    def to_graphviz(self):
        dot = graphviz.Digraph()
        self.add_nodes(dot, self.root)
        return dot

    # def add_nodes(self, dot, node):
    #     if node is None:
    #         return
    #     dot.node(str(id(node)), label=f"State: {node.state}\nDepth: {node.depth}\nReward: {node.r}\nu: {node.u}\nin_S: {node.in_S}")
    #     for child in node.children:
    #         self.add_nodes(dot, child)
    #         dot.edge(str(id(node)), str(id(child)), label=f"Reward: {child.r}")

    def add_nodes(self, dot, node, cum_reward = 0):
        # Define color scheme
        low_reward_color = (240, 180, 180)   # light red
        high_reward_color = (100, 0, 0)      # dark

        # Define color interpolation function
        def color_interp(reward, min_reward, max_reward, low_color, high_color):
            low_r, low_g, low_b = low_color
            high_r, high_g, high_b = high_color
            return (
                int(low_r * (1 - (reward - min_reward) / (max_reward - min_reward)) + high_r * ((reward - min_reward) / (max_reward - min_reward))),
                int(low_g * (1 - (reward - min_reward) / (max_reward - min_reward)) + high_g * ((reward - min_reward) / (max_reward - min_reward))),
                int(low_b * (1 - (reward - min_reward) / (max_reward - min_reward)) + high_b * ((reward - min_reward) / (max_reward - min_reward)))
            )
        
        # Find minimum and maximum rewards
        min_reward = 0
        max_reward = 2
    
        if node is None:
            return
        color = '#%02x%02x%02x' % color_interp(cum_reward + node.r, min_reward, max_reward, low_reward_color, high_reward_color)
        dot.node(str(id(node)), label=f"ΣR: {round(cum_reward + node.r, 2)}", fillcolor=color, style='filled', fontcolor='white', width="0.1")
        for child in node.children:
            self.add_nodes(dot, child, cum_reward + node.r)
            edge_color = "black"
            if node.in_T:
                edge_color = "blue"
            dot.edge(str(id(node)), str(id(child)), label=f"R: {round(child.r, 2)}", color = edge_color)
