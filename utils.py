import random
import graphviz

class TreeNode:
    def __init__(self, state, depth=0, r=0):
        self.state = state
        self.depth = depth
        self.r = r    # discounted reward of transition from parent to node
        self.u = None    # used to store the value u of node in S and u(n) of node in T
        self.in_S = False    # indicator if node is in S
        # self.parent = None    # not needed if updating u values in the end
        self.children = []

    def add_child(self, child):
        self.children.append(child)

class Tree:
    def __init__(self, depth, branching_factor, n, discount_factor=0.9, make_viz=False, sampling="uniform"):
        self.depth = depth
        self.branching_factor = branching_factor
        self.discount_factor = discount_factor
        self.n = n
        self.sampling = sampling
        self.root = self.initialize_root(depth)
        self.make_viz = make_viz
        self.S = [self.root]
        self.T = []

    def initialize_root(self, depth):
        root = self.generate_tree(depth)
        root.in_S = True
        root.u = 0
        return root

    def generate_tree(self, depth, current_depth=0, reward_bound=1):
        if current_depth == depth:
            return None
        
        # assign reward
        discounted_reward = 0
        if current_depth > 0:
            discounted_reward = (self.discount_factor ** current_depth) * random.uniform(0, reward_bound)
        
        root = TreeNode(random.randint(0, 100), current_depth, discounted_reward)
        for i in range(self.branching_factor):
            if self.sampling == "uniform":
                child = self.generate_tree(depth, current_depth + 1, 1)
            elif self.sampling == "asymetric":
                child = self.generate_tree(depth, current_depth + 1, 2 * i / (self.branching_factor - 1))
            if child:
                root.add_child(child)
        return root
    
    def uniform_search(self):
        for i in range(self.n):
            self.expand_uniform()
            # make visualization - use to debug with SMALL trees
            # graph = self.to_graphviz()
            # graph.render(f'viz/tree_{i}', format='png', cleanup=True)
        self.update_u_values(self.root)

        # plot after updating u values
        if self.make_viz:
            graph = self.to_graphviz()
            graph.render(f'viz/tree_final', format='png', cleanup=True)

        return max([node.u for node in self.root.children])

    def expand_uniform(self):

        if len(self.S) == 0:
            pass

        top_node = self.S.pop(0)
        # u_children = []

        # expand children and compute u value
        for node in top_node.children:
            node.u = top_node.u + node.r
            node.in_S = True
            self.S.append(node)
            # u_children.append(node.u)
        
        # update u value for parents recursively - do it only in the end
        # top_node.u = max(u_children)

        # move expanded node to T
        top_node.in_S = False
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

    def add_nodes(self, dot, node):
        if node is None:
            return
        dot.node(str(id(node)), label=f"State: {node.state}\nDepth: {node.depth}\nReward: {node.r}\nu: {node.u}\nin_S: {node.in_S}")
        for child in node.children:
            self.add_nodes(dot, child)
            dot.edge(str(id(node)), str(id(child)), label=f"Reward: {child.r}")
