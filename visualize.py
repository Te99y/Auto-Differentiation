import AutoDiff as ad
import numpy as np
import jax
import jax.numpy as jnp
from graphviz import Digraph
dot = Digraph(comment='Function graph')
dot.attr('node', fontcolor='gray80', color='gray80')
dot.attr('edge', color='gray80')
counter = [0]
nodes = []
edges = []
np.random.seed(20250616)

def add_node(label):
    node_id = f'n{counter[0]}'
    # dot.node(node_id, label)
    nodes.append((node_id, label))
    counter[0] += 1
    return node_id


input_dim = np.random.randint(low=1, high=10)
output_dim = np.random.randint(low=1, high=10)
function_depth = 50
input_width = 8
# function_depth = np.random.randint(low=1, high=10)
x = np.random.uniform(low=-2, high=2, size=(input_width, input_dim))
v = np.random.uniform(low=-2, high=2, size=(input_width, input_dim))
jnp_unary = [jnp.abs, jnp.negative, jnp.exp, jnp.log, jnp.sin, jnp.cos]
jnp_binary = [jnp.add, jnp.subtract, jnp.multiply, jnp.true_divide]

def op_name(op):
    # if op == jnp.add: return '+'
    # if op == jnp.subtract: return '-'
    # if op == jnp.multiply: return 'x'
    # if op == jnp.true_divide: return '÷'
    if op == jnp.add: return 'add'
    if op == jnp.subtract: return 'sub'
    if op == jnp.multiply: return 'mul'
    if op == jnp.true_divide: return 'truediv'


def create_random_function(inputs):
    old_indexes = list(range(len(inputs)))
    old_nodes = [add_node(label=str(i)) for i in range(len(inputs))]
    initial_nodes = set(old_node for old_node in old_nodes)

    # Keep root nodes at the same y
    with dot.subgraph() as s:
        s.attr(rank='same')
        for i, node_id in enumerate(old_nodes):
            s.node(node_id)
            # # Keeps root nodes in order
            # if i+1 < len(old_nodes):
            #     dot.edge(old_nodes[i], old_nodes[i+1], style='invis', weight='1')
    dot.attr('node', fontcolor='white', color='white')
    # dot.attr('edge', color='white')

    def apply_unary(op):
        rand_index = np.random.choice(input_width)
        rand_root = inputs[rand_index]
        new_node = add_node(f'{counter[0]} : {op.__name__}({old_indexes[rand_index]})')
        old_node = old_nodes[rand_index]
        # dot.edge(old_node, new_node)
        edges.append((old_node, new_node))

        old_indexes[rand_index] = counter[0]-1
        old_nodes[rand_index] = new_node
        return op(rand_root) if op != jnp.log else op(jnp.abs(rand_root))  # randomly select a root
    def apply_binary(op):
        # r = np.random.choice(counter[0])
        # print(counter)
        # print(r)
        # print((r+1)%counter[0])
        # rand_indexes = [r, (r+1)%counter[0]]
        r = np.random.choice(input_width)
        rand_indexes = [r, (r+1)%input_width]
        rand_root1, rand_root2 = inputs[rand_indexes[0]], inputs[rand_indexes[1]]
        new_node = add_node(f'{counter[0]} : {op_name(op)}({old_indexes[rand_indexes[0]]}, {old_indexes[rand_indexes[1]]})')
        # dot.edge(old_node, new_node)
        # dot.edge(old_nodes[0], new_node)
        edges.append((old_nodes[rand_indexes[0]], new_node))
        edges.append((old_nodes[rand_indexes[1]], new_node))

        old_indexes[rand_indexes[0]] = counter[0]-1
        old_nodes[r] = new_node

        return op(rand_root1, rand_root2)
    # def apply_binary(op):
    #     rand_index = np.random.choice(function_depth-1) + 1
    #     rand_root = inputs[rand_index]
    #     rand_root_first_operand = np.random.rand() > 0.5
    #     # first_operand, second_operand = np.random.permutation([rand_root, inputs[0]])
    #     (first_operand, second_operand) = (rand_root, inputs[0]) if rand_root_first_operand else (inputs[0], rand_root)
    #     new_node = add_node(f'{counter[0]} : {op_name(op)}({old_indexes[rand_index]}, {old_indexes[0]})') if rand_root_first_operand \
    #         else add_node(f'{counter[0]} : {op_name(op)}({old_indexes[0]}, {old_indexes[rand_index]})')
    #     # new_node = add_node(f'{counter[0]} : {old_indexes[rand_index]} {op_name(op)} {old_indexes[0]}') if rand_root_first_operand \
    #     #     else add_node(f'{counter[0]} : {old_indexes[0]} {op_name(op)} {old_indexes[rand_index]}')
    #     old_node = old_nodes[rand_index]
    #     # dot.edge(old_node, new_node)
    #     # dot.edge(old_nodes[0], new_node)
    #     edges.append((old_node, new_node))
    #     edges.append((old_nodes[0], new_node))
    #
    #     old_indexes[0] = counter[0]-1
    #     old_nodes[0] = new_node
    #
    #     return op(first_operand, second_operand)

    for _ in range(function_depth):
        func = np.random.choice([apply_unary, apply_binary])
        print(func)
        apply_unary(np.random.choice(jnp_unary)) if func == apply_unary \
            else apply_binary(np.random.choice(jnp_binary))

    # Output nodes at same height
    with dot.subgraph() as s:
        s.attr(rank='same')
        for i, node_id in enumerate(old_nodes):
            if node_id not in initial_nodes:
                s.node(node_id)
                # Keep output nodes in order
                # if i+1 < len(old_nodes):
                #     dot.edge(old_nodes[i], old_nodes[i+1], style='invis', weight='1')
    # Input - output nodes in same column
    # for input_node, output_node in zip(initial_nodes, old_nodes):
    #     dot.edge(input_node, output_node, style='invis', weight='1')

    return inputs


f = create_random_function
outputs = f(x)

# nodes = [node for i, node in enumerate(nodes) if node[1] != f'{i}']
# dot.attr(dpi='300')
dot.attr(bgcolor='black')
for node_id, label in nodes:
    dot.node(node_id, label)
for src, dst in edges:
    dot.edge(src, dst)

dot.render('random function', format='png', view=False)
