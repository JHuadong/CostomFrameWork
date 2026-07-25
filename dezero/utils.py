from dezero import Variable
import numpy as np
import os
import subprocess


def _dot_var(v, verbose=False, color='orange'):
    dot_var = '{} [label="{}", color={}, style=filled]\n'

    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)

    return dot_var.format(id(v), name, color)

def _dot_func(f, color='lightblue'):
    dot_func = '{} [label="{}", color={}, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__, color)

    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))  # y is weakref
    return txt

def get_dot_graph(output, verbose=False):
    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            # funcs.sort(key=lambda x: x.generation)
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose=verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose=verbose)

            if x.creator is not None:
                add_func(x.creator)

    return 'digraph g {\n' + txt + '}'

def plot_dot_graph(output, verbose=True, to_file='graph.png', from_file='tmp_graph.dot'):
    dot_graph = get_dot_graph(output, verbose)

    # ①Save the dot data to a file
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tmp_dir = os.path.join(parent_dir, "graphviz\\")
    # print(tmp_dir)
    if not os.path.exists(tmp_dir): # If the ~/.dezero directory does not exist, create it.
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, from_file)
    # print(graph_path)

    with open(graph_path, 'w', encoding='utf-8') as f:
        f.write(dot_graph)

    # with open(graph_path, 'r', encoding='utf-8-sig') as f:
    #     content = f.read()
    #
    # with open(graph_path, 'w', encoding='utf-8') as f:
    #     f.write(content)

    # ②Run the dot command
    extension = os.path.splitext(to_file)[1][1:] # File extensions (png, pdf, etc.)
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, os.path.join(tmp_dir, to_file))
    # print(cmd)
    subprocess.run(cmd, shell=True, text=True)


if __name__ == '__main__':
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(1.0))
    y = x0 + x1
    txt = _dot_func(y.creator)
    print(txt)
