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

def sum_to(x, shape):
    """Sum elements along axes to output an array of a given shape.

    Args:
        x (ndarray): Input array.
        shape:

    Returns:
        ndarray: Output array of the shape.
    """
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y

def reshape_sum_backward(gy, x_shape, axis, keepdims):
    """Reshape gradient appropriately for dezero.functions.sum's backward.

    Args:
        gy (dezero.Variable): Gradient variable from the output by backprop.
        x_shape (tuple): Shape used at sum function's forward.
        axis (None or int or tuple of ints): Axis used at sum function's
            forward.
        keepdims (bool): Keepdims used at sum function's forward.

    Returns:
        dezero.Variable: Gradient variable which is reshaped appropriately
    """
    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    gy = gy.reshape(shape)  # reshape
    return gy

def logsumexp(x, axis=1):
    m = x.max(axis=axis, keepdims=True)
    y = x - m
    np.exp(y, out=y)
    s = y.sum(axis=axis, keepdims=True)
    np.log(s, out=s)
    m += s
    return m


if __name__ == '__main__':
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(1.0))
    y = x0 + x1
    txt = _dot_func(y.creator)
    print(txt)
