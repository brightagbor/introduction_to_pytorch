import numpy as np
import graphviz
from graphviz import Digraph
from IPython.display import display, Image

def visualize_network(
    cache,
    weights,
    biases,
    grads=None,
    filename="neural_network",
    show_values=True,
    view=True
):

    # Number of trainable layers
    n_layers = len(weights)

    # Create Graphviz diagram
    dot = Digraph(
        name="Neural_Network",
        format="png"
    )

    # General graph appearance
    dot.attr(
        rankdir="LR",
        size="22,12!",
        dpi="180",
        nodesep="0.65",
        ranksep="1.0",
        splines="spline",
        label="NEURAL NETWORK: FORWARD AND BACKWARD PROPAGATION",
        labelloc="t",
        fontsize="22",
        fontname="Arial"
    )

    dot.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fontname="Arial",
        fontsize="11",
        margin="0.15,0.10"
    )

    dot.attr(
        "edge",
        fontname="Arial",
        fontsize="9",
        arrowsize="0.8"
    )

    # Helper: format values for display
    def describe(value):
        if value is None:
            return "Not supplied"

        arr = np.asarray(value)

        if arr.ndim == 0:
            return str(round(float(arr), 4))

        return (
            f"Shape: {arr.shape}\n"
            f"Values: {np.array2string(arr, precision=3, threshold=8)}"
        )

    # Helper: create node labels
    def make_label(title, value=None, gradient=None):

        label = title

        if show_values and value is not None:
            label += "\n" + describe(value)

        if gradient is not None:
            label += "\n\nGradient:\n" + describe(gradient)

        return label

    # Get gradient by its dictionary key
    def get_grad(name):
        if grads is None:
            return None

        return grads.get(name)

    # -------------------------------------------------
    # INPUT NODE
    # -------------------------------------------------

    dot.node(
        "input",
        make_label("INPUT FEATURES (X)", cache.get("x")),
        fillcolor="lightblue"
    )

    # Store the last activation node
    previous_node = "input"

    # -------------------------------------------------
    # FORWARD PROPAGATION
    # -------------------------------------------------

    for i in range(1, n_layers + 1):

        # Determine activation name
        activation_name = (
            "y_pred" if i == n_layers
            and "y_pred" in cache
            else f"a{i}"
        )

        z_name = f"z{i}"
        a_name = activation_name

        # Get values
        z_value = cache.get(z_name)
        a_value = cache.get(a_name)

        w_value = weights[i - 1]
        b_value = biases[i - 1]

        # Parameter nodes
        w_node = f"W{i}"
        b_node = f"b{i}"

        dot.node(
            w_node,
            make_label(
                f"WEIGHTS W{i}",
                w_value,
                get_grad(f"dW{i}")
            ),
            fillcolor="lightyellow"
        )

        dot.node(
            b_node,
            make_label(
                f"BIAS b{i}",
                b_value,
                get_grad(f"db{i}")
            ),
            fillcolor="lightyellow"
        )

        # Weighted sum node
        dot.node(
            z_name,
            make_label(
                f"WEIGHTED SUM z{i}\n"
                f"z{i} = previous @ W{i} + b{i}",
                z_value,
                get_grad(f"dz{i}")
            ),
            fillcolor="lightcyan"
        )

        # Activation node
        activation = "Sigmoid" if i == n_layers else "ReLU"

        dot.node(
            a_name,
            make_label(
                f"{activation} ACTIVATION\n"
                f"{a_name} = {activation}(z{i})",
                a_value,
                get_grad(f"da{i}")
            ),
            fillcolor="lightgreen"
        )

        # Forward connections
        dot.edge(previous_node, z_name, label="Input")
        dot.edge(w_node, z_name, label="Weights")
        dot.edge(b_node, z_name, label="Bias")

        dot.edge(
            z_name,
            a_name,
            label=f"{activation}()"
        )

        # Prepare for next layer
        previous_node = a_name

    # -------------------------------------------------
    # LOSS NODE
    # -------------------------------------------------

    loss_value = cache.get("loss")

    dot.node(
        "loss",
        make_label("LOSS FUNCTION", loss_value),
        fillcolor="mistyrose"
    )

    dot.edge(
        previous_node,
        "loss",
        label="Prediction vs Target"
    )

    # -------------------------------------------------
    # BACKWARD PROPAGATION
    # Dashed red arrows represent reverse gradients
    # -------------------------------------------------

    for i in range(n_layers, 0, -1):

        activation_name = (
            "y_pred" if i == n_layers
            and "y_pred" in cache
            else f"a{i}"
        )

        z_name = f"z{i}"

        # Gradient through activation function
        dot.edge(
            activation_name,
            z_name,
            label="Backward gradient",
            color="red",
            style="dashed",
            fontcolor="red",
            constraint="false"
        )

        # Gradient to parameters
        dot.edge(
            z_name,
            f"W{i}",
            label=f"∂L/∂W{i}",
            color="red",
            style="dashed",
            fontcolor="red",
            constraint="false"
        )

        dot.edge(
            z_name,
            f"b{i}",
            label=f"∂L/∂b{i}",
            color="red",
            style="dashed",
            fontcolor="red",
            constraint="false"
        )

        # Gradient to preceding activation
        if i > 1:
            previous_activation = f"a{i-1}"

            dot.edge(
                z_name,
                previous_activation,
                label="Gradient to previous layer",
                color="red",
                style="dashed",
                fontcolor="red",
                constraint="false"
            )

        else:
            dot.edge(
                z_name,
                "input",
                label="Gradient to input",
                color="red",
                style="dashed",
                fontcolor="red",
                constraint="false"
            )

    # -------------------------------------------------
    # RENDER AND SAVE GRAPH
    # -------------------------------------------------

    output_path = dot.render(
        filename=filename,
        cleanup=True,
        view=False
    )

    print(f"Graph saved successfully: {output_path}")

    # Display graph inside Jupyter Notebook
    if view:
        display(Image(filename=output_path))

    return dot




def visualise_flow(
    steps,
    edges=None,
    filename="neural_network_flow",
    direction="TB",
    title="Neural Network Computational Graph",
    show_inline=True,
    size="10,10",
    dpi="120"
):
    """
    Flexible Graphviz visualisation.

    steps:
        List of tuples:
        (node_id, label, shape, color)

    edges:
        Optional list of connections:
        (source, destination, label, color, style)

    direction:
        TB = Top to Bottom
        BT = Bottom to Top
        LR = Left to Right
        RL = Right to Left
    """

    flow = graphviz.Digraph(
        comment=title,
        format="png"
    )

    flow.attr(
        rankdir=direction,
        size=size,
        dpi=dpi,
        nodesep="0.6",
        ranksep="0.8",
        label=title,
        labelloc="t",
        fontsize="20"
    )

    flow.attr(
        "node",
        fontname="Arial",
        fontsize="11",
        style="filled",
        margin="0.15,0.10"
    )

    flow.attr(
        "edge",
        fontname="Arial",
        fontsize="10",
        arrowsize="0.8"
    )

    # Create nodes
    for node_id, label, shape, color in steps:

        flow.node(
            str(node_id),
            str(label),
            shape=shape,
            fillcolor=color if color else "white"
        )

    # Create custom edges
    if edges is not None:

        for edge in edges:

            source, destination = edge[:2]

            label = edge[2] if len(edge) > 2 else ""
            color = edge[3] if len(edge) > 3 else "black"
            style = edge[4] if len(edge) > 4 else "solid"

            flow.edge(
                str(source),
                str(destination),
                label=label,
                color=color,
                style=style,
                fontcolor=color
            )

    else:

        # Connect nodes sequentially
        for i in range(len(steps) - 1):

            flow.edge(
                str(steps[i][0]),
                str(steps[i + 1][0])
            )

    # Save graph
    output_file = flow.render(
        filename=filename,
        view=False,
        cleanup=True
    )

    print(f"Graph saved successfully: {output_file}")

    # Display graph
    if show_inline:
        display(Image(filename=output_file))

    return flow
