from functools import lru_cache
import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from functools import lru_cache

# ------------------------------------------------------------
# Tree representation:
#   tree = (color, children)
#   color: 0, 1, 2
#   children: tuple of child trees, sorted by canonical repr
#
# Example:
#   (0, ())
#   means a single root node colored 0.
#
#   (1, ((0, ()), (2, ())))
#   means root color 1, with two leaf children color 0 and 2.
# ------------------------------------------------------------


COLOR_NAMES = ["A", "B", "C"]
COLOR_PALETTE = ["tab:red", "tab:green", "tab:blue"]


def tree_size(t):
    """Number of vertices."""
    color, children = t
    return 1 + sum(tree_size(c) for c in children)


def canon(t):
    """Canonical string form, used for sorting and deduplication."""
    return repr(t)


def child_multisets(catalog, remaining, start=0):
    """
    Generate unordered multisets of child trees whose total size is `remaining`.

    catalog: list of (tree, size), sorted canonically.
    start: ensures combinations-with-replacement, avoiding permutations.
    """
    if remaining == 0:
        yield ()
        return

    for idx in range(start, len(catalog)):
        child, sz = catalog[idx]
        if sz > remaining:
            continue

        for rest in child_multisets(catalog, remaining - sz, idx):
            yield (child,) + rest


def generate_colored_rooted_trees(max_nodes=5, colors=3):
    """
    Enumerate all unordered rooted trees with node colors in {0, ..., colors-1},
    up to max_nodes vertices.

    Returns:
        by_size: dict[size] -> tuple of trees
        all_trees: list of all trees sorted by size then canonical repr
    """
    by_size = {}

    by_size[1] = tuple((c, ()) for c in range(colors))

    for n in range(2, max_nodes + 1):
        catalog = []
        for s in range(1, n):
            for t in by_size[s]:
                catalog.append((t, tree_size(t)))

        catalog.sort(key=lambda x: canon(x[0]))

        trees = set()

        for children in child_multisets(catalog, n - 1):
            children = tuple(sorted(children, key=canon))

            for root_color in range(colors):
                trees.add((root_color, children))

        by_size[n] = tuple(sorted(trees, key=canon))

    all_trees = []
    for n in range(1, max_nodes + 1):
        all_trees.extend(by_size[n])

    return by_size, all_trees


# ------------------------------------------------------------
# Simplified rooted topological embedding
#
# embeds_at(a, b):
#   Does tree a embed into tree b with roots matched?
#
# embeds_somewhere(a, b):
#   Does tree a embed somewhere inside tree b?
#
# This is a toy version suitable for visualization.
# Formal TREE(n) definitions may use slightly different conventions.
# ------------------------------------------------------------

@lru_cache(None)
def integer_partitions(n, min_part=1):
    """
    Generate integer partitions of n in nondecreasing order.

    Example:
        4 -> (1,1,1,1), (1,1,2), (1,3), (2,2), (4,)
    """
    if n == 0:
        return ((),)

    result = []
    for first in range(min_part, n + 1):
        for rest in integer_partitions(n - first, first):
            result.append((first,) + rest)

    return tuple(result)


def random_tree_exact_size(n, colors=3, rng=None):
    """
    Randomly generate one unordered colored rooted tree with exactly n nodes.

    This is for visualization sampling.
    It does not guarantee perfectly uniform sampling over all possible trees.
    """
    if rng is None:
        rng = random.Random()

    root_color = rng.randrange(colors)

    if n == 1:
        return (root_color, ())

    remaining = n - 1

    # Randomly choose a partition of remaining nodes among root's children.
    child_sizes = rng.choice(integer_partitions(remaining))

    children = []
    for sz in child_sizes:
        child = random_tree_exact_size(sz, colors=colors, rng=rng)
        children.append(child)

    children = tuple(sorted(children, key=canon))

    return (root_color, children)


def random_trees_upto(max_nodes, count, colors=3, seed=None, unique=True):
    """
    Randomly generate `count` trees with node number <= max_nodes.

    If unique=True, canonical deduplication is applied.
    """
    rng = random.Random(seed)
    result = []
    seen = set()

    attempts = 0
    max_attempts = count * 200

    while len(result) < count and attempts < max_attempts:
        attempts += 1

        n = rng.randint(1, max_nodes)
        t = random_tree_exact_size(n, colors=colors, rng=rng)

        if unique:
            key = canon(t)
            if key in seen:
                continue
            seen.add(key)

        result.append(t)

    if len(result) < count:
        print(
            f"Warning: only generated {len(result)} unique trees "
            f"after {attempts} attempts."
        )

    return result

@lru_cache(None)
def embeds_somewhere(a, b):
    if embeds_at(a, b):
        return True

    _, b_children = b
    return any(embeds_somewhere(a, bc) for bc in b_children)


@lru_cache(None)
def embeds_at(a, b):
    a_color, a_children = a
    b_color, b_children = b

    if a_color != b_color:
        return False

    return match_children(a_children, b_children)


def match_children(a_children, b_children):
    """
    Each child of a must be matched into a distinct child-subtree of b.
    We allow skipping intermediate nodes in b through embeds_somewhere().
    """
    if len(a_children) == 0:
        return True

    if len(a_children) > len(b_children):
        return False

    possible = []
    for ac in a_children:
        candidates = []
        for j, bc in enumerate(b_children):
            if embeds_somewhere(ac, bc):
                candidates.append(j)
        if not candidates:
            return False
        possible.append(candidates)

    order = sorted(range(len(a_children)), key=lambda i: len(possible[i]))
    used = set()

    def backtrack(k):
        if k == len(order):
            return True

        i = order[k]
        for j in possible[i]:
            if j not in used:
                used.add(j)
                if backtrack(k + 1):
                    return True
                used.remove(j)

        return False

    return backtrack(0)


# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------


def tree_to_nx(t):
    """
    Convert tree tuple to networkx DiGraph and handmade rooted layout.
    """
    G = nx.DiGraph()
    pos = {}
    labels = {}
    colors = {}

    leaf_x = 0

    def dfs(subtree, depth=0, parent=None):
        nonlocal leaf_x

        node_id = len(G)
        color, children = subtree

        G.add_node(node_id)
        labels[node_id] = COLOR_NAMES[color]
        colors[node_id] = COLOR_PALETTE[color]

        if parent is not None:
            G.add_edge(parent, node_id)

        if len(children) == 0:
            x = leaf_x
            leaf_x += 1
        else:
            child_xs = []
            for child in children:
                child_id, child_x = dfs(child, depth + 1, node_id)
                child_xs.append(child_x)
            x = sum(child_xs) / len(child_xs)

        pos[node_id] = (x, -depth)
        return node_id, x

    dfs(t)

    return G, pos, labels, colors


def plot_trees(trees, cols=5, filename=None):
    """
    Plot many small colored rooted trees.
    """
    total = len(trees)
    rows = math.ceil(total / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.6))

    if rows == 1:
        axes = np.array([axes])
    axes = axes.reshape(rows, cols)

    for ax in axes.ravel():
        ax.axis("off")

    for idx, t in enumerate(trees):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]

        G, pos, labels, node_colors = tree_to_nx(t)

        nx.draw(
            G,
            pos=pos,
            ax=ax,
            labels=labels,
            node_color=[node_colors[n] for n in G.nodes],
            node_size=650,
            font_color="white",
            font_weight="bold",
            arrows=False,
            linewidths=1.2,
            edgecolors="black",
        )

        ax.set_title(f"#{idx + 1}, |V|={tree_size(t)}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=220, bbox_inches="tight")

    plt.show()


def plot_embedding_matrix(trees, filename=None):
    """
    Visualize whether earlier tree i embeds into later tree j.

    M[i, j] = 1 means tree i embeds somewhere in tree j.
    """
    n = len(trees)
    M = np.zeros((n, n), dtype=int)

    for i, a in enumerate(trees):
        for j, b in enumerate(trees):
            if embeds_somewhere(a, b):
                M[i, j] = 1

    plt.figure(figsize=(7, 6))
    plt.imshow(M, interpolation="nearest")
    plt.xlabel("target tree j")
    plt.ylabel("source tree i")
    plt.title("Embedding matrix: M[i, j] = 1 if tree i embeds into tree j")
    plt.colorbar(label="embeds")
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=220, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    MAX_NODES = 10
    LIMIT = 50
    SEED = 114514

    selected = random_trees_upto(
        max_nodes=MAX_NODES,
        count=LIMIT,
        colors=3,
        seed=SEED,
        unique=True,
    )

    print(f"Randomly generated {len(selected)} trees with node <= {MAX_NODES}")

    for i, t in enumerate(selected[:10], start=1):
        print(f"#{i}: size={tree_size(t)}, tree={t}")

    plot_trees(
        selected,
        cols=5,
        filename=f"tree3_random_node_le_{MAX_NODES}.png",
    )

    plot_embedding_matrix(
        selected[:min(20, len(selected))],
        filename=f"tree3_random_embedding_matrix_node_le_{MAX_NODES}.png",
    )
