import argparse
import importlib
import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import time
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
GPU_TREE_THRESHOLD = 50


def tree_size(t):
    """Number of vertices."""
    color, children = t
    return 1 + sum(tree_size(c) for c in children)


def canon(t):
    """Canonical string form, used for sorting and deduplication."""
    return repr(t)


def _load_torch_cuda():
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError(
            "GPU mode requires PyTorch with CUDA support. "
            "Install a CUDA-enabled torch build before running GPU mode."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("GPU mode requires an available CUDA GPU.")

    return torch


class GpuPartitionSampler:
    """
    Samples child-size partitions on CUDA and keeps the sample cache in GPU RAM.

    For large MAX_NODES we avoid enumerating all integer partitions on CPU.
    The returned tree structure is still a Python tuple because NetworkX and
    Matplotlib plotting operate on CPU-side objects.
    """

    def __init__(self, seed=None, cache_batch=512, max_children=64):
        self.torch = _load_torch_cuda()
        self.device = self.torch.device("cuda")
        self.cache_batch = cache_batch
        self.max_children = max_children
        self.cache = {}
        self.generator = self.torch.Generator(device=self.device)
        if seed is not None:
            self.generator.manual_seed(seed)

    def _refill(self, remaining):
        torch = self.torch
        max_children = min(remaining, self.max_children)
        batch = self.cache_batch

        child_counts = torch.randint(
            1,
            max_children + 1,
            (batch,),
            device=self.device,
            generator=self.generator,
        )
        columns = torch.arange(max_children, device=self.device)
        mask = columns.unsqueeze(0) < child_counts.unsqueeze(1)

        extras = remaining - child_counts
        weights = torch.rand(
            (batch, max_children),
            device=self.device,
            generator=self.generator,
        ).masked_fill(~mask, 0.0)
        scaled = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        scaled = scaled * extras.unsqueeze(1)

        alloc = torch.floor(scaled).to(torch.long).masked_fill(~mask, 0)
        remainder = extras - alloc.sum(dim=1)

        frac = (scaled - alloc).masked_fill(~mask, -1.0)
        rank = torch.argsort(frac, dim=1, descending=True)
        source = (columns.unsqueeze(0) < remainder.unsqueeze(1)).to(torch.long)
        add = torch.zeros_like(alloc)
        add.scatter_add_(1, rank, source)

        parts = (alloc + add + 1).masked_fill(~mask, 0)
        parts = torch.sort(parts, dim=1).values

        self.cache[remaining] = {
            "parts": parts,
            "lengths": child_counts,
            "cursor": 0,
        }

    def sample(self, remaining):
        entry = self.cache.get(remaining)
        if entry is None or entry["cursor"] >= entry["parts"].shape[0]:
            self._refill(remaining)
            entry = self.cache[remaining]

        row_idx = entry["cursor"]
        entry["cursor"] += 1

        length = int(entry["lengths"][row_idx].item())
        row = entry["parts"][row_idx, -length:]
        return row.to("cpu").tolist()


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


def count_tree3_cpu_enumeration(sum_node, colors=3):
    """
    Exact CPU baseline: enumerate every colored rooted tree up to sum_node.

    This is intentionally limited to small node counts. It is easy to explain
    to students, but it blows up quickly because it constructs every tree.
    """
    start = time.perf_counter()
    by_size, all_trees = generate_colored_rooted_trees(
        max_nodes=sum_node,
        colors=colors,
    )
    elapsed = time.perf_counter() - start

    counts_by_size = [0] * (sum_node + 1)
    for size in range(1, sum_node + 1):
        counts_by_size[size] = len(by_size[size])

    return len(all_trees), counts_by_size, elapsed


def count_tree3_gpu_dp(sum_node, colors=3):
    """
    Count colored rooted trees up to sum_node with CUDA dynamic programming.

    If there are a_s tree types with s nodes, then children of one root form an
    unordered multiset. The coefficient of product_s (1 - x^s)^(-a_s) gives the
    number of possible child multisets by total child-node count.

    Counts are stored in log-space, so large demos such as node <= 200 still
    finish without overflowing to inf. For node <= 7, the log result rounds back
    to the same exact integer as CPU enumeration.
    """
    torch = _load_torch_cuda()
    device = torch.device("cuda")
    log_colors = math.log(colors)

    torch.cuda.synchronize()
    start = time.perf_counter()

    log_counts = torch.full(
        (sum_node + 1,),
        -float("inf"),
        dtype=torch.float64,
        device=device,
    )
    if sum_node >= 1:
        log_counts[1] = log_colors

    for node_count in range(2, sum_node + 1):
        remaining = node_count - 1
        log_dp = torch.full(
            (remaining + 1,),
            -float("inf"),
            dtype=torch.float64,
            device=device,
        )
        log_dp[0] = 0.0

        for child_size in range(1, remaining + 1):
            log_type_count = log_counts[child_size]
            max_copies = remaining // child_size
            copies = torch.arange(
                max_copies + 1,
                dtype=torch.float64,
                device=device,
            )
            log_multiset_coeff = torch.full_like(copies, -float("inf"))
            log_multiset_coeff[0] = 0.0

            positive = copies > 0
            if bool(positive.any()):
                if float(log_type_count.item()) < 30.0:
                    type_count = torch.exp(log_type_count)
                    log_multiset_coeff[positive] = (
                        torch.lgamma(type_count + copies[positive])
                        - torch.lgamma(type_count)
                        - torch.lgamma(copies[positive] + 1)
                    )
                else:
                    # For huge a, C(a+k-1,k) is well approximated by a^k/k!.
                    log_multiset_coeff[positive] = (
                        copies[positive] * log_type_count
                        - torch.lgamma(copies[positive] + 1)
                    )

            next_log_dp = torch.full_like(log_dp, -float("inf"))
            for copy_count in range(max_copies + 1):
                offset = copy_count * child_size
                candidate = (
                    log_dp[:remaining + 1 - offset]
                    + log_multiset_coeff[copy_count]
                )
                next_log_dp[offset:] = torch.logaddexp(
                    next_log_dp[offset:],
                    candidate,
                )
            log_dp = next_log_dp

        log_counts[node_count] = log_colors + log_dp[remaining]

    total_log = torch.logsumexp(log_counts[1:], dim=0)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return float(total_log.item()), log_counts.cpu().tolist(), elapsed


def format_tree_count_from_log(log_value):
    if log_value == -float("inf"):
        return "0"
    if not math.isfinite(log_value):
        return str(log_value)

    if log_value < math.log(10**15):
        rounded = round(math.exp(log_value))
        return f"{rounded:,}"

    log10_value = log_value / math.log(10)
    if log_value < math.log(np.finfo(np.float64).max):
        return f"{math.exp(log_value):.6e}"

    return f"10^{log10_value:.3f}"


def print_counts_by_size(counts_by_size, label):
    formatted = []
    for size, count in enumerate(counts_by_size):
        if size == 0:
            continue

        if isinstance(count, float):
            formatted.append(f"{size}:{format_tree_count(count)}")
        else:
            formatted.append(f"{size}:{count:,}")

    print(f"{label} counts by exact node size: " + ", ".join(formatted))


def print_log_counts_by_size(log_counts_by_size, label):
    formatted = []
    for size, log_count in enumerate(log_counts_by_size):
        if size == 0:
            continue

        formatted.append(f"{size}:{format_tree_count_from_log(log_count)}")

    print(f"{label} counts by exact node size: " + ", ".join(formatted))


def benchmark_tree3_count(sum_node, colors=3):
    """
    Demonstrate CPU vs CUDA counting for Tree(3)-style colored rooted trees.

    node <= 7:
      Run both exact CPU enumeration and CUDA DP, then compare speed.
    node > 7:
      Skip CPU enumeration because it becomes the lesson rather than the demo.
    """
    print("\n" + "-" * 60)
    print(f"Tree(3) count demo: total colored rooted trees with node <= {sum_node}")

    if sum_node <= 7:
        cpu_total, cpu_counts, cpu_elapsed = count_tree3_cpu_enumeration(
            sum_node,
            colors=colors,
        )
        gpu_total_log, gpu_count_logs, gpu_elapsed = count_tree3_gpu_dp(
            sum_node,
            colors=colors,
        )
        gpu_total_rounded = round(math.exp(gpu_total_log))

        print_counts_by_size(cpu_counts, "CPU")
        print_log_counts_by_size(gpu_count_logs, "GPU")
        print(f"CPU total: {cpu_total:,} in {cpu_elapsed:.6f} s")
        print(
            f"GPU total: {format_tree_count_from_log(gpu_total_log)} "
            f"in {gpu_elapsed:.6f} s"
        )
        print(f"CPU/GPU same result: {cpu_total == gpu_total_rounded}")

        if gpu_elapsed > 0:
            speedup = cpu_elapsed / gpu_elapsed
            print(f"Speed ratio CPU/GPU: {speedup:.2f}x")

        if gpu_elapsed >= cpu_elapsed:
            print(
                "Note: for tiny node counts, CUDA launch overhead can be larger "
                "than the work itself. Increase SUM_NODE above 7 to show why CPU "
                "enumeration is skipped."
            )
    else:
        gpu_total_log, gpu_count_logs, gpu_elapsed = count_tree3_gpu_dp(
            sum_node,
            colors=colors,
        )
        print_log_counts_by_size(gpu_count_logs, "GPU")
        print(
            f"GPU total: {format_tree_count_from_log(gpu_total_log)} "
            f"in {gpu_elapsed:.6f} s"
        )
        print("CPU enumeration skipped because SUM_NODE > 7.")

    print("-" * 60 + "\n")


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


def random_tree_exact_size(n, colors=3, rng=None, partition_sampler=None):
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
    if partition_sampler is None:
        child_sizes = rng.choice(integer_partitions(remaining))
    else:
        child_sizes = partition_sampler.sample(remaining)

    children = []
    for sz in child_sizes:
        child = random_tree_exact_size(
            sz,
            colors=colors,
            rng=rng,
            partition_sampler=partition_sampler,
        )
        children.append(child)

    children = tuple(sorted(children, key=canon))

    return (root_color, children)


def random_trees_upto(
    max_nodes,
    count,
    colors=3,
    seed=None,
    unique=True,
    partition_sampler=None,
):
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
        t = random_tree_exact_size(
            n,
            colors=colors,
            rng=rng,
            partition_sampler=partition_sampler,
        )

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


def tree_color_counts(t, colors=3):
    counts = [0] * colors

    def dfs(subtree):
        color, children = subtree
        counts[color] += 1
        for child in children:
            dfs(child)

    dfs(t)
    return counts


def enable_interactive_navigation(fig, axes, title="Tree(3) explorer"):
    """
    Add classroom-friendly mouse navigation to Matplotlib figures.

    Controls:
      - Mouse wheel: zoom around cursor.
      - Left-click drag: pan the current subplot.
      - Left-click: zoom in.
      - Right-click: zoom out.
      - Middle-click or R: reset all views.
    """
    axes = np.atleast_1d(axes).ravel().tolist()
    original_limits = {
        ax: (ax.get_xlim(), ax.get_ylim())
        for ax in axes
    }
    state = {
        "press": None,
        "moved": False,
    }

    try:
        fig.canvas.manager.set_window_title(title)
    except AttributeError:
        pass

    def zoom(ax, xdata, ydata, scale):
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()

        if xdata is None:
            xdata = (x0 + x1) / 2
        if ydata is None:
            ydata = (y0 + y1) / 2

        ax.set_xlim(
            xdata - (xdata - x0) * scale,
            xdata + (x1 - xdata) * scale,
        )
        ax.set_ylim(
            ydata - (ydata - y0) * scale,
            ydata + (y1 - ydata) * scale,
        )

    def reset_views():
        for ax, (xlim, ylim) in original_limits.items():
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        fig.canvas.draw_idle()

    def on_scroll(event):
        if event.inaxes not in original_limits:
            return

        scale = 0.8 if event.button == "up" else 1.25
        zoom(event.inaxes, event.xdata, event.ydata, scale)
        fig.canvas.draw_idle()

    def on_press(event):
        if event.inaxes not in original_limits:
            return

        if event.button == 2:
            reset_views()
            return

        state["press"] = {
            "ax": event.inaxes,
            "button": event.button,
            "x": event.x,
            "y": event.y,
            "xlim": event.inaxes.get_xlim(),
            "ylim": event.inaxes.get_ylim(),
        }
        state["moved"] = False

    def on_motion(event):
        press = state["press"]
        if press is None or press["button"] != 1 or event.inaxes != press["ax"]:
            return

        ax = press["ax"]
        bbox = ax.get_window_extent()
        if bbox.width == 0 or bbox.height == 0:
            return

        dx_pixels = event.x - press["x"]
        dy_pixels = event.y - press["y"]
        if abs(dx_pixels) + abs(dy_pixels) > 4:
            state["moved"] = True

        x0, x1 = press["xlim"]
        y0, y1 = press["ylim"]
        dx_data = dx_pixels * (x1 - x0) / bbox.width
        dy_data = dy_pixels * (y1 - y0) / bbox.height

        ax.set_xlim(x0 - dx_data, x1 - dx_data)
        ax.set_ylim(y0 - dy_data, y1 - dy_data)
        fig.canvas.draw_idle()

    def on_release(event):
        press = state["press"]
        state["press"] = None
        if press is None or press["ax"] not in original_limits:
            return

        if state["moved"]:
            return

        ax = press["ax"]
        if press["button"] == 1:
            zoom(ax, event.xdata, event.ydata, 0.65)
            fig.canvas.draw_idle()
        elif press["button"] == 3:
            zoom(ax, event.xdata, event.ydata, 1.35)
            fig.canvas.draw_idle()

    def on_key(event):
        if event.key and event.key.lower() == "r":
            reset_views()

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)

    print(
        "Interactive controls: mouse wheel zooms; left-click drags to pan; "
        "left-click zooms in; right-click zooms out; middle-click or R resets."
    )


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

        size = tree_size(t)
        node_size = max(90, min(650, int(2600 / math.sqrt(size))))
        font_size = max(4, min(10, int(18 / math.sqrt(size / 10))))

        nx.draw(
            G,
            pos=pos,
            ax=ax,
            labels=labels,
            node_color=[node_colors[n] for n in G.nodes],
            node_size=node_size,
            font_size=font_size,
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

    enable_interactive_navigation(fig, axes, title="Tree(3) random trees")
    plt.show()


def plot_embedding_matrix(trees, filename=None, use_gpu=False):
    """
    Visualize whether earlier tree i embeds into later tree j.

    M[i, j] = 1 means tree i embeds somewhere in tree j.
    """
    n = len(trees)
    M_gpu = None
    candidate_pairs = None

    if use_gpu:
        torch = _load_torch_cuda()
        device = torch.device("cuda")

        sizes = torch.tensor([tree_size(t) for t in trees], device=device)
        color_counts = torch.tensor(
            [tree_color_counts(t) for t in trees],
            device=device,
        )
        size_ok = sizes.unsqueeze(1) <= sizes.unsqueeze(0)
        color_ok = (
            color_counts.unsqueeze(1) <= color_counts.unsqueeze(0)
        ).all(dim=2)
        candidate_pairs = (size_ok & color_ok).nonzero(as_tuple=False).to("cpu").tolist()
        M_gpu = torch.zeros((n, n), dtype=torch.uint8, device=device)
    else:
        M = np.zeros((n, n), dtype=int)

    pairs = candidate_pairs
    if pairs is None:
        pairs = ((i, j) for i in range(n) for j in range(n))

    for i, j in pairs:
        a = trees[i]
        b = trees[j]
        if M_gpu is None:
            if embeds_somewhere(a, b):
                M[i, j] = 1
        elif embeds_somewhere(a, b):
            M_gpu[i, j] = 1

    if M_gpu is not None:
        M = M_gpu.to("cpu").numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(M, interpolation="nearest")
    ax.set_xlabel("target tree j")
    ax.set_ylabel("source tree i")
    ax.set_title("Embedding matrix: M[i, j] = 1 if tree i embeds into tree j")
    fig.colorbar(image, ax=ax, label="embeds")
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=220, bbox_inches="tight")

    enable_interactive_navigation(fig, ax, title="Tree(3) embedding matrix")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Interactive Tree(3)-style demo: count colored rooted trees, "
            "sample large random trees, and visualize embeddings."
        )
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=50,
        help="Maximum node count for random tree visualization.",
    )
    parser.add_argument(
        "--sum-node",
        type=int,
        default=7,
        help="Count total tree types with node <= SUM_NODE.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=6,
        help="Number of random trees to generate and draw.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=114514,
        help="Random seed for repeatable classroom demos.",
    )
    parser.add_argument(
        "--colors",
        type=int,
        default=3,
        help="Number of node colors. Tree(3) uses 3.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=5,
        help="Number of subplot columns in the tree visualization.",
    )
    parser.add_argument(
        "--gpu-threshold",
        type=int,
        default=GPU_TREE_THRESHOLD,
        help="Use GPU sampling when max-nodes is greater than this value.",
    )
    parser.add_argument(
        "--matrix-limit",
        type=int,
        default=20,
        help="Maximum number of sampled trees used in the embedding matrix.",
    )
    parser.add_argument(
        "--gpu-cache-batch",
        type=int,
        default=512,
        help="Number of partition samples cached per GPU refill.",
    )
    parser.add_argument(
        "--gpu-max-children",
        type=int,
        default=64,
        help="Maximum sampled children per node in GPU random-tree mode.",
    )
    parser.add_argument(
        "--output-prefix",
        default="tree3_random",
        help="Prefix for saved PNG files.",
    )
    parser.add_argument(
        "--skip-count",
        action="store_true",
        help="Skip the CPU/GPU counting benchmark.",
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Only run the counting benchmark, then exit.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Generate trees but do not open the interactive tree plot.",
    )
    parser.add_argument(
        "--no-matrix",
        action="store_true",
        help="Skip the embedding matrix plot.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save generated PNG files.",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Allow duplicate random trees in the visualization sample.",
    )
    parser.add_argument(
        "--force-gpu-sampling",
        action="store_true",
        help="Use GPU random-tree sampling even when max-nodes is small.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    use_gpu = args.force_gpu_sampling or args.max_nodes > args.gpu_threshold
    partition_sampler = None

    if not args.skip_count:
        benchmark_tree3_count(args.sum_node, colors=args.colors)

    if args.count_only:
        return

    if use_gpu:
        partition_sampler = GpuPartitionSampler(
            seed=args.seed,
            cache_batch=args.gpu_cache_batch,
            max_children=args.gpu_max_children,
        )
        print(
            f"max_nodes={args.max_nodes} uses GPU sampling; "
            f"using CUDA and GPU RAM cache on {partition_sampler.device}"
        )

    selected = random_trees_upto(
        max_nodes=args.max_nodes,
        count=args.limit,
        colors=args.colors,
        seed=args.seed,
        unique=not args.allow_duplicates,
        partition_sampler=partition_sampler,
    )

    print(
        f"Randomly generated {len(selected)} trees "
        f"with node <= {args.max_nodes}"
    )

    for i, t in enumerate(selected[:10], start=1):
        print(f"#{i}: size={tree_size(t)}, tree={t}")

    if not args.no_plot:
        tree_filename = None
        if not args.no_save:
            tree_filename = f"{args.output_prefix}_node_le_{args.max_nodes}.png"

        plot_trees(
            selected,
            cols=args.cols,
            filename=tree_filename,
        )

    if not args.no_matrix:
        matrix_filename = None
        if not args.no_save:
            matrix_filename = (
                f"{args.output_prefix}_embedding_matrix_node_le_{args.max_nodes}.png"
            )

        plot_embedding_matrix(
            selected[:min(args.matrix_limit, len(selected))],
            filename=matrix_filename,
            use_gpu=use_gpu,
        )


if __name__ == "__main__":
    main()
