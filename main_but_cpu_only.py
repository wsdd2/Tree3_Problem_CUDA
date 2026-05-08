import argparse
import ctypes
import math
import os
import random
import sys
import time
from functools import lru_cache

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


COLOR_NAMES = ["A", "B", "C"]
COLOR_PALETTE = ["tab:red", "tab:green", "tab:blue"]
DEFAULT_TIMEOUT_SECONDS = 30 * 60
DEFAULT_RAM_LIMIT_GB = 8


class ResourceLimitExceeded(RuntimeError):
    pass


class ResourceGuard:
    def __init__(
        self,
        max_seconds=DEFAULT_TIMEOUT_SECONDS,
        max_ram_gb=DEFAULT_RAM_LIMIT_GB,
        check_interval=0.25,
    ):
        self.max_seconds = max_seconds
        self.max_ram_bytes = int(max_ram_gb * 1024**3)
        self.check_interval = check_interval
        self.start_time = time.perf_counter()
        self.last_check = 0.0

    def check(self, label="working"):
        now = time.perf_counter()
        if now - self.last_check < self.check_interval:
            return

        self.last_check = now
        elapsed = now - self.start_time
        if elapsed > self.max_seconds:
            raise ResourceLimitExceeded(
                f"Stopped while {label}: runtime {elapsed:.1f}s exceeded "
                f"{self.max_seconds:.1f}s."
            )

        ram = current_process_ram_bytes()
        if ram is not None and ram > self.max_ram_bytes:
            raise ResourceLimitExceeded(
                f"Stopped while {label}: RAM {ram / 1024**3:.2f}GB exceeded "
                f"{self.max_ram_bytes / 1024**3:.2f}GB."
            )


def current_process_ram_bytes():
    if os.name == "nt":
        return _current_process_ram_bytes_windows()
    return _current_process_ram_bytes_posix()


def _current_process_ram_bytes_windows():
    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(
        handle,
        ctypes.byref(counters),
        counters.cb,
    )
    if not ok:
        return None

    return int(counters.PrivateUsage or counters.WorkingSetSize)


def _current_process_ram_bytes_posix():
    try:
        import resource
    except ImportError:
        return None

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(usage)
    return int(usage * 1024)


def tree_size(t):
    color, children = t
    return 1 + sum(tree_size(child) for child in children)


def canon(t):
    return repr(t)


def random_partition_streaming(total, rng, max_children=64):
    """
    Low-memory random partition sampler.

    It does not enumerate all integer partitions. It picks a random number of
    children, randomly distributes nodes among them, then sorts the sizes to
    keep the rooted tree unordered.
    """
    if total <= 0:
        return []

    child_count = rng.randint(1, min(total, max_children))
    parts = [1] * child_count
    remaining = total - child_count

    for _ in range(remaining):
        parts[rng.randrange(child_count)] += 1

    parts.sort()
    return parts


def random_tree_exact_size(n, colors=3, rng=None, max_children=64, guard=None):
    if guard is not None:
        guard.check("generating random trees")

    if rng is None:
        rng = random.Random()

    root_color = rng.randrange(colors)
    if n == 1:
        return (root_color, ())

    child_sizes = random_partition_streaming(n - 1, rng, max_children=max_children)
    children = tuple(
        sorted(
            (
                random_tree_exact_size(
                    child_size,
                    colors=colors,
                    rng=rng,
                    max_children=max_children,
                    guard=guard,
                )
                for child_size in child_sizes
            ),
            key=canon,
        )
    )
    return (root_color, children)


def random_trees_upto(
    max_nodes,
    count,
    colors=3,
    seed=None,
    unique=True,
    max_children=64,
    guard=None,
):
    rng = random.Random(seed)
    result = []
    seen = set()
    attempts = 0
    max_attempts = max(count * 200, count)

    while len(result) < count and attempts < max_attempts:
        if guard is not None:
            guard.check("sampling random trees")

        attempts += 1
        node_count = rng.randint(1, max_nodes)
        tree = random_tree_exact_size(
            node_count,
            colors=colors,
            rng=rng,
            max_children=max_children,
            guard=guard,
        )

        if unique:
            key = canon(tree)
            if key in seen:
                continue
            seen.add(key)

        result.append(tree)

    if len(result) < count:
        print(
            f"Warning: only generated {len(result)} unique trees "
            f"after {attempts} attempts."
        )

    return result


def logaddexp_pair(a, b):
    if a == -float("inf"):
        return b
    if b == -float("inf"):
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def log_comb_repetition(log_type_count, copies):
    if copies == 0:
        return 0.0

    if log_type_count < 30.0:
        type_count = math.exp(log_type_count)
        return (
            math.lgamma(type_count + copies)
            - math.lgamma(type_count)
            - math.lgamma(copies + 1)
        )

    return copies * log_type_count - math.lgamma(copies + 1)


def count_tree3_cpu_dp(sum_node, colors=3, guard=None):
    """
    CPU-only count using dynamic programming in log-space.

    This avoids constructing every tree, so memory use stays small. For tiny
    node counts it matches exact enumeration, while larger counts are reported
    as magnitudes such as 10^x.
    """
    start = time.perf_counter()
    log_colors = math.log(colors)
    log_counts = [-float("inf")] * (sum_node + 1)

    if sum_node >= 1:
        log_counts[1] = log_colors

    for node_count in range(2, sum_node + 1):
        if guard is not None:
            guard.check("counting trees on CPU")

        remaining = node_count - 1
        log_dp = [-float("inf")] * (remaining + 1)
        log_dp[0] = 0.0

        for child_size in range(1, remaining + 1):
            if guard is not None:
                guard.check("counting trees on CPU")

            max_copies = remaining // child_size
            coeffs = [
                log_comb_repetition(log_counts[child_size], copies)
                for copies in range(max_copies + 1)
            ]
            next_log_dp = [-float("inf")] * (remaining + 1)

            for current_sum, current_log in enumerate(log_dp):
                if current_log == -float("inf"):
                    continue

                for copies, coeff_log in enumerate(coeffs):
                    new_sum = current_sum + copies * child_size
                    if new_sum > remaining:
                        break
                    next_log_dp[new_sum] = logaddexp_pair(
                        next_log_dp[new_sum],
                        current_log + coeff_log,
                    )

            log_dp = next_log_dp

        log_counts[node_count] = log_colors + log_dp[remaining]

    total_log = -float("inf")
    for value in log_counts[1:]:
        total_log = logaddexp_pair(total_log, value)

    elapsed = time.perf_counter() - start
    return total_log, log_counts, elapsed


def format_count_from_log(log_value):
    if log_value == -float("inf"):
        return "0"
    if not math.isfinite(log_value):
        return str(log_value)

    if log_value < math.log(10**15):
        return f"{round(math.exp(log_value)):,}"

    if log_value < math.log(sys.float_info.max):
        return f"{math.exp(log_value):.6e}"

    return f"10^{log_value / math.log(10):.3f}"


def print_counts(log_counts, label):
    parts = []
    for size, log_count in enumerate(log_counts):
        if size == 0:
            continue
        parts.append(f"{size}:{format_count_from_log(log_count)}")
    print(f"{label} counts by exact node size: " + ", ".join(parts))


@lru_cache(None)
def embeds_somewhere(a, b):
    if embeds_at(a, b):
        return True

    _, b_children = b
    return any(embeds_somewhere(a, child) for child in b_children)


@lru_cache(None)
def embeds_at(a, b):
    a_color, a_children = a
    b_color, b_children = b

    if a_color != b_color:
        return False

    return match_children(a_children, b_children)


def match_children(a_children, b_children):
    if len(a_children) == 0:
        return True

    if len(a_children) > len(b_children):
        return False

    possible = []
    for a_child in a_children:
        candidates = []
        for idx, b_child in enumerate(b_children):
            if embeds_somewhere(a_child, b_child):
                candidates.append(idx)
        if not candidates:
            return False
        possible.append(candidates)

    order = sorted(range(len(a_children)), key=lambda idx: len(possible[idx]))
    used = set()

    def backtrack(k):
        if k == len(order):
            return True

        child_idx = order[k]
        for target_idx in possible[child_idx]:
            if target_idx not in used:
                used.add(target_idx)
                if backtrack(k + 1):
                    return True
                used.remove(target_idx)

        return False

    return backtrack(0)


def tree_color_counts(t, colors=3):
    counts = [0] * colors

    def dfs(subtree):
        color, children = subtree
        counts[color] += 1
        for child in children:
            dfs(child)

    dfs(t)
    return counts


def tree_to_nx(t):
    graph = nx.DiGraph()
    pos = {}
    labels = {}
    colors = {}
    leaf_x = 0

    def dfs(subtree, depth=0, parent=None):
        nonlocal leaf_x

        node_id = len(graph)
        color, children = subtree
        graph.add_node(node_id)
        labels[node_id] = COLOR_NAMES[color]
        colors[node_id] = COLOR_PALETTE[color]

        if parent is not None:
            graph.add_edge(parent, node_id)

        if len(children) == 0:
            x = leaf_x
            leaf_x += 1
        else:
            child_xs = []
            for child in children:
                _, child_x = dfs(child, depth + 1, node_id)
                child_xs.append(child_x)
            x = sum(child_xs) / len(child_xs)

        pos[node_id] = (x, -depth)
        return node_id, x

    dfs(t)
    return graph, pos, labels, colors


def enable_interactive_navigation(fig, axes, title="Tree(3) CPU explorer"):
    axes = np.atleast_1d(axes).ravel().tolist()
    original_limits = {ax: (ax.get_xlim(), ax.get_ylim()) for ax in axes}
    state = {"press": None, "moved": False}

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
        ax.set_xlim(xdata - (xdata - x0) * scale, xdata + (x1 - xdata) * scale)
        ax.set_ylim(ydata - (ydata - y0) * scale, ydata + (y1 - ydata) * scale)

    def reset_views():
        for ax, (xlim, ylim) in original_limits.items():
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        fig.canvas.draw_idle()

    def on_scroll(event):
        if event.inaxes not in original_limits:
            return
        zoom(event.inaxes, event.xdata, event.ydata, 0.8 if event.button == "up" else 1.25)
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
        ax.set_xlim(x0 - dx_pixels * (x1 - x0) / bbox.width, x1 - dx_pixels * (x1 - x0) / bbox.width)
        ax.set_ylim(y0 - dy_pixels * (y1 - y0) / bbox.height, y1 - dy_pixels * (y1 - y0) / bbox.height)
        fig.canvas.draw_idle()

    def on_release(event):
        press = state["press"]
        state["press"] = None
        if press is None or state["moved"]:
            return
        if press["button"] == 1:
            zoom(press["ax"], event.xdata, event.ydata, 0.65)
        elif press["button"] == 3:
            zoom(press["ax"], event.xdata, event.ydata, 1.35)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key and event.key.lower() == "r":
            reset_views()

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)


def plot_trees(trees, cols=5, filename=None):
    total = len(trees)
    rows = max(1, math.ceil(total / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.6))

    if rows == 1:
        axes = np.array([axes])
    axes = axes.reshape(rows, cols)

    for ax in axes.ravel():
        ax.axis("off")

    for idx, tree in enumerate(trees):
        ax = axes[idx // cols, idx % cols]
        graph, pos, labels, node_colors = tree_to_nx(tree)
        size = tree_size(tree)
        node_size = max(90, min(650, int(2600 / math.sqrt(size))))
        font_size = max(4, min(10, int(18 / math.sqrt(size / 10))))

        nx.draw(
            graph,
            pos=pos,
            ax=ax,
            labels=labels,
            node_color=[node_colors[n] for n in graph.nodes],
            node_size=node_size,
            font_size=font_size,
            font_color="white",
            font_weight="bold",
            arrows=False,
            linewidths=1.2,
            edgecolors="black",
        )
        ax.set_title(f"#{idx + 1}, |V|={size}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=220, bbox_inches="tight")

    enable_interactive_navigation(fig, axes)
    plt.show()


def plot_embedding_matrix(trees, filename=None, guard=None):
    n = len(trees)
    matrix = np.zeros((n, n), dtype=np.uint8)
    sizes = [tree_size(tree) for tree in trees]
    color_counts = [tree_color_counts(tree) for tree in trees]

    for i, source in enumerate(trees):
        if guard is not None:
            guard.check("building embedding matrix")

        for j, target in enumerate(trees):
            if sizes[i] > sizes[j]:
                continue
            if any(a > b for a, b in zip(color_counts[i], color_counts[j])):
                continue
            if embeds_somewhere(source, target):
                matrix[i, j] = 1

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, interpolation="nearest")
    ax.set_xlabel("target tree j")
    ax.set_ylabel("source tree i")
    ax.set_title("CPU embedding matrix: M[i, j] = 1 if tree i embeds into tree j")
    fig.colorbar(image, ax=ax, label="embeds")
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=220, bbox_inches="tight")

    enable_interactive_navigation(fig, ax, title="Tree(3) CPU embedding matrix")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="CPU-only Tree(3)-style demo with time/RAM auto-stop guards."
    )
    parser.add_argument("--max-nodes", type=int, default=50)
    parser.add_argument("--sum-node", type=int, default=7)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--seed", type=int, default=114514)
    parser.add_argument("--colors", type=int, default=3)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--matrix-limit", type=int, default=20)
    parser.add_argument("--max-children", type=int, default=64)
    parser.add_argument("--timeout-minutes", type=float, default=30.0)
    parser.add_argument("--ram-limit-gb", type=float, default=8.0)
    parser.add_argument("--output-prefix", default="tree3_cpu_only")
    parser.add_argument("--skip-count", action="store_true")
    parser.add_argument("--count-only", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-matrix", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--allow-duplicates", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    guard = ResourceGuard(
        max_seconds=args.timeout_minutes * 60,
        max_ram_gb=args.ram_limit_gb,
    )

    if not args.skip_count:
        print(
            f"CPU-only count demo: total colored rooted trees with "
            f"node <= {args.sum_node}"
        )
        total_log, log_counts, elapsed = count_tree3_cpu_dp(
            args.sum_node,
            colors=args.colors,
            guard=guard,
        )
        print_counts(log_counts, "CPU")
        print(f"CPU total: {format_count_from_log(total_log)} in {elapsed:.6f} s")

    if args.count_only:
        return

    selected = random_trees_upto(
        max_nodes=args.max_nodes,
        count=args.limit,
        colors=args.colors,
        seed=args.seed,
        unique=not args.allow_duplicates,
        max_children=args.max_children,
        guard=guard,
    )
    print(
        f"Randomly generated {len(selected)} CPU-only trees "
        f"with node <= {args.max_nodes}"
    )

    for idx, tree in enumerate(selected[:10], start=1):
        print(f"#{idx}: size={tree_size(tree)}, tree={tree}")

    if not args.no_plot:
        tree_filename = None
        if not args.no_save:
            tree_filename = f"{args.output_prefix}_node_le_{args.max_nodes}.png"
        plot_trees(selected, cols=args.cols, filename=tree_filename)

    if not args.no_matrix:
        matrix_filename = None
        if not args.no_save:
            matrix_filename = (
                f"{args.output_prefix}_embedding_matrix_node_le_{args.max_nodes}.png"
            )
        plot_embedding_matrix(
            selected[:min(args.matrix_limit, len(selected))],
            filename=matrix_filename,
            guard=guard,
        )


if __name__ == "__main__":
    try:
        main()
    except ResourceLimitExceeded as exc:
        print(f"\nAUTO-STOP: {exc}", file=sys.stderr)
        sys.exit(2)
