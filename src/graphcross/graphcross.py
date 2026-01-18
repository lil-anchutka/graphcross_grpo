

import argparse
import json
import pathlib
import random
import string
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Set

from base.data import Data
from base.env import Env
from base.verifier import Verifier

from graphcross.graphcross_prompt import prompt_graphcross
from graphcross.graphcross_verifier import GraphCrossVerifier


class GraphCrossEnv(Env):
    """
    GraphCross environment:
    - Variables: slots with fixed lengths
    - Constraints: explicit intersections (slot_u[i] == slot_v[j])
    - Candidates: gold + distractors
        * d0: corrupt ALL intersection chars for a slot (hard conflict)
        * d1: corrupt 1..deg(slot) intersection chars (partial conflict)
        * d2 (xor_traps): chained trap along a path; locally plausible, globally impossible
    """

    def __init__(self, name: str = "graphcross", verifier: Verifier = GraphCrossVerifier):
        super().__init__(name=name, verifier=verifier)

    def extract_answer(self, test_solution: str):
        return self.verifier.extract_answer(test_solution)

    # ---- difficulty -> hyperparams ----
    @staticmethod
    def difficulty_to_params(difficulty: int) -> dict:
        table = {
        0:  dict(n_slots=4,  avg_deg=(1.5, 2.0), cand=(2, 2),   d0=0, d1=0, xor_depth=(0, 0), xor_traps=(0, 0)),
        1:  dict(n_slots=5,  avg_deg=(1.8, 2.2), cand=(2, 3),   d0=0, d1=0, xor_depth=(0, 0), xor_traps=(0, 0)),
        2:  dict(n_slots=6,  avg_deg=(2.0, 2.5), cand=(3, 3),   d0=1, d1=0, xor_depth=(0, 0), xor_traps=(0, 0)),
        3:  dict(n_slots=7,  avg_deg=(2.2, 2.8), cand=(3, 4),   d0=1, d1=1, xor_depth=(0, 0), xor_traps=(0, 0)),
        4:  dict(n_slots=8,  avg_deg=(2.5, 3.0), cand=(4, 4),   d0=2, d1=1, xor_depth=(0, 0), xor_traps=(0, 0)),

        5:  dict(n_slots=9,  avg_deg=(2.8, 3.3), cand=(4, 5),   d0=2, d1=2, xor_depth=(1, 2), xor_traps=(0, 1)),
        6:  dict(n_slots=10, avg_deg=(3.0, 3.6), cand=(5, 6),   d0=3, d1=2, xor_depth=(1, 3), xor_traps=(1, 2)),

        7:  dict(n_slots=12, avg_deg=(3.3, 3.9), cand=(6, 7),   d0=3, d1=3, xor_depth=(2, 3), xor_traps=(1, 2)),
        8:  dict(n_slots=13, avg_deg=(3.6, 4.2), cand=(7, 8),   d0=4, d1=3, xor_depth=(2, 4), xor_traps=(2, 3)),
        9:  dict(n_slots=14, avg_deg=(4.0, 4.6), cand=(9, 10),  d0=4, d1=4, xor_depth=(3, 5), xor_traps=(3, 4)),
        10: dict(n_slots=15, avg_deg=(4.5, 5.2), cand=(10, 12), d0=5, d1=5, xor_depth=(4, 6), xor_traps=(4, 6)),
        }

        d = max(1, min(10, int(difficulty)))
        return table[d].copy()

    # ---- helpers ----
    @staticmethod
    def _rand_char(exclude: Optional[Set[str]] = None, alphabet: str = string.ascii_lowercase) -> str:
        exclude = exclude or set()
        choices = [c for c in alphabet if c not in exclude]
        return random.choice(choices) if choices else random.choice(string.ascii_lowercase)

    @staticmethod
    def _build_connected_graph(n: int, m_edges: int) -> List[Tuple[int, int]]:
        if n <= 1:
            return []
        edges = set()
        for v in range(1, n):
            u = random.randrange(0, v)
            edges.add(tuple(sorted((u, v))))
        while len(edges) < m_edges:
            u, v = random.sample(range(n), 2)
            edges.add(tuple(sorted((u, v))))
        return list(edges)

    @staticmethod
    def _adj_from_edges(n: int, edges: List[Tuple[int, int]]) -> Dict[int, Set[int]]:
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    @staticmethod
    def _edge_pos_map(inter: List[Tuple[int, int, int, int]]) -> Dict[Tuple[int, int], Tuple[int, int]]:
        """
        Map undirected edge (min(u,v),max(u,v)) -> (pos_in_u, pos_in_v) in that order.
        """
        mp = {}
        for u, pu, v, pv in inter:
            a, b = (u, v) if u < v else (v, u)
            p_a, p_b = (pu, pv) if u < v else (pv, pu)
            mp[(a, b)] = (p_a, p_b)
        return mp

    #  SOLVER (UNIQUE CHECK)
    def _count_solutions(
        self,
        candidates: Dict[str, List[str]],
        intersections: List[Tuple[str, int, str, int]],
        max_solutions: int = 2,
    ) -> int:
        slots = list(candidates.keys())
        slot_idx = {s: i for i, s in enumerate(slots)}

        adj = defaultdict(list)
        for a, ia, b, ib in intersections:
            u, v = slot_idx[a], slot_idx[b]
            adj[u].append((v, ia, ib))
            adj[v].append((u, ib, ia))

        domains = {i: list(candidates[slots[i]]) for i in range(len(slots))}
        solutions = 0

        def consistent(i, w, assign):
            for j, pi, pj in adj[i]:
                if j in assign:
                    if w[pi] != assign[j][pj]:
                        return False
            return True

        def forward(domains, i, w, assign):
            new_domains = {}
            for j in domains:
                if j in assign or j == i:
                    continue
                filt = []
                for cand in domains[j]:
                    ok = True
                    for k, pi, pj in adj[j]:
                        # constraint: slot j at pi == slot k at pj
                        if k == i:
                            if cand[pi] != w[pj]:
                                ok = False
                                break
                        elif k in assign:
                            if cand[pi] != assign[k][pj]:
                                ok = False
                                break
                    if ok:
                        filt.append(cand)
                if not filt:
                    return None
                new_domains[j] = filt
            return new_domains

        def backtrack(assign, domains):
            nonlocal solutions
            if solutions >= max_solutions:
                return
            if len(assign) == len(slots):
                solutions += 1
                return

            var = min((i for i in domains if i not in assign),
                      key=lambda x: len(domains[x]))

            for w in domains[var]:
                if not consistent(var, w, assign):
                    continue
                new_assign = dict(assign)
                new_assign[var] = w
                new_domains = dict(domains)
                new_domains[var] = [w]
                pruned = forward(new_domains, var, w, assign)
                if pruned is None:
                    continue
                new_domains.update(pruned)
                backtrack(new_assign, new_domains)
                if solutions >= max_solutions:
                    return

        backtrack({}, domains)
        return solutions

    # ---- d2 XOR/TRAP generation ----
    def _add_xor_traps(
        self,
        *,
        n_slots: int,
        adj: Dict[int, Set[int]],
        inter: List[Tuple[int, int, int, int]],
        lengths: Dict[int, int],
        gold: Dict[str, str],
        candidates: Dict[str, List[str]],
        d2_traps: int,
        depth_range: Tuple[int, int],
        alphabet: str,
    ) -> None:
        """
        Create chained traps:
        - Pick a simple path u0-u1-...-ud (d = depth)
        - For each path edge, pick fresh char x_k and set BOTH endpoints' intersection chars to x_k
        - For the terminal ud, also corrupt ONE external edge (ud - w where w not in path) to char y
          chosen to NOT appear at that external neighbor position in ANY of its candidates.
        Result:
        - Along the path, these trap words look mutually consistent (so search explores them)
        - But when you finally connect to w, it's impossible (global dead-end).
        """
        if d2_traps <= 0:
            return

        edge_pos = self._edge_pos_map(inter)

        def inter_positions_for_node(i: int) -> List[int]:
            pos = []
            for u, pu, v, pv in inter:
                if u == i:
                    pos.append(pu)
                elif v == i:
                    pos.append(pv)
            return pos

        # Precompute which chars appear at each slot's intersection position among its current candidates
        def chars_at_slot_pos(slot: int, pos: int) -> Set[str]:
            sid = f"S{slot}"
            return {w[pos] for w in candidates[sid]}

        # Helper: find a random simple path of a given length
        def random_simple_path(path_len_edges: int) -> Optional[List[int]]:
            start = random.randrange(n_slots)
            path = [start]
            used = {start}
            for _ in range(path_len_edges):
                cur = path[-1]
                nbrs = [x for x in adj[cur] if x not in used]
                if not nbrs:
                    return None
                nxt = random.choice(nbrs)
                path.append(nxt)
                used.add(nxt)
            return path

        # Try multiple times to place all traps
        placed = 0
        for _ in range(d2_traps * 25):
            if placed >= d2_traps:
                break

            dmin, dmax = depth_range
            if dmax <= 0:
                break
            depth = random.randint(max(1, dmin), max(1, dmax))

            path = random_simple_path(depth)
            if not path:
                continue

            path_set = set(path)
            terminal = path[-1]

            # pick external neighbor for terminal not in path
            ext_neighbors = [w for w in adj[terminal] if w not in path_set]
            if not ext_neighbors:
                continue
            w = random.choice(ext_neighbors)

            # need positions for terminal-w edge
            a, b = (terminal, w) if terminal < w else (w, terminal)
            if (a, b) not in edge_pos:
                continue
            p_a, p_b = edge_pos[(a, b)]
            p_term = p_a if terminal == a else p_b
            p_w = p_b if terminal == a else p_a

            # choose y that is guaranteed NOT to be achievable by neighbor w at its position p_w
            forbidden = chars_at_slot_pos(w, p_w)
            y_choices = [c for c in alphabet if c not in forbidden]
            if not y_choices:
                continue
            y = random.choice(y_choices)

            # For each path edge, pick fresh x_k
            x_chars = [random.choice(alphabet) for _ in range(len(path) - 1)]

            # Build per-node modifications: list[(pos, char)]
            mods: Dict[int, List[Tuple[int, str]]] = {i: [] for i in path}
            for k in range(len(path) - 1):
                u = path[k]
                v = path[k + 1]
                a, b = (u, v) if u < v else (v, u)
                p_a, p_b = edge_pos[(a, b)]
                p_u = p_a if u == a else p_b
                p_v = p_b if u == a else p_a
                x = x_chars[k]
                mods[u].append((p_u, x))
                mods[v].append((p_v, x))

            # Terminal extra conflict to external neighbor
            mods[terminal].append((p_term, y))

            # Add exactly one trap word per path node
            ok_place = True
            for node in path:
                sid = f"S{node}"
                wlist = list(gold[sid])

                # apply modifications (could include duplicates; keep last)
                for pos, ch in mods[node]:
                    wlist[pos] = ch

                # also: keep all other intersection positions as in gold to maximize "local plausibility"
                # (no extra changes)

                trap_word = "".join(wlist)
                if trap_word == gold[sid]:
                    ok_place = False
                    break
                if trap_word not in candidates[sid]:
                    candidates[sid].append(trap_word)

            if not ok_place:
                continue

            placed += 1

    #  GENERATION
    def generate(
        self,
        num_of_questions: int = 100,
        max_attempts: int = 100,
        difficulty: Optional[int] = 1,
        **kwargs
    ) -> List[Data]:
        results: List[Data] = []
        diff = 1 if difficulty is None else int(difficulty)

        margin_range = kwargs.get("margin_range", (4, 10))
        len_min = kwargs.get("len_min", 6)
        len_max = kwargs.get("len_max", 20)
        alphabet = kwargs.get("alphabet", string.ascii_lowercase)

        base = self.difficulty_to_params(diff)

        n_slots = int(kwargs.get("n_slots", base["n_slots"]))
        avg_deg_range = kwargs.get("avg_deg", base["avg_deg"])
        cand_range = kwargs.get("cand_per_slot", base["cand"])
        d0 = int(kwargs.get("d0", base["d0"]))
        d1 = int(kwargs.get("d1", base["d1"]))
        xor_depth_range = kwargs.get("xor_depth", base["xor_depth"])
        xor_traps_range = kwargs.get("xor_traps", base["xor_traps"])

        for _ in range(num_of_questions):
            last_err = None

            for _attempt in range(max_attempts):
                try:
                    # --- graph ---
                    target_avg = random.uniform(*avg_deg_range)
                    m_edges = max(n_slots - 1, int(round(n_slots * target_avg / 2)))
                    edges = self._build_connected_graph(n_slots, m_edges)
                    adj = self._adj_from_edges(n_slots, edges)
                    degrees = {i: len(adj[i]) for i in range(n_slots)}

                    # --- lengths ---
                    margin = random.randint(*margin_range)
                    lengths = {}
                    for i in range(n_slots):
                        L = max(degrees[i] + margin, len_min)
                        if L > len_max:
                            raise ValueError(f"length overflow: slot={i} L={L} len_max={len_max}")
                        lengths[i] = L

                    # --- intersections ---
                    free_pos = {i: list(range(lengths[i])) for i in range(n_slots)}
                    for i in free_pos:
                        random.shuffle(free_pos[i])

                    inter = []
                    for u, v in edges:
                        inter.append((u, free_pos[u].pop(), v, free_pos[v].pop()))
                    inter_named = [(f"S{u}", pu, f"S{v}", pv) for u, pu, v, pv in inter]

                    # --- gold ---
                    gold_chars = {i: [None] * lengths[i] for i in range(n_slots)}
                    for u, pu, v, pv in inter:
                        c = random.choice(alphabet)
                        gold_chars[u][pu] = c
                        gold_chars[v][pv] = c
                    for i in range(n_slots):
                        for j in range(lengths[i]):
                            if gold_chars[i][j] is None:
                                gold_chars[i][j] = random.choice(alphabet)

                    gold = {f"S{i}": "".join(gold_chars[i]) for i in range(n_slots)}

                    # --- candidates ---
                    candidates = {f"S{i}": [gold[f"S{i}"]] for i in range(n_slots)}

                    for i in range(n_slots):
                        sid = f"S{i}"

                        # ---- d0: corrupt ALL intersection chars for this slot (conflict with all neighbors)
                        for _ in range(d0):
                            w = list(gold[sid])
                            for (u, pu, v, pv) in inter:
                                if u == i:
                                    w[pu] = self._rand_char({gold[sid][pu]}, alphabet)
                                elif v == i:
                                    w[pv] = self._rand_char({gold[sid][pv]}, alphabet)
                            cand = "".join(w)
                            if cand not in candidates[sid]:
                                candidates[sid].append(cand)

                        # ---- d1: corrupt ONLY intersection chars; count corrupted = random(1..deg)
                        inter_pos = []
                        for (u, pu, v, pv) in inter:
                            if u == i:
                                inter_pos.append(pu)
                            elif v == i:
                                inter_pos.append(pv)

                        for _ in range(d1):
                            if not inter_pos:
                                continue
                            w = list(gold[sid])
                            k = random.randint(1, len(inter_pos))
                            for pos in random.sample(inter_pos, k):
                                w[pos] = self._rand_char({gold[sid][pos]}, alphabet)
                            cand = "".join(w)
                            if cand not in candidates[sid]:
                                candidates[sid].append(cand)

                    # ---- d2 (xor traps): locally plausible, globally impossible
                    d2_traps = random.randint(*xor_traps_range)
                    self._add_xor_traps(
                        n_slots=n_slots,
                        adj=adj,
                        inter=inter,
                        lengths=lengths,
                        gold=gold,
                        candidates=candidates,
                        d2_traps=d2_traps,
                        depth_range=xor_depth_range,
                        alphabet=alphabet,
                    )

                    # --- trim + fill ---
                    cand_min, cand_max = cand_range
                    for i in range(n_slots):
                        sid = f"S{i}"

                        if len(candidates[sid]) > cand_max:
                            gold_first = candidates[sid][0]
                            rest = candidates[sid][1:]
                            random.shuffle(rest)
                            candidates[sid] = [gold_first] + rest[:cand_max - 1]

                        while len(candidates[sid]) < cand_min:
                            w = "".join(random.choice(alphabet) for _ in range(lengths[i]))
                            if w not in candidates[sid]:
                                candidates[sid].append(w)

                    # --- UNIQUE CHECK ---
                    n_sol = self._count_solutions(candidates, inter_named)
                    if n_sol != 1:
                        # if extra solution appears, try max_attempts more times
                        if n_sol > 1 and max_attempts > 0:
                            max_attempts -= 1
                            last_err = RuntimeError(f"non-unique solutions: {n_sol} (attemps_left={max_attempts})")
                            continue
                        last_err = RuntimeError(f"non-unique solutions: {n_sol} (attempts_left={max_attempts})")
                        continue

                    question = prompt_graphcross(
                        {f"S{i}": lengths[i] for i in range(n_slots)},
                        inter_named,
                        candidates
                    )

                    results.append(
                        Data(
                            question=question,
                            answer=json.dumps(gold, sort_keys=True),
                            difficulty=diff,
                            metadata={
                                "slots": lengths,
                                "intersections": inter_named,
                                "candidates": candidates,
                                "d2_traps": d2_traps,
                                "xor_depth_range": xor_depth_range,
                            },
                        )
                    )
                    break

                except Exception as e:
                    last_err = e
                    continue

            else:
                raise RuntimeError(f"Generation failed. Last error: {last_err}")

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphCross task generator")

    # --- core ---
    parser.add_argument("--num_of_data", type=int, default=100)
    parser.add_argument("--max_attempts", type=int, default=500)
    parser.add_argument("--difficulty", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)

    # --- optional hyperparameter overrides ---
    parser.add_argument("--n_slots", type=int)
    parser.add_argument("--avg_deg_min", type=float)
    parser.add_argument("--avg_deg_max", type=float)

    parser.add_argument("--cand_min", type=int)
    parser.add_argument("--cand_max", type=int)

    parser.add_argument("--d0", type=int)
    parser.add_argument("--d1", type=int)

    parser.add_argument("--xor_depth_min", type=int)
    parser.add_argument("--xor_depth_max", type=int)

    parser.add_argument("--xor_traps_min", type=int)
    parser.add_argument("--xor_traps_max", type=int)

    parser.add_argument("--margin_min", type=int, default=4)
    parser.add_argument("--margin_max", type=int, default=10)

    parser.add_argument("--out_dir", type=str, default="data")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # ---- collect kwargs for generate() ----
    gen_kwargs = {}

    if args.n_slots is not None:
        gen_kwargs["n_slots"] = args.n_slots

    if args.avg_deg_min is not None and args.avg_deg_max is not None:
        gen_kwargs["avg_deg"] = (args.avg_deg_min, args.avg_deg_max)

    if args.cand_min is not None and args.cand_max is not None:
        gen_kwargs["cand_per_slot"] = (args.cand_min, args.cand_max)

    if args.d0 is not None:
        gen_kwargs["d0"] = args.d0
    if args.d1 is not None:
        gen_kwargs["d1"] = args.d1

    if args.xor_depth_min is not None and args.xor_depth_max is not None:
        gen_kwargs["xor_depth"] = (args.xor_depth_min, args.xor_depth_max)

    if args.xor_traps_min is not None and args.xor_traps_max is not None:
        gen_kwargs["xor_traps"] = (args.xor_traps_min, args.xor_traps_max)

    gen_kwargs["margin_range"] = (args.margin_min, args.margin_max)

    # ---- output dir ----
    out_dir = pathlib.Path(args.out_dir) / f"difficulty_{args.difficulty}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "data.jsonl"

    # ---- run generation ----
    env = GraphCrossEnv()

    start = time.time()
    print("Generating GraphCross data...")

    data_list = env.generate(
        num_of_questions=args.num_of_data,
        max_attempts=args.max_attempts,
        difficulty=args.difficulty,
        **gen_kwargs
    )

    elapsed = time.time() - start
    print(f"Generated {len(data_list)} samples in {elapsed:.2f}s")

    with open(out_file, "w", encoding="utf-8") as f:
        for d in data_list:
            f.write(json.dumps(d.to_json(), ensure_ascii=False) + "\n")

    print(f"Saved to {out_file}")
