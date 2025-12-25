import random
import sqlite3
from collections import Counter

DB = "loop.db"
USER_ID = 5274709649
N = 20000
EXPLORATION = 0.2


def load_prefs():
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute(
        """
        SELECT style, alpha, beta
        FROM user_style_pref
        WHERE user_id = ?
    """,
        (USER_ID,),
    )
    rows = cur.fetchall()
    con.close()
    return {style: (float(a), float(b)) for style, a, b in rows}


prefs = load_prefs()
styles = list(prefs.keys())
if not styles:
    raise SystemExit("No prefs found; check user_id and database path.")


def weighted_choice(items, weights):
    total = sum(weights)
    if total <= 0:
        return random.choice(items)
    target = random.random() * total
    cumulative = 0.0
    for item, weight in zip(items, weights):
        if weight <= 0:
            continue
        cumulative += weight
        if target <= cumulative:
            return item
    return items[-1]


def pick_style_pair():
    use_explore = random.random() < EXPLORATION
    if use_explore:
        weights = [1.0] * len(styles)
    else:
        weights = [
            random.betavariate(*prefs.get(style, (1.0, 1.0)))
            for style in styles
        ]
    first = weighted_choice(styles, weights)
    second = weighted_choice(styles, weights)
    canonical = tuple(sorted([first, second]))
    display = [canonical[0], canonical[1]]
    random.shuffle(display)
    return tuple(display), canonical


pair_counts = Counter()
single_counts = Counter()
left_counts = Counter()
right_counts = Counter()

for _ in range(N):
    display, canonical = pick_style_pair()
    pair_counts[canonical] += 1
    single_counts[display[0]] += 1
    single_counts[display[1]] += 1
    left_counts[display[0]] += 1
    right_counts[display[1]] += 1

print("Singles (share of 2*N picks):")
total = 2 * N
for style, count in single_counts.most_common():
    print(f"{style:10s} {count/total:.3%}")

print("\nTop pairs (unordered):")
for (a, b), count in pair_counts.most_common(10):
    print(f"{a} + {b}: {count/N:.3%}")

print("\nOVAL+OVAL:", pair_counts.get(("OVAL", "OVAL"), 0) / N)

print("\nLeft/Right share (should be close):")
for style in single_counts:
    left = left_counts[style] / N
    right = right_counts[style] / N
    print(f"{style:10s} L={left:.3%} R={right:.3%}")
