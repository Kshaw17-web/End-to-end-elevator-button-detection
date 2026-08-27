FLOOR_VOCAB = set([str(i) for i in range(1, 31)] + ["B1", "B2", "B3"])


def normalize(text):
    return "".join(c for c in text.upper().strip() if c.isalnum())


def select(candidates):
    valid = [(normalize(t), c) for t, c in candidates if normalize(t) in FLOOR_VOCAB]
    if not valid:
        return "UNREADABLE", 0.0
    tally = {}
    for label, conf in valid:
        if label not in tally:
            tally[label] = {"max_conf": conf, "count": 0}
        tally[label]["count"] += 1
        tally[label]["max_conf"] = max(tally[label]["max_conf"], conf)
    ranked = sorted(
        tally.items(),
        key=lambda x: x[1]["max_conf"] + x[1]["count"] * 0.1,
        reverse=True,
    )
    best_label, best_meta = ranked[0]
    return best_label, round(best_meta["max_conf"], 4)
