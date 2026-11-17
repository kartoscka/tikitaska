"""A made-up match in the same shapes as src/features.load_inputs(). Reads no match data:
everything here is simulated. Used by the tests to run the whole pipeline without the real data.

What it imitates (from the data dictionary, not from the data): 46 slots (22 home, 24 away) that
are tracklets, not people; ball positions normalised 0-1 with NaN while the tracker has lost it;
long ball gaps during stoppages and a few in live play; a declared speed column in an odd unit;
one homography per frame from a panning, zooming camera; Smart Tagging phases and set pieces.
"""

import numpy as np

FPS = 25
PITCH = (105.3, 68.0)
N_FRAMES = 15000
HALVES = {1: (300, 7199), 2: (7800, 14699)}
SLOTS = {0: 22, 1: 24}                    # slots per team, as in the real metadata
HOME_NAME, AWAY_NAME = "Barcelona", "Athletic Club"
DECLARED_SPEED_FACTOR = 0.42              # the real speed column matches no unit; neither does this

# 1-4-4-2 for a team attacking towards +x, metres.
FORMATION = np.array([[5, 34], [25, 10], [22, 26], [22, 42], [25, 58], [48, 12], [45, 28],
                      [45, 40], [48, 56], [68, 26], [68, 42]], dtype=float)
CODES_PLAY = ["BUILD UP", "PROGRESSION", "BALL IN FINAL THIRD", "CREATING CHANCES",
              "ATTACKING TRANSITION", "BALL IN THE BOX"]
CODES_DEF = ["DEFENDING IN MIDDLE THIRD", "DEFENDING IN DEFENSIVE THIRD"]


def _camera(pan_x, focal):
    """Normalised-pitch -> 1920x1080 homography of a camera on the near side at halfway."""
    L, W = PITCH
    c = np.array([L / 2, W + 45.0, 26.0])
    look = np.array([pan_x, W * 0.45, 0.0])
    z = look - c
    z /= np.linalg.norm(z)
    x = np.array([1.0, 0, 0]) - z[0] * z
    x /= np.linalg.norm(x)
    y = np.cross(x, z)                      # image y grows towards the camera (down)
    R = np.stack([x, y, z])
    K = np.array([[focal, 0, 960], [0, focal, 540], [0, 0, 1.0]])
    H = K @ np.column_stack([R[:, 0], R[:, 1], -R @ c])
    H = H @ np.diag([L, W, 1.0])
    return H / H[2, 2]


class _Sim:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.n = N_FRAMES
        self.half = np.zeros(self.n, np.int8)
        for h, (a, b) in HALVES.items():
            self.half[a:b + 1] = h
        self.pos = np.zeros((self.n, 22, 2))           # 0-10 home, 11-21 away, metres
        self.ball = np.full((self.n, 2), np.nan)        # true ball
        self.atd_ball = np.full((self.n, 2), np.nan)
        self.label = np.full(self.n, "X", dtype="<U1")
        self.events = []

    def anchors(self, f, ball):
        """Formation targets for both teams, shifted with the ball. Home attacks +x in half 1."""
        L, W = PITCH
        flip = self.half[f] == 2
        home = FORMATION.copy()
        away = FORMATION.copy()
        away[:, 0] = L - away[:, 0]
        if flip:
            home[:, 0], away[:, 0] = L - home[:, 0], L - away[:, 0]
        both = np.concatenate([home, away])
        both[1:11, 0] += 0.35 * (ball[0] - L / 2)
        both[12:, 0] += 0.35 * (ball[0] - L / 2)
        both[:, 1] += 0.25 * (ball[1] - W / 2)
        return both

    def attack_sign(self, f, team):
        s = 1 if team == 0 else -1
        return -s if self.half[f] == 2 else s

    def tag(self, code, a, b, team=None, kind=None):
        tags = {"Team": {0: HOME_NAME, 1: AWAY_NAME}.get(team, "N/A")}
        if kind:
            tags["Type"] = kind
        tags["Half"] = "1st Half" if self.half[min(a, self.n - 1)] == 1 else "2nd Half"
        self.events.append({"code": code, "start_frame": int(a), "end_frame": int(b),
                            "start_s": a / FPS, "end_s": b / FPS, "tags": tags})

    def run(self):
        rng = self.rng
        L, W = PITCH
        p = self.anchors(HALVES[1][0], (L / 2, W / 2))
        vel = np.zeros((22, 2))
        ball = np.array([L / 2, W / 2])
        f = 0
        state, team, carrier = "dead", 0, 9
        until, flight = HALVES[1][0], None
        restart = ("Kick Off (Start)", None)
        phase_start, lost_until = None, -1
        while f < self.n:
            h = self.half[f]
            if h == 0:                                  # before the match, halftime, after
                p += (self.anchors(max(f, 1), (L / 2, W / 2)) - p) * 0.02
                self.pos[f] = p
                f += 1
                if f in (HALVES[1][0], HALVES[2][0]):
                    ball = np.array([L / 2, W / 2])
                    state, until = "dead", f + 50
                    team, carrier = (0, 9) if f == HALVES[1][0] else (1, 20)
                    restart = ("Kick Off (Start)", None)
                continue
            target = self.anchors(f, ball)
            if state == "live":
                if flight is not None:
                    a, b, t0, t1, receiver = flight
                    k = (f - t0) / max(t1 - t0, 1)
                    ball = a + (b - a) * min(k, 1.0)
                    if f >= t1:
                        flight = None
                        if receiver is None:            # ball went out
                            state, until = "dead", f + rng.integers(150, 450)
                            self.tag_phase_end(phase_start, f, team)
                            phase_start = None
                            restart = self.restart_kind(ball, team, f)
                        else:
                            carrier = receiver
                            new_team = 0 if receiver < 11 else 1
                            if new_team != team:
                                self.tag_phase_end(phase_start, f, team)
                                team, phase_start = new_team, f
                                self.tag("ATTACKING TRANSITION", f, f + 150, team)
                else:
                    # Clipped so a long carry stops at the lines instead of leaving the pitch.
                    target[carrier] = np.clip(
                        ball + 0.8 * np.array([self.attack_sign(f, team), 0]), 2, [L - 2, W - 2])
                    ball = np.clip(p[carrier] + np.array([0.7 * self.attack_sign(f, team), 0.2]),
                                   0.5, [L - 0.5, W - 0.5])
                    if rng.random() < 1 / 45:
                        flight = self.pass_ball(f, ball, team, p)
                opp = slice(11, 22) if team == 0 else slice(0, 11)
                presser = np.argmin(np.hypot(*(p[opp] - ball).T)) + opp.start
                target[presser] = ball
                if f > lost_until and rng.random() < 1 / 2500:
                    lost_until = f + rng.integers(125, 260)     # tracker loses a live ball
                self.label[f] = "H" if team == 0 else "A"
                if phase_start is None:
                    phase_start = f
                limit = 7.5
            else:                                       # dead ball
                self.label[f] = "D"
                limit = 1.6
                if f >= until:
                    kind, by = restart
                    self.tag("SET PIECES", f - 50, f + 75, by, kind)
                    state, flight = "live", None
                    if not kind.startswith("Kick Off"):
                        team = int(by) if by is not None else 1 - team
                    if kind and kind.startswith("Kick Off"):
                        ball = np.array([L / 2, W / 2])
                        carrier = 9 if team == 0 else 20
                    else:
                        carrier = int(np.argmin(np.hypot(*(p[0:11] if team == 0
                                                           else p[11:]).T - ball[:, None])))
                        carrier += 0 if team == 0 else 11
                        ball = np.clip(ball, [0.5, 0.5], [L - 0.5, W - 0.5])
                    phase_start = f
            step = np.clip(target - p, -limit / FPS * 3, limit / FPS * 3)
            vel = 0.85 * vel + 0.15 * step * FPS / 3
            sp = np.hypot(*vel.T)
            vel *= np.minimum(1, limit / np.maximum(sp, 1e-9))[:, None]
            p = p + vel / FPS + rng.normal(0, 0.01, p.shape)
            self.pos[f] = p
            self.ball[f] = ball
            if state == "live" and f > lost_until:
                self.atd_ball[f] = ball + rng.normal(0, 0.15, 2)
            f += 1
        return self

    def pass_ball(self, f, ball, team, p):
        rng = self.rng
        L, W = PITCH
        mates = list(range(11)) if team == 0 else list(range(11, 22))
        sign = self.attack_sign(f, team)
        if rng.random() < 0.06:                          # out of play
            end = ball + np.array([sign * rng.uniform(10, 30), rng.choice([-1, 1]) * 40])
            end = np.clip(end, [-3, -3], [L + 3, W + 3])
            if 0 <= end[1] <= W:
                end[1] = -2 if end[1] < W / 2 else W + 2
            receiver = None
        else:
            forward = [m for m in mates if (p[m, 0] - ball[0]) * sign > -5]
            receiver = int(rng.choice(forward or mates))
            end = p[receiver].copy()
            if rng.random() < 0.18:                      # intercepted
                opp = range(11, 22) if team == 0 else range(11)
                mid = (ball + end) / 2
                receiver = min(opp, key=lambda o: np.hypot(*(p[o] - mid)))
                end = p[receiver].copy()
        dist = np.hypot(*(end - ball))
        speed = rng.uniform(11, 22)
        return ball.copy(), end, f, f + max(3, int(dist / speed * FPS)), receiver

    def restart_kind(self, ball, team, f):
        L, W = PITCH
        other = 1 - team
        if ball[1] < 0 or ball[1] > W:
            return "Throw-in", None
        if self.rng.random() < 0.15:
            return "Free Kick", other
        attacking_end = (ball[0] > L / 2) == (self.attack_sign(f, team) > 0)
        return ("Goal Kick", other) if attacking_end else ("Corner Kick", other)

    def tag_phase_end(self, start, f, team):
        if start is None or f - start < 50:
            return
        mid = (start + f) // 2
        code = CODES_PLAY[int(self.rng.integers(len(CODES_PLAY)))]
        self.tag(code, start, min(f, start + 400), team)
        self.tag(CODES_DEF[int(self.rng.integers(len(CODES_DEF)))], mid, f, 1 - team)


def make_inputs(seed=0):
    """Same keys as src/features.load_inputs(), plus away_name, fps, pitch, labels."""
    sim = _Sim(seed).run()
    rng = np.random.default_rng(seed + 1)
    n = sim.n
    L, W = PITCH
    n_slots = SLOTS[0] + SLOTS[1]
    team_of_slot = np.array([0] * SLOTS[0] + [1] * SLOTS[1], dtype=np.int8)
    players = np.full((n, n_slots, 3), np.nan, dtype=np.float32)

    # Slots are tracklets: each player starts on one slot and moves to a free one now and then,
    # with a short blank in between. A few spare slots shadow a player for a moment (duplicates).
    speed = np.zeros((n, 22))
    speed[1:] = np.hypot(*(np.diff(sim.pos, axis=0)).transpose(2, 0, 1)) * FPS
    def put(slot, pid, a, b):
        players[a:b, slot, 0] = sim.pos[a:b, pid, 0] / L
        players[a:b, slot, 1] = sim.pos[a:b, pid, 1] / W
        players[a:b, slot, 2] = speed[a:b, pid] * DECLARED_SPEED_FACTOR

    base = {0: 0, 1: SLOTS[0]}
    free = {t: list(range(base[t] + 11, base[t] + SLOTS[t])) for t in (0, 1)}
    slot_of = {(t, k): base[t] + k for t in (0, 1) for k in range(11)}
    start = {key: 0 for key in slot_of}
    changes = []
    for key in slot_of:
        t = int(rng.integers(25 * 20, 25 * 200))
        while t < n:
            changes.append((t, key))
            t += int(rng.integers(25 * 40, 25 * 240))
    for t, key in sorted(changes):                       # in time order, so a freed slot is
        team, k = key                                    # only reused after it was freed
        put(slot_of[key], k + 11 * team, start[key], t)
        new = free[team].pop(int(rng.integers(len(free[team]))))
        free[team].append(slot_of[key])
        slot_of[key], start[key] = new, t + int(rng.integers(0, 20))
    for (team, k), slot in slot_of.items():
        put(slot, k + 11 * team, start[(team, k)], n)
    for _ in range(12):                                  # duplicates
        f = int(rng.integers(0, n - 100))
        pid = int(rng.integers(22))
        team = 0 if pid < 11 else 1
        b = f + int(rng.integers(25, 100))
        busy = np.isfinite(players[f:b, :, 0]).any(axis=0)
        spare = [s for s in np.flatnonzero(team_of_slot == team) if not busy[s]]
        if not spare:
            continue
        s = spare[0]
        players[f:b, s, 0] = (sim.pos[f:b, pid, 0] + 0.6) / L
        players[f:b, s, 1] = sim.pos[f:b, pid, 1] / W
        players[f:b, s, 2] = speed[f:b, pid] * DECLARED_SPEED_FACTOR
    outside = sim.half == 0
    players[outside] = np.where(rng.random((outside.sum(), n_slots, 1)) < 0.9, np.nan,
                                players[outside])

    ball = np.full((n, 3), np.nan, dtype=np.float32)
    ball[:, 0] = sim.atd_ball[:, 0] / L
    ball[:, 1] = sim.atd_ball[:, 1] / W
    bspeed = np.zeros(n)
    bspeed[1:] = np.hypot(*np.diff(sim.atd_ball, axis=0).T) * FPS
    ball[:, 2] = np.where(np.isfinite(ball[:, 0]), bspeed * DECLARED_SPEED_FACTOR, np.nan)
    for f in rng.integers(0, n, 8):                      # one-frame jumps
        if np.isfinite(ball[f, 0]):
            ball[f, 0] += 0.2

    pan = np.full(n, L / 2)
    for f in range(1, n):
        target = sim.ball[f, 0] if np.isfinite(sim.ball[f, 0]) else pan[f - 1]
        pan[f] = pan[f - 1] + 0.03 * (np.clip(target, 18, L - 18) - pan[f - 1])
    focal = 2600 + 400 * np.sin(np.arange(n) / (25 * 40))
    matrices = np.stack([_camera(pan[f], focal[f]) for f in range(n)])
    error = np.abs(rng.normal(0.25, 0.08, n)) + np.where(sim.half == 0, 1.2, 0)

    return {"ball": ball, "players": players, "team_of_slot": team_of_slot, "half": sim.half,
            "matrices": matrices, "hom_valid": np.ones(n, bool), "hom_error": error,
            "events": sorted(sim.events, key=lambda e: e["start_frame"]),
            "home_name": HOME_NAME, "away_name": AWAY_NAME, "meta_path": None, "fps": FPS,
            "pitch": PITCH, "n": n, "halves": dict(HALVES),
            "labels": {"synthetic_truth": sim.label}}
