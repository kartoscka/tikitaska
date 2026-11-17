"""rule_learnt_ensemble: the extended rules plus one learnt vote.

What it adds. The extended rules (extended_rules.py) add up the votes of a few hand-made rules
(reach, gap, out, restart, pass2, phase) and then pick the best label path with a switching cost.
Here one more vote is added to that sum: a small logistic regression (a straight-line classifier,
no trees, no network) looks at the same rule votes and at some player-movement numbers of the frame,
and says how likely H, A and D are. Its three probabilities vote with a weight of 0.5 (VOTE_WEIGHT).
Everything else is the extended rules, unchanged: with the vote switched off, the labels are exactly
those of extended_rules.py.

What the classifier looks at (FEATURE_NAMES, in the fitted block below): whether the ball is seen,
its speed, how long it has been lost, the distance of each team to it, how many players are near,
the team widths, the Smart Tagging restart tag, the rules' own votes, how fast the players speed up
or slow down together, whether the players stand in a corner or goal-kick shape, how fast each team
closes on the ball, and moving averages and 1 s lags of some of these.

Where the numbers come from. The fitted numbers below come only from Metrica's two public sample
games 1 and 2 (public data), each at three levels of ball loss (0, 15 and 30 %), with the restart tags
thinned like BAR-ATH's and no phase tags, every 5th frame that has a label. The labels are built from
Metrica's public event logs (external/metrica_samples/). No BAR-ATH frame, label or prediction is
used; BAR-ATH's feature table is only used by the training script to copy how long its ball gaps are
and how many restart tags it has. src/training/train_rule_learnt_ensemble.py rewrites the block
between BEGIN FITTED and END FITTED; it gives the same file every time.

predict() needs the width columns (x_h_width, x_a_width) and the player-movement columns (pd_*). If
the feature table does not have them it reads the raw tracking once (src/features.load_inputs) and
computes them. `inputs` (a dict with players, team_of_slot and half) can be given instead of reading
the raw tracking.

Uses numpy and pandas only (plus extended_rules and base_rules).
"""

import warnings

import numpy as np
import pandas as pd

from src.models import extended_rules as N
from src.models import base_rules as _core
from src.models.base_rules import _col, _runs, _passes, _decode, H, A, D, LETTERS, FPS

VOTE_WEIGHT = 0.5                    # the weight of the learnt vote (the other weights are extended_rules')

# --- BEGIN FITTED (written by src/training/train_rule_learnt_ensemble.py; do not edit) ---
N_TRAIN = 171702
FEATURE_NAMES = [
    'known',
    'ball_speed',
    'gap_s',
    'out',
    'edge',
    'h_dist',
    'a_dist',
    'dist_diff',
    'h_n2',
    'h_n5',
    'h_speed',
    'h_nspeed',
    'h_offcam',
    'a_n2',
    'a_n5',
    'a_speed',
    'a_nspeed',
    'a_offcam',
    'width_ratio',
    'width_ratio_gap',
    'tag_sp',
    'tag_h',
    'tag_a',
    'v_reach_H',
    'v_reach_A',
    'v_gap_D',
    'v_out_D',
    'v_restart_H',
    'v_restart_A',
    'v_restart_D',
    'v_pass2_H',
    'v_pass2_A',
    'pd_speed_all',
    'pd_acc_all',
    'pd_dec_share',
    'pd_acc_share',
    'pd_h_acc',
    'pd_a_acc',
    'pd_corner_h',
    'pd_corner_a',
    'pd_gk_h',
    'pd_gk_a',
    'pd_h_close',
    'pd_a_close',
    'pd_h_align',
    'pd_a_align',
    'close_diff',
    'dist_diff_m5',
    'close_diff_m5',
    'pd_h_acc_m5',
    'pd_a_acc_m5',
    'pd_speed_all_m5',
    'pd_dec_share_m5',
    'ball_speed_m5',
    'known_m5',
    'dist_diff_m25',
    'close_diff_m25',
    'pd_h_acc_m25',
    'pd_a_acc_m25',
    'pd_speed_all_m25',
    'pd_dec_share_m25',
    'ball_speed_m25',
    'known_m25',
    'dist_diff_lag25',
    'known_lag25',
    'dec_share_lag25',
    'third_def_h',
    'third_att_h',
    'ball_vs_h_block',
    'ball_vs_a_block',
]
MEAN = [
    0.5206637080523232,
    3.900998552525354,
    4.6578017734795205,
    0.0042107837998392565,
    12.635560941091278,
    1.4749788436632896,
    1.5093707439031778,
    -0.03439190030722607,
    0.3616818980078077,
    0.7584872314007987,
    1.7406895381314242,
    2.929346943635937,
    0.008503104215443036,
    0.3395899282989743,
    0.764102506739449,
    1.7168975155417208,
    2.9777301664685933,
    0.003150807794900467,
    0.00505929891776123,
    0.0025389026871922077,
    0.025858755285319914,
    0.011479190690848098,
    0.014379564594471818,
    0.17489115426320134,
    0.16979192991772296,
    0.22687195255215367,
    0.08809449178951236,
    0.004368032987385121,
    0.00541636090435755,
    0.2763916553097809,
    0.05070412691756648,
    0.045410070936855715,
    1.72879372213697,
    9.153218064867065e-05,
    0.02784980312944319,
    0.026758958544833067,
    0.00012803133860429317,
    4.662777531032492e-05,
    0.008805954502568403,
    0.003337177202362232,
    0.11082573295593529,
    0.10668483768389418,
    0.06511429599737097,
    0.09274414444875105,
    0.4682081978041719,
    0.457158998168849,
    -0.027629848598438804,
    -0.03454860411628989,
    -0.025033086188110183,
    6.341845242190804e-05,
    -1.7719963030517433e-05,
    1.7287737154597593,
    0.0278529227250523,
    3.9043152688592255,
    0.5207266078006575,
    -0.034291380695492875,
    -0.03336379122664986,
    0.00011347250510290805,
    1.0671796952247852e-05,
    1.7288184104653992,
    0.027860454205185275,
    3.90517516264567,
    0.5207615519842804,
    -0.03439190030722607,
    0.520754694999767,
    0.02785467163568916,
    0.13517023680562837,
    0.12392983191809065,
    7.11745186695806,
    -6.47197957296449,
]
STD = [
    0.49957282869438163,
    5.414369796189515,
    4.918976220998549,
    0.06475378830316836,
    6.694382121774293,
    0.720655897150049,
    0.7365251310797757,
    1.172450808607291,
    0.49840490440901386,
    0.6855733555003642,
    0.938437519848978,
    1.833527680607331,
    0.09181939574043194,
    0.49160321936003143,
    0.7307613921580941,
    0.9579633814171326,
    1.8228242117537583,
    0.056043556321294874,
    0.3409310367676341,
    0.2610349596003669,
    0.15871383071561113,
    0.1065242642400837,
    0.11904953891768305,
    0.3256218723088366,
    0.32168264820806486,
    0.41792032715008615,
    0.1562414194321108,
    0.05432478917801814,
    0.0594189356961824,
    0.6070496559357579,
    0.21939284042830537,
    0.2082018165013817,
    0.9195516649830363,
    0.3645759352569205,
    0.057136970858681156,
    0.05597930115745951,
    0.415112053346894,
    0.40096264244260027,
    0.09342595821224163,
    0.05767183411924562,
    0.3139162147302142,
    0.30871213628892324,
    3.501244520118164,
    3.6782862068256854,
    0.6174734589445068,
    0.6211799774325305,
    3.808372508967471,
    1.1685352833142444,
    3.5499902250530417,
    0.40978423290765514,
    0.3972655526956779,
    0.9192102992919373,
    0.056217929911159696,
    5.338348874795819,
    0.4982964663584976,
    1.1140752006382868,
    2.7297652231255256,
    0.3600058281253713,
    0.3558724340448374,
    0.9136005952892475,
    0.05128853367136271,
    4.871028930280741,
    0.4926960744549672,
    1.172450808607291,
    0.49956905692404874,
    0.05714077815776837,
    0.34190531421323134,
    0.32950148509363614,
    15.07562663355198,
    14.959782651130622,
]
FLAG_COLS = [
    4,
    5,
    6,
    7,
    8,
    9,
    11,
    13,
    14,
    16,
    33,
    34,
    35,
    36,
    37,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    52,
    55,
    56,
    57,
    58,
    60,
    63,
    64,
    65,
    68,
    69,
]
W = [
    [-0.2446258246587753, -0.09372814031602374, 0.33835396497526116],
    [0.047031549888182735, -0.03584142124175855, -0.011190128646315144],
    [-0.23227503974443514, 0.12606998729831742, 0.10620505244608708],
    [-0.05770097145809802, 0.06362105300695686, -0.005920081548867376],
    [0.002182895923383896, 0.023288656832707033, -0.025471552756091673],
    [-0.20190318457894882, -0.20403085412106595, 0.405934038700132],
    [-0.36511975369049404, -0.21373476614988837, 0.5788545198405601],
    [0.1052642839725436, 0.00885762118114684, -0.11412190515358188],
    [0.11406496592221609, -0.1435464870990746, 0.0294815211768267],
    [0.13761055025092164, -0.04802681847975471, -0.08958373177124915],
    [-0.36880953305675657, 0.6048177517592492, -0.23600821870279465],
    [-0.019812082881298057, 0.05528240752901849, -0.03547032464748531],
    [-0.061685306626757634, 0.039865029794974145, 0.021820276831797932],
    [-0.11154049327045536, 0.08711976917476275, 0.02442072409567321],
    [-0.025643798726097844, 0.005373733715942508, 0.020270065010049687],
    [0.5623498822650699, -0.4484936256838612, -0.1138562565816265],
    [-0.01647011189398522, 0.035709330766259505, -0.019239218872216256],
    [-0.059569063122013904, 0.030251982270083636, 0.029317080851960633],
    [0.6161078357185819, -0.9111101069924415, 0.29500227127280576],
    [-0.23400023980868345, 0.3874332242874691, -0.1534329844789371],
    [0.003017796497772968, 0.01867264945317233, -0.02169044595083615],
    [0.07808954534686748, -0.06659021452501428, -0.011499330821837845],
    [-0.06585044672770786, 0.08447812081958175, -0.01862767409178171],
    [0.27890082576770525, -0.37608879811547347, 0.09718797234786666],
    [-0.5452587736624281, 0.23473704439607443, 0.31052172926652527],
    [-0.276983340379831, -0.5763813548403441, 0.8533646952200858],
    [0.030974021444487174, -0.1472859293964934, 0.1163119079515701],
    [0.017138521105281735, 0.023727004408186054, -0.04086552551344373],
    [-0.0382549278902574, 0.04170692122069016, -0.0034519933305536873],
    [-0.05589128444920329, -0.10418393804384204, 0.16007522249318093],
    [0.17220176091409198, -0.21514002016628336, 0.04293825925218654],
    [-0.30362978007386404, 0.21249504980112324, 0.0911347302729388],
    [0.10268503961232842, 0.07422867877359596, -0.17691371838574146],
    [0.028566533574160287, 0.030932949284698443, -0.05949948285868482],
    [0.07739872364086374, 0.024959260794701187, -0.10235798443547132],
    [0.09417588723522076, 0.1051762753639941, -0.1993521625991059],
    [0.024626246377584347, -0.02467036930615188, 4.412292862722981e-05],
    [0.05626671566797397, 0.05897322110224845, -0.11523993677028099],
    [-0.07602706784982682, -0.042104750914800655, 0.11813181876472727],
    [0.01258602054354204, -0.09469877606698754, 0.08211275552358954],
    [-0.2057872754801477, -0.10217118473350097, 0.3079584602137518],
    [0.1830724296520724, -0.35067091458978117, 0.16759848493741558],
    [-0.01337484718223011, -0.050816062671331305, 0.06419090985354119],
    [-0.08295779324805964, 0.028062795977490684, 0.0548949972706026],
    [-0.17942132651855508, 0.1997358741601607, -0.020314547641565265],
    [0.15365825774737532, -0.17521438053104557, 0.021556122783618785],
    [0.0678280845028846, -0.07382211238968328, 0.00599402788679108],
    [0.35043041375836403, -0.334719134456116, -0.015711279302165548],
    [-0.037331067676646554, 0.07092431749601644, -0.03359324981940768],
    [-0.0388250635048531, 0.002505785660376892, 0.036319277843982145],
    [-0.0623686792407959, -0.08858230043210708, 0.1509509796728489],
    [-0.1296105397258256, -0.04304362387736158, 0.17265416360288635],
    [-0.034059175813820855, 0.029998265200593288, 0.004060910613201542],
    [0.012052722970898686, 0.06464880344891792, -0.0767015264198046],
    [0.6273518339349367, 0.25381690046101685, -0.881168734395781],
    [-0.8484415112903736, 0.8147885226287234, 0.03365298866172176],
    [-0.16638495808441728, 0.16198708080572102, 0.004397877278634517],
    [0.027476962688942147, -0.07744796078557332, 0.0499709980967535],
    [-0.01501284600838221, 0.05009097974626547, -0.0350781337380886],
    [-0.03091999900932827, 0.016582289723537574, 0.014337709285683793],
    [0.02442777695522384, 0.013339469698278084, -0.03776724665359715],
    [0.21949526144827403, 0.16483232063120826, -0.3843275820798374],
    [0.12281597029551641, 0.37133741452828517, -0.49415338482347565],
    [-0.31099002627200306, 0.3091858354689994, 0.0018041908030425814],
    [-0.2625498565141241, -0.25106170940566447, 0.5136115659195862],
    [0.02709558217194806, 0.03372693732600209, -0.060822519497868724],
    [0.05742268117115338, -0.08197425398091243, 0.02455157280984684],
    [-0.004958116137375411, 0.049553193901534225, -0.044595077764487866],
    [0.48261387343815404, -0.7715343344615738, 0.28892046102351754],
    [-0.27217684248996227, 0.46196218464349825, -0.18978534215336323],
    [0.10615266955672194, 0.016562224573021395, -0.12271489412923432],
    [0.10615266955672194, 0.016562224573021395, -0.12271489412923432],
    [0.10615266955672194, 0.016562224573021395, -0.12271489412923432],
    [0.10615266955672194, 0.016562224573021395, -0.12271489412923432],
    [0.10615266955672194, 0.016562224573021395, -0.12271489412923432],
    [0.10615266955672194, 0.016562224573021395, -0.12271489412923432],
    [0.12461030661291096, 0.011806393917340499, -0.1364167005297483],
    [0.10615266955672194, 0.016562224573021395, -0.12271489412923432],
    [0.10615266955672194, 0.016562224573021395, -0.12271489412923432],
    [0.10615266955672194, 0.016562224573021395, -0.12271489412923432],
    [-0.11537754971764502, 0.018944199432100163, 0.09643335028554499],
    [-0.16989057760186793, 0.031172177386901204, 0.13871840021496676],
    [-0.16989057760186793, 0.031172177386901204, 0.13871840021496676],
    [-0.11537754971764502, 0.018944199432100163, 0.09643335028554499],
    [-0.11537754971764502, 0.018944199432100163, 0.09643335028554499],
    [-0.0013379788034189768, 0.0804654347067995, -0.07912745590286162],
    [-0.0013379788034189768, 0.0804654347067995, -0.07912745590286162],
    [0.16710235543589871, -0.1334390970269563, -0.03366325840897748],
    [-0.2327691166781352, 0.3476192198750321, -0.1148501031970398],
    [-0.0013379788034189768, 0.0804654347067995, -0.07912745590286162],
    [-0.18544080937241322, -0.25646381870715534, 0.44190462808000086],
    [-0.11954111635309178, -0.07543783362365715, 0.1949789499772138],
    [-0.10914730771102274, 0.029526883576965292, 0.0796204241340575],
    [-0.10914730771102274, 0.029526883576965292, 0.0796204241340575],
    [-0.16366033559524548, 0.041754861531766316, 0.1219054740634792],
    [0.0036447384149663536, -0.1038209180570269, 0.10017617964244178],
    [-0.0636689013366023, 0.026859285760453384, 0.03680961557653294],
    [-0.06852969567777371, 0.053769829460378886, 0.014759866217394817],
    [-0.06852969567777371, 0.053769829460378886, 0.014759866217394817],
    [-0.0873802383474668, 0.07120159909774625, 0.01617863924972057],
    [0.0714253006567424, 0.13235908852145195, -0.20378438917784333],
    [-0.08388553385353303, 0.0714203651734671, 0.012465168680065855],
    [-0.3234585960622461, 0.09687987274523674, 0.22657872331700907],
    [0.10615266955672194, 0.016562224573021395, -0.12271489412923432],
    [0.10615266955672194, 0.016562224573021395, -0.12271489412923432],
]
B = [
    -0.5200344179480332,
    -0.7600402952914931,
    1.2800747132393293,
]
# --- END FITTED ---

PD_COLUMNS = ("pd_speed_all", "pd_acc_all", "pd_dec_share", "pd_acc_share", "pd_h_acc", "pd_a_acc",
              "pd_corner_h", "pd_corner_a", "pd_gk_h", "pd_gk_a",
              "pd_h_close", "pd_a_close", "pd_h_align", "pd_a_align")

# player-movement settings (metres, m/s, m/s^2)
ACC_LAG = 12                         # acceleration over 0.48 s
SHARE_LAG = 25                       # share of players speeding up / slowing down over 1 s ...
SHARE_DELTA = 1.5                    # ... by at least this
BOX_D, BOX_W = 16.5, 40.32           # penalty area
GOAL_D, GOAL_W = 5.5, 18.32          # goal area
CORNER_R = 2.5                       # corner shape: a player this near a corner flag
QUIET = 2.0                          # the mean speed under which play is stopped
MIN_STRETCH = 25                     # a corner or goal-kick shape must last 1 s
STEP = 5                             # training: every 5th graded frame (used by the training script)


PITCH_Y = 68.0                       # pitch width (m), for the team widths


def _features_module():
    """src/features.py (the per-frame feature code)."""
    from src import features as F
    return F


def width_columns(players_norm, team_of_slot):
    """{"x_h_width", "x_a_width"}: std of each team's players across the pitch (m), per frame.
    players_norm: (n, slots, >=2) in 0-1 pitch units, as src/features.load_inputs gives them.
    The team with the ball opens the pitch; the other closes up."""
    y = np.asarray(players_norm)[:, :, 1].astype(np.float64) * PITCH_Y
    tos = np.asarray(team_of_slot)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {"x_h_width": np.nanstd(y[:, tos == 0], axis=1).astype(np.float32),
                "x_a_width": np.nanstd(y[:, tos == 1], axis=1).astype(np.float32)}


# --- the rule votes, one array per rule (unweighted): extended_rules', as separate parts ------------

def parts(df):
    """{rule: (n, 3) votes of H, A, D}, in this order: reach, gap, out, restart, pass2, phase."""
    n = len(df)
    known = np.isfinite(_col(df, "ball_x_s"))
    out = _col(df, "ball_out", 0) > 0
    dist = np.stack([_col(df, "h_nearest_dist"), _col(df, "a_nearest_dist")])
    offcam = np.stack([_col(df, "h_nearest_offcam", 0) > 0, _col(df, "a_nearest_offcam", 0) > 0])
    team_speed = np.fmax(_col(df, "h_mean_speed"), _col(df, "a_mean_speed"))
    slow_teams = np.nan_to_num(team_speed, nan=np.inf) < _core.SLOW_TEAM_MS
    passer = _passes(df, dist)
    P = {}

    # reach: the nearest player of each team (nothing during a pass flight, N.FLIGHT_DAMP)
    reach = np.zeros((n, 3))
    for t in (H, A):
        vote = np.nan_to_num(np.exp(-0.5 * (dist[t] / _core.REACH_M) ** 2))
        vote[offcam[t]] *= _core.OFFCAM_FACTOR
        vote[passer >= 0] *= N.FLIGHT_DAMP
        reach[:, t] = vote
    P["reach"] = reach

    # gap: a lost ball is a dead ball, only for gaps that begin near a line, or when everybody is slow
    gap_len = _col(df, "ball_gap_len", 0)
    gap_vote = np.where(known, 0.0, np.clip(gap_len / (_core.GAP_FULL_S * FPS), 0, 1))
    gap_vote[~_core._near_line_gaps(known, out, _col(df, "ball_edge_dist"), N.NEAR_LINE_M)] = 0.0
    with warnings.catch_warnings():                      # frames where both teams are absent: NaN, as in extended_rules
        warnings.simplefilter("ignore", category=RuntimeWarning)
        all_speed = np.nanmean(np.stack([_col(df, "h_mean_speed"), _col(df, "a_mean_speed")]), axis=0)
    gap_vote[~known & (np.nan_to_num(all_speed, nan=np.inf) < N.SLOW_ALL_MS)] = 1.0
    g = np.zeros((n, 3))
    g[:, D] = gap_vote
    P["gap"] = g

    # out: ball over a line (full), or both teams almost stop (a third)
    ov = np.zeros((n, 3))
    ov[:, D] = np.where(out, 1.0, np.where(slow_teams, 1 / 3, 0.0))
    P["out"] = ov

    # restart: Smart Tagging set pieces and goals (extended_rules' own function)
    P["restart"] = N._restart_votes(df, known, out, slow_teams, N.RESTART_TYPE_WEIGHTS, N.GOAL_WEIGHT)

    # pass2: during a pass flight, the passer's team
    p2 = np.zeros((n, 3))
    for t in (H, A):
        p2[:, t] = (passer == t)
    P["pass2"] = p2

    # phase: Smart Tagging phases (the weights are already in the votes; zero without phase tags)
    P["phase"] = N._phase_votes(df, N.PHASE_WEIGHTS)
    return P


def weights():
    """The weight of each part: extended_rules', phase 1.0 (already inside its votes), the learnt vote 0.5."""
    w = dict(N.WEIGHTS)
    w["phase"] = 1.0
    w["lv"] = VOTE_WEIGHT
    return w


def score_of(P, w):
    """Sum of weight x part, in the order of P (the order of the parts above)."""
    s = 0.0
    for k, v in P.items():
        if w.get(k, 0.0):
            s = s + w[k] * v
    return s if not np.isscalar(s) else np.zeros_like(P["reach"])


# --- the classifier's inputs --------------------------------------------------------------------------

def _c(df, name, fill=np.nan):
    """One column as a float array; a missing column is all `fill`."""
    if name not in df.columns:
        return np.full(len(df), fill)
    return df[name].to_numpy(dtype=float, na_value=np.nan)


def _roll(x, w):
    return pd.Series(x).rolling(w, center=False, min_periods=1).mean().to_numpy()


def _lag(x, k):
    out = np.full_like(x, np.nan)
    out[k:] = x[:-k]
    return out


def features(df, names=None, P=None):
    """(X float32 (n, p), names). names=None: the data decides which of the rule votes are
    used (a vote that is zero on every frame is left out); this is how the training sample is built.
    names=list: exactly these columns, in this order (a vote column the table does not have is zero)."""
    P = parts(df) if P is None else P
    f = {}
    known = np.isfinite(_c(df, "ball_x_s")).astype(float)
    f["known"] = known
    f["ball_speed"] = np.clip(np.nan_to_num(_c(df, "ball_speed"), nan=0.0), 0, 35)
    f["gap_s"] = np.clip(np.nan_to_num(_c(df, "ball_gap_len"), nan=0.0) / 25.0, 0, 10) * (1 - known)
    f["out"] = np.nan_to_num(_c(df, "ball_out"), nan=0.0)
    f["edge"] = np.clip(_c(df, "ball_edge_dist"), 0, 20)
    hd = np.log1p(np.clip(_c(df, "h_nearest_dist"), 0, 60))
    ad = np.log1p(np.clip(_c(df, "a_nearest_dist"), 0, 60))
    f["h_dist"], f["a_dist"], f["dist_diff"] = hd, ad, hd - ad
    for t in ("h", "a"):
        f[t + "_n2"] = _c(df, t + "_n_within_2m")
        f[t + "_n5"] = _c(df, t + "_n_within_5m")
        f[t + "_speed"] = _c(df, t + "_mean_speed")
        f[t + "_nspeed"] = _c(df, t + "_nearest_speed")
        f[t + "_offcam"] = np.nan_to_num(_c(df, t + "_nearest_offcam"), nan=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        wr = np.log(_c(df, "x_h_width")) - np.log(_c(df, "x_a_width"))
    f["width_ratio"] = np.where(np.isfinite(wr), wr, np.nan)
    f["width_ratio_gap"] = np.nan_to_num(f["width_ratio"]) * (1 - known)
    f["tag_sp"] = df["st_set_pieces"].to_numpy(dtype=float)
    side = df["st_set_pieces_side"].astype(str).to_numpy()
    f["tag_h"], f["tag_a"] = (side == "H").astype(float), (side == "A").astype(float)
    for r in ("reach", "gap", "out", "restart", "pass2"):          # the rules' own votes as inputs
        for j, s in enumerate("HAD"):
            if names is not None or np.any(P[r][:, j]):
                f["v_{}_{}".format(r, s)] = P[r][:, j]
    for c in ("pd_speed_all", "pd_acc_all", "pd_dec_share", "pd_acc_share", "pd_h_acc", "pd_a_acc",
              "pd_corner_h", "pd_corner_a", "pd_gk_h", "pd_gk_a"):
        f[c] = _c(df, c)
    for c in ("pd_h_close", "pd_a_close", "pd_h_align", "pd_a_align"):
        f[c] = _c(df, c)
    f["close_diff"] = _c(df, "pd_h_close") - _c(df, "pd_a_close")
    for w in (5, 25):
        for c in ("dist_diff", "close_diff", "pd_h_acc", "pd_a_acc", "pd_speed_all", "pd_dec_share",
                  "ball_speed", "known"):
            f["{}_m{}".format(c, w)] = _roll(f[c], w)
    f["dist_diff_lag25"] = _lag(f["dist_diff"], 25)
    f["known_lag25"] = _lag(known, 25)
    f["dec_share_lag25"] = _lag(f["pd_dec_share"], 25)
    # position: the third of the pitch in home's attacking direction, the ball against each block
    pos = _c(df, "h_attacks_pos_x")
    bx = _c(df, "ball_x_s")
    px = 105.0
    xa = np.where(pos == 1, bx, px - bx)
    f["third_def_h"] = (xa < px / 3).astype(float)
    f["third_att_h"] = (xa > 2 * px / 3).astype(float)
    for t in ("h", "a"):
        cx = _c(df, t + "_centroid_x")
        ca = np.where(pos == 1, cx, px - cx)
        f["ball_vs_{}_block".format(t)] = xa - ca           # metres ahead of the block, in home's direction
    if names is None:
        names = list(f)
    X = np.stack([np.asarray(f[k], dtype=np.float32) for k in names], axis=1)
    return X, list(names)


# --- the player-movement (pd_*) columns, from the cleaned players and the ball columns -----------------

def _nanmean(a, axis=1):
    if a.shape[axis] == 0:
        return np.full(a.shape[0], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _lagged(a, k, half):
    """a[t-k], NaN where t-k crosses a half boundary or half == 0."""
    out = np.full(a.shape, np.nan, dtype=np.float32)
    out[k:] = a[:-k]
    ok = np.zeros(len(a), bool)
    ok[k:] = (half[k:] == half[:-k]) & (half[k:] > 0)
    out[~ok] = np.nan
    return out


def _stretches(mask, minlen=MIN_STRETCH):
    """Keep only runs of True lasting at least minlen frames."""
    out = np.zeros(len(mask), bool)
    for a, b in _runs(mask):
        if b - a >= minlen:
            out[a:b] = True
    return out


def _box(px, py, depth, width, at_x0, PX, PY):
    """Inside a box of the given depth and width at the x = 0 end (at_x0 per row), else at the x = PX end."""
    dx = np.where(at_x0[:, None], px, PX - px)
    with np.errstate(invalid="ignore"):
        return (dx <= depth) & (np.abs(py - PY / 2) <= width / 2)


def pd_columns(players_norm, team_of_slot, half, df):
    """{name: float32 (n,)} of PD_COLUMNS. players_norm: (n, slots, >=2) in 0-1 pitch units (as
    src/features.load_inputs gives them); half: 0 outside the halves; df: the feature table
    (h_attacks_pos_x and the ball columns)."""
    F = _features_module()
    PX, PY = F.PITCH_X, F.PITCH_Y
    n = len(df)
    assert len(players_norm) == n, "players and feature table differ in length"
    half = np.asarray(half)
    assert len(half) == n, "half and feature table differ in length"
    tos = np.asarray(team_of_slot)
    cl = F.clean_players(players_norm)
    att = pd.array(df["h_attacks_pos_x"], dtype="boolean").to_numpy(dtype=float, na_value=np.nan)
    out = {}

    # all players: speed, acceleration, the share of players slowing down / speeding up together
    sp = cl["speed"].astype(np.float32)
    px, py = cl["px"], cl["py"]
    out["pd_speed_all"] = _nanmean(sp)
    acc = (sp - _lagged(sp, ACC_LAG, half)) / (ACC_LAG / FPS)
    out["pd_acc_all"] = _nanmean(acc)
    prev = _lagged(sp, SHARE_LAG, half)
    both = np.isfinite(sp) & np.isfinite(prev)
    cnt = both.sum(axis=1)
    with np.errstate(invalid="ignore"):
        dec = (both & (prev - sp >= SHARE_DELTA)).sum(axis=1)
        inc = (both & (sp - prev >= SHARE_DELTA)).sum(axis=1)
    for name, c in (("pd_dec_share", dec), ("pd_acc_share", inc)):
        out[name] = np.where(cnt >= 6, c / np.maximum(cnt, 1), np.nan)

    # corner and goal-kick shapes
    quiet = np.nan_to_num(out["pd_speed_all"], nan=np.inf) < QUIET
    present = np.isfinite(px) & np.isfinite(py)
    all0 = np.ones(n, bool)
    in0 = _box(px, py, BOX_D, BOX_W, all0, PX, PY) & present       # penalty area at the x = 0 end
    inX = _box(px, py, BOX_D, BOX_W, ~all0, PX, PY) & present      # ... at the x = PX end
    n0, nX = in0.sum(axis=1), inX.sum(axis=1)
    known = np.isfinite(att)
    for t, team in (("h", 0), ("a", 1)):
        cols = np.flatnonzero(tos == team)
        opp = np.flatnonzero(tos != team)
        out["pd_%s_acc" % t] = _nanmean(acc[:, cols])
        tx, ty = px[:, cols], py[:, cols]
        # own goal end: home defends x = 0 when att == 1, away defends x = 0 when att == 0
        defends0 = (att == 1) if t == "h" else (att == 0)

        # corner shape: a player of this team at a corner flag, 5 or more players in that end's box, play quiet
        near0 = np.zeros(n, bool)
        nearX = np.zeros(n, bool)
        for cx, cy in ((0.0, 0.0), (0.0, PY), (PX, 0.0), (PX, PY)):
            with np.errstate(invalid="ignore"):
                hit = (np.hypot(tx - cx, ty - cy) <= CORNER_R).any(axis=1) if len(cols) else near0
            if cx == 0.0:
                near0 = near0 | hit
            else:
                nearX = nearX | hit
        corner = ((near0 & (n0 >= 5)) | (nearX & (nX >= 5))) & quiet
        out["pd_corner_%s" % t] = _stretches(corner)

        # goal-kick shape: the deepest player in the own goal area, nobody of the opponent in the own box
        depth = np.where(defends0[:, None], tx, PX - tx)
        depth = np.where(np.isfinite(depth), depth, np.inf)
        if len(cols):
            k = depth.argmin(axis=1)
            r = np.arange(n)
            dd, yy = depth[r, k], ty[r, k]
            with np.errstate(invalid="ignore"):
                in_goal = np.isfinite(dd) & (dd <= GOAL_D) & (np.abs(yy - PY / 2) <= GOAL_W / 2)
        else:
            in_goal = np.zeros(n, bool)
        if len(opp):
            opp_in = np.where(defends0[:, None], in0[:, opp], inX[:, opp]).any(axis=1)
        else:
            opp_in = np.zeros(n, bool)
        out["pd_gk_%s" % t] = _stretches(in_goal & ~opp_in & quiet & known)

    # the ball: closing speed of each team's 3 nearest players, and how well the team moves with the ball
    bx = df["ball_x_s"].to_numpy(dtype=np.float64)
    by = df["ball_y_s"].to_numpy(dtype=np.float64)
    bvx = df["ball_vx"].to_numpy(dtype=np.float64)
    bvy = df["ball_vy"].to_numpy(dtype=np.float64)
    ball_ok = np.isfinite(bx) & np.isfinite(by) & np.isfinite(bvx) & np.isfinite(bvy)
    bspeed = np.hypot(bvx, bvy)
    qx, qy = px.astype(np.float64), py.astype(np.float64)
    vx, vy = cl["vx"].astype(np.float64), cl["vy"].astype(np.float64)
    r = np.arange(n)[:, None]
    for t, team in (("h", 0), ("a", 1)):
        cols = np.flatnonzero(tos == team)
        res = np.full(n, np.nan)
        if len(cols):
            dx, dy = bx[:, None] - qx[:, cols], by[:, None] - qy[:, cols]
            d = np.hypot(dx, dy)
            with np.errstate(invalid="ignore", divide="ignore"):
                clos = ((vx[:, cols] - bvx[:, None]) * dx + (vy[:, cols] - bvy[:, None]) * dy) / d
            dkey = np.where(np.isfinite(d), d, np.inf)
            idx = np.argsort(dkey, axis=1, kind="stable")[:, :min(3, len(cols))]
            sel = np.where(np.isfinite(dkey[r, idx]), clos[r, idx], np.nan)
            res = _nanmean(sel)
            res[~ball_ok] = np.nan
        out["pd_%s_close" % t] = res
        mvx, mvy = _nanmean(vx[:, cols]), _nanmean(vy[:, cols])
        ts = np.hypot(mvx, mvy)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = (mvx * bvx + mvy * bvy) / (ts * bspeed)
        out["pd_%s_align" % t] = np.where(ball_ok & (bspeed >= 1.0) & (ts >= 0.5), cos, np.nan)

    res = {}
    for k in PD_COLUMNS:
        res[k] = np.asarray(out[k], dtype=np.float32)
        assert len(res[k]) == n, k
    return res


_RAW = {}   # BAR-ATH's raw tracking inputs, read once per session


def _raw_inputs():
    if "inputs" not in _RAW:
        _RAW["inputs"] = _features_module().load_inputs()
    return _RAW["inputs"]


def with_tracking_columns(df, inputs=None):
    """df with x_h_width / x_a_width and the pd_* columns. Those the table has are kept; the others are
    computed from the raw tracking (inputs: dict with players, team_of_slot, half; default
    src/features.load_inputs, read once)."""
    need_w = not ("x_h_width" in df.columns and "x_a_width" in df.columns)
    need_pd = [c for c in PD_COLUMNS if c not in df.columns]
    if not need_w and not need_pd:
        return df
    inp = inputs if inputs is not None else _raw_inputs()
    assert len(inp["players"]) == len(df), "the raw tracking and the feature table differ in length"
    df = df.copy()
    if need_w:
        for name, col in width_columns(inp["players"], inp["team_of_slot"]).items():
            df[name] = col
    if need_pd:
        cols = pd_columns(inp["players"], inp["team_of_slot"], inp["half"], df)
        for name in need_pd:
            df[name] = cols[name]
    return df


# --- the classifier ---------------------------------------------------------------------------------

def standardise(X, mean, std, flag_cols):
    """(X - mean) / std with NaN as 0, then a 0/1 'was missing' column for each column in flag_cols."""
    X = np.asarray(X, dtype=np.float64)
    nan = np.isnan(X)
    Z = (X - mean) / std
    Z[nan] = 0.0
    if len(flag_cols):
        Z = np.hstack([Z, nan[:, flag_cols].astype(np.float64)])
    return Z


def softmax(L):
    L = L - L.max(axis=1, keepdims=True)
    np.exp(L, out=L)
    return L / L.sum(axis=1, keepdims=True)


def predict_proba(X):
    """(n, 3) probabilities of H, A, D from the fitted numbers."""
    Z = standardise(X, np.array(MEAN, dtype=np.float64), np.array(STD, dtype=np.float64),
                    np.array(FLAG_COLS, dtype=np.int64))
    return softmax(Z @ np.array(W, dtype=np.float64) + np.array(B, dtype=np.float64))


def probabilities(df, inputs=None):
    """The classifier's P(H, A, D) for every frame of a feature table."""
    assert FEATURE_NAMES, "no fitted numbers: run python -m src.training.train_rule_learnt_ensemble"
    df = with_tracking_columns(df, inputs)
    X, _ = features(df, names=FEATURE_NAMES)
    return predict_proba(X)


def predict(data_frames, inputs=None):
    """Labels H/A/D per frame. inputs: optional raw tracking (players, team_of_slot, half) for the
    columns the table lacks; by default src/features.load_inputs."""
    df = with_tracking_columns(data_frames, inputs)
    P = parts(df)
    X, _ = features(df, names=FEATURE_NAMES, P=P)
    P["lv"] = predict_proba(X)                                   # the learnt vote
    halves = _runs(df["half"].to_numpy() > 0)
    return LETTERS[_decode(score_of(P, weights()), halves)]
