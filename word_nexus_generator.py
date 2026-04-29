#!/usr/bin/env python3
"""
Word Nexus Board Generator — fully offline, zero API calls
==========================================================
Uses a hand-crafted semantic graph (centers → sub-themes → words)
plus numpy/scipy for scoring and diversity checks.

Board layout (indices 0-8):
  [0:TL] [1:T ] [2:TR]
  [3:L ] [4:C ] [5:R ]
  [6:BL] [7:B ] [8:BR]

EIGHT equations — every line of three must satisfy `outer + outer = middle`:

  Row 1 (top)    : TL + TR = T          e.g. ship   + harbor = dock
  Row 2 (mid)    : L  + R  = C          e.g. found. + camera = ANCHOR
  Row 3 (bottom) : BL + BR = B          e.g. sailor + fix    = burden
  Col 1 (left)   : TL + BL = L          e.g. ship   + sailor = foundation
  Col 2 (mid)    : T  + B  = C          e.g. dock   + burden = ANCHOR
  Col 3 (right)  : TR + BR = R          e.g. harbor + fix    = camera
  Diag ↘         : TL + BR = C          e.g. ship   + fix    = ANCHOR
  Diag ↗         : TR + BL = C          e.g. harbor + sailor = ANCHOR

So the four CORNERS (TL, TR, BL, BR) and the CENTER (C) are the "free"
choices.  The four EDGES (T, B, L, R) are *derived* — each is the
strongest concept jointly evoked by its two flanking corners — and they
must themselves pair to C through the middle column / middle row.

A good board has corners drawn from DIFFERENT sub-themes of C, so the
nine words feel varied rather than redundant.

Usage:
  python word_nexus_generator.py
  python word_nexus_generator.py --seed fire
  python word_nexus_generator.py --boards 12 --output puzzles.json
  python word_nexus_generator.py --list-centers
"""

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from itertools import combinations, permutations
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.distance import cosine as cosine_dist

# ══════════════════════════════════════════════════════════════════
#  SEMANTIC GRAPH
#  Format:  CENTERS[center][subtheme] = list of (word, strength)
#  strength: 3 = iconic/strong,  2 = solid,  1 = lateral/clever
#
#  Sub-themes matter: within a center, words from the SAME sub-theme
#  are tightly bonded (good for adjacent corners that need a tight
#  middle).  Words from DIFFERENT sub-themes diversify a board.
# ══════════════════════════════════════════════════════════════════

CENTERS: dict[str, dict[str, list[tuple[str, int]]]] = {

    "fire": {
        "combustion":  [("flame", 3), ("ash", 3), ("ember", 3), ("smoke", 3),
                        ("spark", 3), ("heat", 2), ("burn", 3), ("char", 2)],
        "destruction": [("destroy", 2), ("ruin", 2), ("disaster", 2), ("alarm", 2)],
        "warmth":      [("torch", 3), ("candle", 3), ("hearth", 3), ("beacon", 2)],
        "passion":     [("passion", 2), ("anger", 2), ("fury", 2), ("desire", 2)],
        "dismissal":   [("dismiss", 2), ("resign", 1), ("layoff", 2)],
        "cooking":     [("grill", 2), ("roast", 2), ("forge", 2)],
    },

    "shadow": {
        "light_dark":  [("darkness", 3), ("shade", 3), ("eclipse", 3), ("silhouette", 3),
                        ("dim", 2), ("dusk", 2), ("night", 2), ("veil", 2)],
        "stealth":     [("spy", 2), ("hide", 2), ("lurk", 2), ("cloak", 2),
                        ("ghost", 2), ("phantom", 2), ("trace", 2)],
        "doubt":       [("doubt", 2), ("suspicion", 2), ("threat", 2), ("gloom", 2)],
        "following":   [("follow", 2), ("trail", 2), ("mimic", 2)],
    },

    "anchor": {
        "nautical":    [("ship", 3), ("harbor", 3), ("dock", 3), ("chain", 2),
                        ("sea", 2), ("sailor", 3), ("vessel", 2)],
        "stability":   [("stable", 2), ("heavy", 2), ("ground", 2), ("foundation", 3),
                        ("hold", 2), ("fix", 2)],
        "media":       [("broadcast", 2), ("reporter", 2), ("studio", 2), ("camera", 2)],
        "constraint":  [("burden", 2), ("drag", 2), ("trap", 2)],
    },

    "bridge": {
        "architecture":[("arch", 3), ("span", 3), ("river", 3), ("tower", 2),
                        ("crossing", 3), ("cable", 2), ("steel", 2)],
        "connection":  [("connect", 3), ("link", 3), ("gap", 3), ("divide", 2),
                        ("negotiate", 2), ("treaty", 2), ("diplomat", 2)],
        "card_game":   [("trump", 2), ("suit", 2), ("deal", 2), ("bid", 2)],
        "music":       [("chorus", 2), ("verse", 2), ("melody", 2)],
    },

    "crown": {
        "royalty":     [("king", 3), ("queen", 3), ("throne", 3), ("reign", 3),
                        ("scepter", 3), ("royal", 2), ("noble", 2), ("empire", 2)],
        "achievement": [("trophy", 2), ("champion", 2), ("victor", 2), ("summit", 2),
                        ("glory", 2), ("triumph", 2)],
        "anatomy":     [("tooth", 2), ("head", 2), ("skull", 2)],
        "botany":      [("tree", 2), ("canopy", 2), ("bloom", 2)],
    },

    "pressure": {
        "physics":     [("weight", 3), ("force", 3), ("compress", 3), ("crush", 2),
                        ("squeeze", 2), ("steam", 2), ("valve", 2), ("gauge", 2)],
        "stress":      [("stress", 3), ("anxiety", 3), ("burden", 2), ("demand", 2),
                        ("tension", 2), ("deadline", 2), ("urgent", 2)],
        "persuasion":  [("coerce", 2), ("bully", 2), ("push", 2), ("lobby", 2)],
        "health":      [("pulse", 2), ("heart", 2), ("vein", 2)],
        "atmosphere":  [("weather", 2), ("altitude", 2), ("depth", 2)],
    },

    "echo": {
        "sound":       [("sound", 3), ("cave", 3), ("valley", 2), ("reverb", 3),
                        ("bounce", 2), ("repeat", 2), ("ring", 2), ("resonate", 2)],
        "mythology":   [("nymph", 2), ("narcissus", 2), ("myth", 2)],
        "imitation":   [("mimic", 2), ("copy", 2), ("reflect", 2), ("mirror", 2)],
        "memory":      [("memory", 2), ("ghost", 2), ("trace", 1), ("remnant", 2)],
        "tech":        [("device", 1), ("speaker", 2), ("voice", 2)],
    },

    "gate": {
        "structure":   [("fence", 3), ("wall", 3), ("door", 2), ("lock", 2),
                        ("keystone", 2), ("arch", 2), ("barrier", 2)],
        "access":      [("entry", 3), ("exit", 2), ("pass", 2), ("border", 2),
                        ("ticket", 2), ("checkpoint", 2)],
        "scandal":     [("scandal", 2), ("cover", 2), ("expose", 2), ("corrupt", 2)],
        "electronics": [("circuit", 2), ("logic", 2), ("binary", 2), ("switch", 2)],
        "airport":     [("flight", 2), ("board", 2), ("depart", 2)],
    },

    "root": {
        "botany":      [("soil", 3), ("tree", 3), ("plant", 3), ("grow", 2),
                        ("stem", 2), ("earth", 2), ("dig", 2), ("bulb", 2)],
        "origin":      [("origin", 3), ("ancestor", 3), ("heritage", 2), ("culture", 2),
                        ("homeland", 2), ("tradition", 2)],
        "math":        [("square", 2), ("radical", 2), ("equation", 1)],
        "cause":       [("source", 2), ("cause", 2), ("basis", 2), ("core", 2)],
        "dental":      [("canal", 2), ("tooth", 2), ("nerve", 2)],
    },

    "flood": {
        "water":       [("rain", 3), ("river", 3), ("dam", 3), ("surge", 3),
                        ("overflow", 3), ("deluge", 3), ("torrent", 2)],
        "disaster":    [("escape", 2), ("refuge", 2), ("ruin", 2), ("rescue", 2)],
        "overwhelming":[("swamp", 2), ("overwhelm", 2), ("saturate", 2), ("drown", 2)],
        "light":       [("light", 2), ("beam", 2), ("stage", 2), ("bright", 2)],
        "biblical":    [("ark", 3), ("Noah", 2), ("raven", 2)],
    },

    "ghost": {
        "supernatural":[("spirit", 3), ("haunt", 3), ("specter", 3), ("soul", 2),
                        ("apparition", 3), ("phantom", 2), ("undead", 2)],
        "fear":        [("scare", 2), ("pale", 2), ("chill", 2), ("dread", 2)],
        "absence":     [("vanish", 2), ("fade", 2), ("absent", 2), ("silent", 2)],
        "tech":        [("machine", 1), ("data", 1), ("memory", 2)],
        "pop_culture": [("sheet", 2), ("Halloween", 2), ("trick", 1)],
    },

    "mirror": {
        "optics":      [("reflect", 3), ("glass", 3), ("image", 3), ("surface", 2),
                        ("lens", 2), ("light", 2)],
        "identity":    [("self", 2), ("vanity", 3), ("narcissus", 2), ("ego", 2)],
        "imitation":   [("copy", 2), ("reverse", 2), ("flip", 2), ("twin", 2)],
        "truth":       [("truth", 2), ("illusion", 2), ("mask", 2), ("reveal", 2)],
        "fairy_tale":  [("queen", 2), ("fairest", 2), ("magic", 2)],
    },

    "storm": {
        "weather":     [("thunder", 3), ("lightning", 3), ("rain", 2), ("wind", 3),
                        ("hail", 2), ("cloud", 2), ("gale", 2), ("cyclone", 2)],
        "conflict":    [("battle", 2), ("rage", 2), ("chaos", 2), ("fury", 2),
                        ("outrage", 2), ("protest", 2), ("upheaval", 2)],
        "military":    [("assault", 2), ("invade", 2), ("siege", 2), ("charge", 2)],
        "brain":       [("idea", 2), ("creative", 2), ("inspire", 2)],
    },

    "chain": {
        "physical":    [("link", 3), ("metal", 2), ("lock", 2), ("fence", 2),
                        ("hook", 2), ("connect", 2), ("bind", 2)],
        "captivity":   [("prison", 2), ("slave", 2), ("captive", 2), ("bound", 2)],
        "sequence":    [("sequence", 2), ("series", 2), ("reaction", 2), ("domino", 3)],
        "business":    [("supply", 2), ("retail", 2), ("store", 2), ("franchise", 2)],
        "biology":     [("predator", 2), ("prey", 2), ("ecosystem", 2)],
    },

    "spark": {
        "fire_elec":   [("ignite", 3), ("flame", 2), ("electric", 3), ("plug", 3),
                        ("static", 2), ("wire", 2), ("voltage", 2)],
        "inspiration": [("inspire", 3), ("idea", 3), ("genius", 2), ("creative", 2)],
        "romance":     [("romance", 2), ("attraction", 2), ("chemistry", 2), ("flirt", 2)],
        "beginning":   [("start", 2), ("trigger", 2), ("catalyst", 2), ("origin", 2)],
    },

    "vault": {
        "architecture":[("arch", 3), ("ceiling", 2), ("dome", 2), ("column", 2)],
        "security":    [("safe", 3), ("bank", 3), ("lock", 2), ("steel", 2),
                        ("treasure", 2), ("secret", 2)],
        "athletics":   [("jump", 3), ("pole", 3), ("leap", 2), ("athlete", 2)],
        "death":       [("tomb", 2), ("crypt", 2), ("bury", 2), ("coffin", 2)],
    },

    "veil": {
        "fabric":      [("fabric", 2), ("curtain", 2), ("cover", 2), ("sheet", 2)],
        "secrecy":     [("hide", 2), ("secret", 2), ("mystery", 2), ("mask", 2)],
        "marriage":    [("bride", 3), ("wedding", 3), ("ceremony", 2), ("altar", 2)],
        "religion":    [("nun", 2), ("sacred", 2), ("divine", 2), ("ritual", 2)],
        "metaphor":    [("death", 2), ("illusion", 2)],
    },

    "wave": {
        "ocean":       [("ocean", 3), ("surf", 3), ("tide", 3), ("shore", 2),
                        ("crest", 2), ("swell", 2), ("beach", 2)],
        "physics":     [("frequency", 3), ("sound", 2), ("light", 2), ("radio", 2),
                        ("vibration", 2), ("signal", 2)],
        "gesture":     [("greet", 2), ("farewell", 2), ("hand", 2), ("salute", 2)],
        "movement":    [("trend", 2), ("surge", 2), ("mob", 2)],
    },

    "needle": {
        "sewing":      [("thread", 3), ("sew", 3), ("fabric", 2), ("stitch", 2),
                        ("thimble", 2), ("tailor", 2)],
        "medicine":    [("inject", 3), ("syringe", 3), ("vaccine", 2), ("vein", 2),
                        ("blood", 2), ("hospital", 2)],
        "precision":   [("sharp", 2), ("point", 2), ("pierce", 2), ("precise", 2)],
        "compass":     [("compass", 3), ("north", 2), ("navigate", 2), ("direction", 2)],
        "haystack":    [("search", 2), ("find", 2), ("rare", 2)],
    },

    "lens": {
        "optics":      [("glass", 2), ("focus", 3), ("light", 2), ("refract", 3),
                        ("magnify", 3), ("clear", 2), ("optical", 2)],
        "camera":      [("camera", 3), ("photo", 2), ("zoom", 2), ("frame", 2),
                        ("aperture", 2), ("capture", 2)],
        "eye":         [("eye", 3), ("cornea", 2), ("sight", 2), ("vision", 2)],
        "perspective": [("view", 2), ("angle", 2), ("bias", 2), ("context", 2)],
    },

    "current": {
        "electricity": [("voltage", 3), ("circuit", 3), ("wire", 2), ("charge", 2),
                        ("battery", 2), ("conductor", 2)],
        "water":       [("flow", 3), ("stream", 3), ("tide", 2), ("drift", 2),
                        ("river", 2), ("eddy", 2)],
        "time":        [("present", 2), ("today", 2), ("modern", 2), ("trend", 2),
                        ("latest", 2), ("news", 2)],
    },

    "key": {
        "locks":       [("lock", 3), ("door", 3), ("open", 2), ("access", 2),
                        ("entry", 2), ("escape", 2)],
        "music":       [("note", 2), ("chord", 3), ("scale", 3), ("piano", 3),
                        ("tone", 2), ("harmony", 2)],
        "importance":  [("crucial", 2), ("vital", 2), ("critical", 2), ("core", 2)],
        "computer":    [("keyboard", 3), ("press", 2), ("type", 2)],
        "crypto":      [("cipher", 2), ("decode", 2), ("encrypt", 2)],
    },

    "balance": {
        "physics":     [("scale", 3), ("weight", 2), ("tilt", 2), ("level", 2),
                        ("stable", 2), ("center", 2)],
        "finance":     [("debt", 2), ("credit", 2), ("account", 2), ("budget", 2),
                        ("ledger", 2), ("audit", 2)],
        "justice":     [("justice", 3), ("court", 2), ("judge", 2), ("fair", 2),
                        ("equal", 2), ("rights", 2)],
        "wellness":    [("yoga", 2), ("posture", 2), ("harmony", 2), ("calm", 2)],
    },

    "trap": {
        "capture":     [("cage", 3), ("snare", 3), ("bait", 3), ("lure", 2),
                        ("hunter", 2), ("prey", 2)],
        "deception":   [("trick", 2), ("deceive", 2), ("ambush", 2), ("scheme", 2),
                        ("plot", 2), ("betray", 2)],
        "confined":    [("prison", 2), ("escape", 2), ("stuck", 2), ("web", 2)],
        "music":       [("beat", 2), ("bass", 2), ("rhythm", 2), ("rapper", 1)],
        "drums":       [("drum", 2), ("percussion", 1)],
    },

    "forge": {
        "metalwork":   [("metal", 3), ("anvil", 3), ("hammer", 3), ("heat", 2),
                        ("iron", 2), ("steel", 2), ("smith", 3)],
        "creation":    [("create", 2), ("build", 2), ("craft", 2), ("shape", 2)],
        "deception":   [("fake", 3), ("counterfeit", 3), ("copy", 2), ("fraud", 2),
                        ("sign", 2), ("document", 2)],
        "alliance":    [("bond", 2), ("alliance", 2), ("unite", 2), ("treaty", 2)],
    },

    "spiral": {
        "shape":       [("curve", 2), ("coil", 3), ("helix", 3), ("spin", 2),
                        ("rotate", 2), ("swirl", 2), ("twist", 2)],
        "decline":     [("decline", 2), ("collapse", 2), ("crisis", 2), ("downfall", 2),
                        ("worse", 2), ("chaos", 2)],
        "cosmos":      [("galaxy", 3), ("nebula", 2), ("cosmos", 2), ("orbit", 2)],
        "nature":      [("shell", 2), ("snail", 2), ("fern", 2), ("nature", 2)],
    },

    "feast": {
        "food":        [("banquet", 3), ("dinner", 2), ("table", 2), ("plenty", 2),
                        ("abundance", 2), ("spread", 2), ("meal", 2)],
        "religion":    [("fast", 2), ("prayer", 2), ("holiday", 2), ("sacred", 2),
                        ("church", 2), ("ritual", 2)],
        "indulgence":  [("eye", 2), ("indulge", 2), ("luxury", 2), ("pleasure", 2)],
        "royalty":     [("king", 2), ("castle", 2), ("hall", 2), ("mead", 2)],
    },

    "seed": {
        "botany":      [("plant", 3), ("soil", 2), ("grow", 2), ("flower", 2),
                        ("fruit", 2), ("harvest", 2), ("sprout", 2)],
        "origin":      [("idea", 2), ("begin", 2), ("origin", 2), ("start", 2),
                        ("embryo", 2), ("potential", 2)],
        "sports":      [("rank", 2), ("bracket", 2), ("tournament", 2), ("compete", 1)],
        "tech":        [("database", 1), ("random", 1), ("algorithm", 1)],
    },

    "beacon": {
        "light":       [("lighthouse", 3), ("flash", 2), ("signal", 3), ("light", 2),
                        ("tower", 2), ("lamp", 2), ("warn", 2)],
        "navigation":  [("guide", 3), ("navigate", 2), ("north", 2), ("compass", 2),
                        ("direction", 2), ("landmark", 2)],
        "hope":        [("hope", 3), ("symbol", 2), ("inspire", 2), ("rally", 2),
                        ("hero", 2), ("example", 2)],
    },

    "thread": {
        "sewing":      [("needle", 3), ("fabric", 3), ("weave", 2), ("stitch", 2),
                        ("sew", 2), ("loom", 2), ("cloth", 2)],
        "narrative":   [("story", 2), ("plot", 2), ("follow", 2), ("connect", 2),
                        ("link", 2), ("trace", 2)],
        "internet":    [("post", 2), ("reply", 2), ("forum", 2), ("comment", 2)],
        "hardware":    [("bolt", 2), ("screw", 2), ("grip", 2), ("twist", 2)],
    },

    "pulse": {
        "heart":       [("heart", 3), ("beat", 3), ("artery", 2), ("blood", 2),
                        ("vital", 2), ("alive", 2), ("rhythm", 2)],
        "music":       [("tempo", 2), ("drum", 2), ("bass", 2), ("groove", 2)],
        "signal":      [("signal", 2), ("wave", 2), ("frequency", 2), ("current", 2)],
        "energy":      [("city", 2), ("energy", 2), ("vibrant", 2), ("rush", 2)],
    },

    "venom": {
        "creatures":   [("snake", 3), ("bite", 3), ("fang", 3), ("spider", 2),
                        ("scorpion", 2), ("cobra", 2)],
        "poison":      [("poison", 3), ("toxic", 2), ("lethal", 2), ("antidote", 2)],
        "hatred":      [("hate", 2), ("spite", 2), ("bitter", 2), ("cruel", 2),
                        ("malice", 2), ("sharp", 1)],
        "medicine":    [("cure", 2), ("dose", 2), ("extract", 2)],
    },

    "quarry": {
        "mining":      [("stone", 3), ("mine", 3), ("excavate", 2), ("rock", 2),
                        ("blast", 2), ("marble", 2), ("limestone", 2)],
        "hunting":     [("prey", 3), ("hunt", 3), ("chase", 2), ("pursue", 2),
                        ("target", 2), ("escape", 2)],
        "birds":       [("hawk", 1), ("falcon", 1), ("eagle", 1)],
    },

    "tide": {
        "ocean":       [("moon", 3), ("ocean", 3), ("shore", 2), ("ebb", 3),
                        ("flow", 2), ("wave", 2), ("lunar", 2)],
        "cycles":      [("shift", 2), ("cycle", 2), ("turn", 2), ("change", 2),
                        ("fortune", 2), ("fate", 2)],
        "politics":    [("opinion", 2), ("war", 2), ("politics", 2), ("sentiment", 2)],
    },

}


# ══════════════════════════════════════════════════════════════════
#  DERIVED INDICES
# ══════════════════════════════════════════════════════════════════

# word → {center: max strength}
WORD_INDEX: dict[str, dict[str, int]] = defaultdict(dict)

# (center, word) → subtheme name
WORD_SUBTHEME: dict[tuple[str, str], str] = {}

# center → list of (word, subtheme, strength)
CENTER_WORDS: dict[str, list[tuple[str, str, int]]] = defaultdict(list)

# center → list of subtheme names
CENTER_SUBTHEMES: dict[str, list[str]] = {}

for _center, _themes in CENTERS.items():
    CENTER_SUBTHEMES[_center] = list(_themes.keys())
    for _theme, _entries in _themes.items():
        for _word, _strength in _entries:
            w = _word.lower()
            WORD_INDEX[w][_center] = max(WORD_INDEX[w].get(_center, 0), _strength)
            # First sub-theme assignment wins for a (center, word) pair
            WORD_SUBTHEME.setdefault((_center, w), _theme)
            CENTER_WORDS[_center].append((w, _theme, _strength))


# ══════════════════════════════════════════════════════════════════
#  MORPHOLOGICAL DEDUPLICATION
# ══════════════════════════════════════════════════════════════════

_SUFFIXES = [
    "ations", "ation", "ments", "ment", "nesses", "ness",
    "ings", "ing", "tions", "tion", "ers", "ies",
    "est", "ful", "less", "ally", "ily", "ly", "al", "er",
    "ed", "es", "s",
]

def morph_root(word: str) -> str:
    w = word.lower()
    for sfx in _SUFFIXES:
        if w.endswith(sfx) and len(w) - len(sfx) >= 3:
            w = w[: len(w) - len(sfx)]
            break
    if w.endswith("e") and len(w) > 3:
        w = w[:-1]
    return w


def has_morph_overlap(words: list[str]) -> bool:
    roots = [morph_root(w) for w in words]
    for i in range(len(roots)):
        for j in range(i + 1, len(roots)):
            ri, rj = roots[i], roots[j]
            if ri == rj:
                return True
            if len(ri) >= 4 and len(rj) >= 4:
                if ri.startswith(rj) or rj.startswith(ri):
                    return True
    return False


# ══════════════════════════════════════════════════════════════════
#  PAIR STRENGTH  (both words → some target concept)
# ══════════════════════════════════════════════════════════════════

def pair_strength(word_a: str, word_b: str, target: str) -> float:
    """
    Score a semantic pair (A, B → TARGET).
    Both words must evoke `target` (i.e. target appears in both their
    WORD_INDEX entries).  Returns 0.0 if either link is missing.
    """
    sa = WORD_INDEX.get(word_a, {}).get(target, 0)
    sb = WORD_INDEX.get(word_b, {}).get(target, 0)
    if sa == 0 or sb == 0:
        return 0.0

    geo = math.sqrt(sa * sb)

    # Diversity bonus: words that hit the target through different
    # sub-themes feel cleverer than two synonyms forcing the same angle.
    ta = WORD_SUBTHEME.get((target, word_a))
    tb = WORD_SUBTHEME.get((target, word_b))
    if ta and tb and ta != tb:
        geo += 0.4
    return round(geo, 3)


# ══════════════════════════════════════════════════════════════════
#  EDGE FINDER  —  best concept jointly evoked by two corners
# ══════════════════════════════════════════════════════════════════

def edge_score(a: str, b: str, edge: str, center: str) -> float:
    """
    Score how good `edge` is as the joint concept of `a` and `b`,
    given that the whole board is anchored to `center`.

    Requires:
      • a, b, edge are all in CENTERS[center] (they all evoke center).
      • edge itself is in CENTERS[center] so it can later pair to C.

    Bonuses:
      • If a, b, edge all share the same sub-theme of center → tight (×1.6).
      • If exactly two of the three share a sub-theme → moderate (×1.25).
      • Otherwise base score from raw strengths.
    """
    sa = WORD_INDEX.get(a, {}).get(center, 0)
    sb = WORD_INDEX.get(b, {}).get(center, 0)
    se = WORD_INDEX.get(edge, {}).get(center, 0)
    if sa == 0 or sb == 0 or se == 0:
        return 0.0
    base = (sa * sb * se) ** (1 / 3)

    ta = WORD_SUBTHEME.get((center, a))
    tb = WORD_SUBTHEME.get((center, b))
    te = WORD_SUBTHEME.get((center, edge))
    themes = [t for t in (ta, tb, te) if t]
    if len(themes) == 3 and themes[0] == themes[1] == themes[2]:
        base *= 1.6
    elif len(themes) >= 2 and (ta == tb == te or
                               (ta == te and ta is not None) or
                               (tb == te and tb is not None) or
                               (ta == tb and ta is not None)):
        # at least one matching pair
        if (ta == te and ta is not None) or (tb == te and tb is not None):
            base *= 1.25  # edge shares theme with at least one corner
        else:
            base *= 1.10  # corners share theme but edge is different
    return round(base, 3)


def best_edge(a: str, b: str, center: str, exclude: set[str]) -> Optional[tuple[str, float]]:
    """
    Return (edge_word, score) — the best concept that both a and b evoke
    within CENTERS[center], excluding any word in `exclude`.
    Returns None if no valid edge exists.
    """
    pool = [w for (w, _t, _s) in CENTER_WORDS[center]]
    seen = set()
    best_w, best_s = None, 0.0
    for w in pool:
        if w in exclude or w in seen:
            continue
        seen.add(w)
        if w == a or w == b:
            continue
        if has_morph_overlap([a, b, w]):
            continue
        s = edge_score(a, b, w, center)
        if s > best_s:
            best_s, best_w = s, w
    return (best_w, round(best_s, 3)) if best_w else None


# ══════════════════════════════════════════════════════════════════
#  CORNER DIVERSITY
# ══════════════════════════════════════════════════════════════════

def corner_diversity(corners: list[str], center: str) -> tuple[int, float]:
    """
    Returns (n_distinct_subthemes, vector_distance_score).
      • n_distinct_subthemes: how many DIFFERENT sub-themes of `center`
        the four corners cover (1–4 — higher is better).
      • vector_distance_score: 0–10 average pairwise cosine distance of
        corner association vectors across all centers.
    """
    themes = {WORD_SUBTHEME.get((center, c)) for c in corners}
    themes.discard(None)
    n_themes = len(themes)

    center_list = list(CENTERS.keys())
    n = len(center_list)

    def vec(word: str) -> np.ndarray:
        v = np.zeros(n)
        for i, c in enumerate(center_list):
            v[i] = WORD_INDEX.get(word, {}).get(c, 0)
        return v

    vecs = [vec(w) for w in corners]
    dists = []
    for i, j in combinations(range(4), 2):
        a, b = vecs[i], vecs[j]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 0 and nb > 0:
            dists.append(cosine_dist(a, b))
        else:
            dists.append(0.5)
    return n_themes, round(float(np.mean(dists)) * 10, 2)


# ══════════════════════════════════════════════════════════════════
#  BOARD SCORING — eight equations
# ══════════════════════════════════════════════════════════════════

def score_board(board: dict) -> dict:
    g = board["grid"]
    tl, t, tr, l, c, r, bl, b, br = g

    line_scores = {
        "row_top":     pair_strength(tl, tr, t),    # TL + TR = T
        "row_mid":     pair_strength(l,  r,  c),    # L  + R  = C
        "row_bot":     pair_strength(bl, br, b),    # BL + BR = B
        "col_left":    pair_strength(tl, bl, l),    # TL + BL = L
        "col_mid":     pair_strength(t,  b,  c),    # T  + B  = C
        "col_right":   pair_strength(tr, br, r),    # TR + BR = R
        "diag_main":   pair_strength(tl, br, c),    # TL + BR = C
        "diag_anti":   pair_strength(tr, bl, c),    # TR + BL = C
    }

    weakest = min(line_scores.values())
    avg     = sum(line_scores.values()) / 8

    n_themes, corner_div = corner_diversity([tl, tr, bl, br], c)

    overall = round(
        0.45 * weakest +
        0.30 * avg +
        0.15 * (corner_div / 10) * 3 +
        0.10 * (n_themes / 4) * 3,
        2,
    )

    playable = (
        weakest >= 1.5 and
        avg >= 2.0 and
        n_themes >= 3 and
        overall >= 1.8
    )

    return {
        "line_scores":      line_scores,
        "weakest_line":     weakest,
        "average_line":     round(avg, 2),
        "corner_diversity": corner_div,
        "corner_subthemes": n_themes,
        "overall":          overall,
        "playable":         playable,
    }


# ══════════════════════════════════════════════════════════════════
#  BOARD ASSEMBLY
# ══════════════════════════════════════════════════════════════════

def _corner_candidates(center: str, min_strength: int = 2) -> list[str]:
    seen, out = set(), []
    for (w, _theme, s) in CENTER_WORDS[center]:
        if s >= min_strength and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def assemble_board(
    center: str,
    rng: random.Random,
    max_corner_pool: int = 18,
    min_diag_strength: float = 2.0,
    min_edge_score: float = 1.6,
    min_subthemes: int = 3,
) -> Optional[dict]:
    """
    Build a board that satisfies all 8 equations.

    Strategy:
      1. Pick a pool of corner candidates (words evoking `center`).
      2. Iterate over 4-corner combinations that span ≥`min_subthemes`
         distinct sub-themes of `center`.
      3. For each ordering of (TL, TR, BL, BR):
           a. Diagonals must point at C with strength ≥ min_diag_strength.
           b. Derive each edge as the best joint concept of its corners.
           c. Verify T+B → C and L+R → C also have ≥ min_diag_strength.
      4. Return the first valid board found (caller can re-shuffle for variety).
    """
    pool = _corner_candidates(center, min_strength=2)
    if len(pool) < 6:
        return None
    rng.shuffle(pool)
    pool = pool[:max_corner_pool]

    # Pre-compute themes for fast filtering
    themes_of = {w: WORD_SUBTHEME.get((center, w)) for w in pool}

    quad_combos = list(combinations(pool, 4))
    rng.shuffle(quad_combos)

    for quad in quad_combos:
        # corner diversity gate
        ts = {themes_of[w] for w in quad}
        ts.discard(None)
        if len(ts) < min_subthemes:
            continue

        # morph overlap among corners alone
        if has_morph_overlap(list(quad)):
            continue

        # try corner orderings — there are 24 permutations but symmetries
        # make many equivalent.  We sample to keep it fast.
        orders = list(permutations(quad))
        rng.shuffle(orders)

        for tl, tr, bl, br in orders[:8]:
            # diagonals must hit center
            d1 = pair_strength(tl, br, center)
            d2 = pair_strength(tr, bl, center)
            if d1 < min_diag_strength or d2 < min_diag_strength:
                continue

            used = {tl, tr, bl, br, center}

            # row 1 edge: TL + TR -> T
            t_pick = best_edge(tl, tr, center, exclude=used)
            if not t_pick or t_pick[1] < min_edge_score:
                continue
            t_word = t_pick[0]
            used.add(t_word)

            # row 3 edge: BL + BR -> B
            b_pick = best_edge(bl, br, center, exclude=used)
            if not b_pick or b_pick[1] < min_edge_score:
                continue
            b_word = b_pick[0]
            used.add(b_word)

            # col 1 edge: TL + BL -> L
            l_pick = best_edge(tl, bl, center, exclude=used)
            if not l_pick or l_pick[1] < min_edge_score:
                continue
            l_word = l_pick[0]
            used.add(l_word)

            # col 3 edge: TR + BR -> R
            r_pick = best_edge(tr, br, center, exclude=used)
            if not r_pick or r_pick[1] < min_edge_score:
                continue
            r_word = r_pick[0]

            # the four edges must themselves pair to C through middle
            # row and middle column
            cm_v = pair_strength(t_word, b_word, center)
            cm_h = pair_strength(l_word, r_word, center)
            if cm_v < min_diag_strength or cm_h < min_diag_strength:
                continue

            grid = [tl, t_word, tr, l_word, center, r_word, bl, b_word, br]

            if len(set(grid)) != 9:
                continue
            if has_morph_overlap(grid):
                continue

            return {
                "grid":   grid,
                "center": center,
                "edges": {
                    "row_top":   {"pair": [tl, tr], "result": t_word, "score": t_pick[1]},
                    "row_bot":   {"pair": [bl, br], "result": b_word, "score": b_pick[1]},
                    "col_left":  {"pair": [tl, bl], "result": l_word, "score": l_pick[1]},
                    "col_right": {"pair": [tr, br], "result": r_word, "score": r_pick[1]},
                },
                "lines_to_center": {
                    "diag_main": {"pair": [tl, br], "score": d1},
                    "diag_anti": {"pair": [tr, bl], "score": d2},
                    "col_mid":   {"pair": [t_word, b_word], "score": cm_v},
                    "row_mid":   {"pair": [l_word, r_word], "score": cm_h},
                },
            }
    return None


# ══════════════════════════════════════════════════════════════════
#  CENTER SELECTION  (diverse centers for multi-board runs)
# ══════════════════════════════════════════════════════════════════

def center_distance_matrix() -> dict[tuple[str, str], float]:
    center_list = list(CENTERS.keys())
    n = len(center_list)
    all_words = list(WORD_INDEX.keys())
    m = len(all_words)
    mat = np.zeros((n, m))
    for i, c in enumerate(center_list):
        for j, w in enumerate(all_words):
            mat[i, j] = WORD_INDEX[w].get(c, 0)
    dist = {}
    for i, j in combinations(range(n), 2):
        a, b = mat[i], mat[j]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        d = float(cosine_dist(a, b)) if na > 0 and nb > 0 else 1.0
        dist[(center_list[i], center_list[j])] = round(d, 3)
        dist[(center_list[j], center_list[i])] = round(d, 3)
    return dist


def pick_diverse_centers(
    n: int,
    seed_word: str = "",
    rng: random.Random = None,
) -> list[str]:
    rng = rng or random.Random()
    all_centers = list(CENTERS.keys())

    if seed_word:
        seed_lower = seed_word.lower()
        if seed_lower in CENTERS:
            starters = [seed_lower]
        else:
            starters = []
            for c in all_centers:
                for theme_words in CENTERS[c].values():
                    if any(seed_lower == w or seed_lower in w or w in seed_lower
                           for w, _ in theme_words):
                        starters.append(c)
                        break
        if not starters:
            starters = rng.sample(all_centers, min(2, len(all_centers)))
    else:
        starters = rng.sample(all_centers, min(2, len(all_centers)))

    dist = center_distance_matrix()

    chosen = [starters[0]]
    remaining = [c for c in all_centers if c not in chosen]

    while len(chosen) < n and remaining:
        best, best_score = None, -1.0
        rng.shuffle(remaining)
        for c in remaining:
            min_dist = min(dist.get((c, ch), 1.0) for ch in chosen)
            if min_dist > best_score:
                best_score = min_dist
                best = c
        chosen.append(best)
        remaining.remove(best)

    return chosen[:n]


# ══════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════

def generate_boards(
    seed_word: str = "",
    target: int = 6,
    min_overall: float = 1.8,
    random_seed: Optional[int] = None,
    verbose: bool = True,
) -> list[dict]:

    rng = random.Random(random_seed)

    def vprint(*a):
        if verbose:
            print(*a, flush=True)

    vprint(f"\n{'═'*56}")
    vprint("  WORD NEXUS — Offline Board Generator (8-equation)")
    vprint(f"{'═'*56}")
    vprint(f"  Seed word : {seed_word or '(none)'}")
    vprint(f"  Target    : {target} boards")
    vprint(f"  Min score : {min_overall}")
    vprint(f"{'═'*56}\n")

    n_centers_to_try = min(len(CENTERS), max(target * 4, 16))
    centers = pick_diverse_centers(n_centers_to_try, seed_word=seed_word, rng=rng)

    accepted: list[dict] = []

    for idx, center in enumerate(centers):
        if len(accepted) >= target:
            break

        vprint(f"▶  [{idx+1:02d}/{len(centers)}] center = \"{center.upper()}\"")

        board = None
        for attempt in range(8):
            board = assemble_board(center, rng=rng)
            if board:
                break

        if board is None:
            vprint(f"      ✗  could not assemble a valid 8-equation board\n")
            continue

        sc = score_board(board)
        board["score"] = sc

        vprint(f"      grid       : {board['grid']}")
        vprint(f"      overall    : {sc['overall']}  |  weakest line: {sc['weakest_line']}  "
               f"|  themes used: {sc['corner_subthemes']}/4")

        if sc["overall"] >= min_overall and sc["playable"]:
            accepted.append(board)
            vprint(f"      ✓  accepted  ({len(accepted)}/{target})\n")
        else:
            vprint(f"      ✗  below quality threshold\n")

    vprint(f"\n{'═'*56}")
    vprint(f"  Finished.  {len(accepted)} board(s) accepted.")
    vprint(f"{'═'*56}\n")
    return accepted


# ══════════════════════════════════════════════════════════════════
#  DISPLAY
# ══════════════════════════════════════════════════════════════════

def print_board(board: dict, index: int = 1):
    g = board["grid"]
    sc = board.get("score", {})
    tl, t, tr, l, c, r, bl, b, br = g

    w = max(len(x) for x in g) + 2
    row = lambda a, b_, c_: f"  │ {a:^{w}} │ {b_:^{w}} │ {c_:^{w}} │"
    bar = "  ├" + ("─" * (w + 2) + "┼") * 2 + "─" * (w + 2) + "┤"
    top = "  ┌" + ("─" * (w + 2) + "┬") * 2 + "─" * (w + 2) + "┐"
    bot = "  └" + ("─" * (w + 2) + "┴") * 2 + "─" * (w + 2) + "┘"

    print(f"\n── Board #{index} {'─'*40}")
    print(f"  Center      : {c.upper()}")
    print(f"  Overall     : {sc.get('overall','?')}  |  "
          f"Corner div: {sc.get('corner_diversity','?')}/10  |  "
          f"Sub-themes: {sc.get('corner_subthemes','?')}/4  |  "
          f"Playable: {sc.get('playable','?')}")
    print()
    print(top)
    print(row(tl, t, tr))
    print(bar)
    print(row(l, c.upper(), r))
    print(bar)
    print(row(bl, b, br))
    print(bot)
    print()
    ls = sc.get("line_scores", {})

    def line(label, key, a, b_, target):
        s = ls.get(key, "?")
        target_disp = target.upper() if target == c else target
        print(f"  {label:<10}:  {a:>14} + {b_:<14} →  {target_disp:<14} [{s}]")

    print("  ── Rows ──")
    line("Row top",  "row_top",   tl, tr, t)
    line("Row mid",  "row_mid",   l,  r,  c)
    line("Row bot",  "row_bot",   bl, br, b)
    print("  ── Columns ──")
    line("Col left", "col_left",  tl, bl, l)
    line("Col mid",  "col_mid",   t,  b,  c)
    line("Col right","col_right", tr, br, r)
    print("  ── Diagonals ──")
    line("Diag ↘",   "diag_main", tl, br, c)
    line("Diag ↗",   "diag_anti", tr, bl, c)
    print()


# ══════════════════════════════════════════════════════════════════
#  JSON SERIALISATION
# ══════════════════════════════════════════════════════════════════

def boards_to_json(boards: list[dict]) -> dict:
    out = []
    for b in boards:
        sc = b.get("score", {})
        out.append({
            "grid":   b["grid"],
            "center": b["center"],
            "edges":  b.get("edges", {}),
            "lines_to_center": b.get("lines_to_center", {}),
            "score": {
                "overall":          sc.get("overall"),
                "weakest_line":     sc.get("weakest_line"),
                "average_line":     sc.get("average_line"),
                "corner_diversity": sc.get("corner_diversity"),
                "corner_subthemes": sc.get("corner_subthemes"),
                "playable":         sc.get("playable"),
                "line_scores":      sc.get("line_scores", {}),
            },
            "algorithm_params": {
                "method":               "subthemed_semantic_graph",
                "equations_validated":  8,
                "scoring":              "geometric_mean_with_subtheme_bonus",
                "morph_dedup":          True,
                "uniqueness_check":     True,
                "min_overall_score":    1.8,
                "min_corner_subthemes": 3,
                "center_selection":     "max_cosine_distance",
            },
        })
    return {"boards": out, "total": len(out)}


# ══════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate Word Nexus boards — fully offline, no API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--seed",       default="",    help="Optional theme seed word")
    parser.add_argument("--boards",     type=int, default=6, help="Target boards (default: 6)")
    parser.add_argument("--min-score",  type=float, default=1.8, help="Min overall score (default: 1.8)")
    parser.add_argument("--random-seed",type=int, default=None, help="RNG seed for reproducibility")
    parser.add_argument("--output",     default="word_nexus_boards.json", help="Output JSON path")
    parser.add_argument("--quiet",      action="store_true", help="Suppress verbose output")
    parser.add_argument("--list-centers", action="store_true", help="Print all available centers and exit")
    args = parser.parse_args()

    if args.list_centers:
        print("\nAvailable centers:")
        for i, c in enumerate(sorted(CENTERS.keys()), 1):
            n_words = sum(len(v) for v in CENTERS[c].values())
            n_themes = len(CENTERS[c])
            print(f"  {i:3d}.  {c:<20}  ({n_words} words, {n_themes} sub-themes)")
        print()
        return

    boards = generate_boards(
        seed_word=args.seed,
        target=args.boards,
        min_overall=args.min_score,
        random_seed=args.random_seed,
        verbose=not args.quiet,
    )

    for i, board in enumerate(boards, 1):
        print_board(board, index=i)

    path = Path(args.output)
    path.write_text(json.dumps(boards_to_json(boards), indent=2))
    print(f"✓  Saved {len(boards)} board(s) to {path}")


if __name__ == "__main__":
    main()
