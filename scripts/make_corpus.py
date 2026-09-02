"""
Genereaza un corpus romanesc dintr-o gramatica cu acorduri reale.

De ce nu text luat de pe internet: aici *stim* regulile. Putem masura direct
daca modelul a invatat acordul gen/numar/articol la distanta, nu doar daca
"suna bine".
"""
import os, sys, random

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "corpus", "ro.txt")

# (nearticulat_sg, articulat_sg, nearticulat_pl, articulat_pl, gen)
NOUNS = [
    ("câine", "câinele", "câini", "câinii", "m"),
    ("munte", "muntele", "munți", "munții", "m"),
    ("copil", "copilul", "copii", "copiii", "m"),
    ("vecin", "vecinul", "vecini", "vecinii", "m"),
    ("pescar", "pescarul", "pescari", "pescarii", "m"),
    ("brad", "bradul", "brazi", "brazii", "m"),
    ("prieten", "prietenul", "prieteni", "prietenii", "m"),
    ("călător", "călătorul", "călători", "călătorii", "m"),
    ("cocoș", "cocoșul", "cocoși", "cocoșii", "m"),
    ("lup", "lupul", "lupi", "lupii", "m"),
    ("cal", "calul", "cai", "caii", "m"),
    ("bătrân", "bătrânul", "bătrâni", "bătrânii", "m"),
    ("casă", "casa", "case", "casele", "f"),
    ("fată", "fata", "fete", "fetele", "f"),
    ("pisică", "pisica", "pisici", "pisicile", "f"),
    ("pădure", "pădurea", "păduri", "pădurile", "f"),
    ("floare", "floarea", "flori", "florile", "f"),
    ("carte", "cartea", "cărți", "cărțile", "f"),
    ("barcă", "barca", "bărci", "bărcile", "f"),
    ("fereastră", "fereastra", "ferestre", "ferestrele", "f"),
    ("poveste", "povestea", "povești", "poveștile", "f"),
    ("vulpe", "vulpea", "vulpi", "vulpile", "f"),
    ("pasăre", "pasărea", "păsări", "păsările", "f"),
    ("umbră", "umbra", "umbre", "umbrele", "f"),
]

# (m_sg, f_sg, m_pl, f_pl)
ADJS = [
    ("mare", "mare", "mari", "mari"),
    ("mic", "mică", "mici", "mici"),
    ("verde", "verde", "verzi", "verzi"),
    ("alb", "albă", "albi", "albe"),
    ("vechi", "veche", "vechi", "vechi"),
    ("frumos", "frumoasă", "frumoși", "frumoase"),
    ("liniștit", "liniștită", "liniștiți", "liniștite"),
    ("rece", "rece", "reci", "reci"),
    ("adânc", "adâncă", "adânci", "adânci"),
    ("tânăr", "tânără", "tineri", "tinere"),
    ("galben", "galbenă", "galbeni", "galbene"),
    ("puternic", "puternică", "puternici", "puternice"),
    ("singuratic", "singuratică", "singuratici", "singuratice"),
    ("obosit", "obosită", "obosiți", "obosite"),
]

# (prezent_3sg, prezent_3pl, participiu)
VERBS = [
    ("doarme", "dorm", "dormit"),
    ("aleargă", "aleargă", "alergat"),
    ("cântă", "cântă", "cântat"),
    ("privește", "privesc", "privit"),
    ("așteaptă", "așteaptă", "așteptat"),
    ("coboară", "coboară", "coborât"),
    ("urcă", "urcă", "urcat"),
    ("pleacă", "pleacă", "plecat"),
    ("rămâne", "rămân", "rămas"),
    ("trece", "trec", "trecut"),
    ("tremură", "tremură", "tremurat"),
    ("odihnește", "odihnesc", "odihnit"),
]

LOCS = ["pe deal", "în pădure", "lângă râu", "sub pod", "peste câmp",
        "la fereastră", "spre sat", "între brazi", "în curte", "pe mal",
        "în vale", "la marginea drumului", "sub cerul senin", "în poiană"]

TIMES = ["Dimineața", "Seara", "Iarna", "Toamna", "De obicei", "Uneori",
         "În fiecare zi", "Noaptea târziu", "La răsărit", "Spre amiază"]

RELATIVE_SHARE = 0.22   # cat de des apar propozitiile cu acord la distanta
LONG_SHARE = 0.33       # din acestea, cat de des varianta lunga (plural, ~95 car.)

NUMS = ["trei", "patru", "cinci", "șase", "șapte", "opt", "nouă", "zece"]


def indef(n):
    return ("un " if n[4] == "m" else "o ") + n[0]


def adj_for(a, gen, plural):
    return a[2 if gen == "m" else 3] if plural else a[0 if gen == "m" else 1]


def relative(r):
    """
    Propozitii cu acord LA DISTANTA: adjectivul final trebuie sa se acorde cu
    substantivul de la inceput, peste o relativa care contine alt substantiv,
    adesea de gen opus. Pronumele "care" nu tradeaza genul, deci un model care
    se uita doar la cuvantul precedent nu poate reusi.
    """
    n = r.choice(NOUNS)
    d = r.choice(NOUNS)
    a = r.choice(ADJS)
    v = r.choice(VERBS)
    loc, loc2 = r.choice(LOCS), r.choice(LOCS)
    t = r.random()
    if t < (1 - LONG_SHARE) * 0.5:
        return f"{n[1].capitalize()}, care {v[0]} {loc}, este {adj_for(a, n[4], False)}."
    if t < (1 - LONG_SHARE):
        return (f"{n[1].capitalize()}, despre care {d[1]} a vorbit {loc}, "
                f"este {adj_for(a, n[4], False)}.")
    return (f"{n[3].capitalize()}, despre care {d[1]} a vorbit {loc} "
            f"{r.choice(TIMES).lower()} și {d[3]} au tăcut {loc2}, "
            f"sunt {adj_for(a, n[4], True)}.")


def sentence(r):
    t = r.random()
    if t < RELATIVE_SHARE:
        return relative(r)
    n = r.choice(NOUNS)
    n2 = r.choice(NOUNS)
    a = r.choice(ADJS)
    a2 = r.choice(ADJS)
    v = r.choice(VERBS)
    v2 = r.choice(VERBS)
    loc, loc2 = r.choice(LOCS), r.choice(LOCS)

    if t < 0.16:
        return f"{indef(n).capitalize()} {adj_for(a, n[4], False)} {v[0]} {loc}."
    if t < 0.30:
        return f"{r.choice(TIMES)}, {n[1]} {adj_for(a, n[4], False)} {v[0]} {loc}."
    if t < 0.44:
        return (f"Când {n[1]} {v[0]} {loc}, {n2[1]} {adj_for(a2, n2[4], False)} "
                f"{v2[0]} {loc2}.")
    if t < 0.56:
        return f"Sunt {r.choice(NUMS)} {n[2]} {adj_for(a, n[4], True)} {loc}."
    if t < 0.68:
        return f"{n[3].capitalize()} {adj_for(a, n[4], True)} {v[1]} {loc}."
    if t < 0.78:
        return f"Unde {v[0]} {n[1]}? {n[1].capitalize()} {v[0]} {loc}."
    if t < 0.88:
        return (f"{n[1].capitalize()} și {n2[1]} au {v[2]} {loc} "
                f"până când {r.choice(TIMES).lower()} s-a făcut liniște.")
    return (f"{r.choice(TIMES)}, {indef(n)} {adj_for(a, n[4], False)} a {v[2]} "
            f"{loc}, iar {n2[1]} {adj_for(a2, n2[4], False)} l-a {v2[2]} {loc2}.")


def main(target_bytes=420_000, seed=7, out=None, rel=None, long=None):
    global OUT, RELATIVE_SHARE, LONG_SHARE
    if rel is not None:
        RELATIVE_SHARE = rel
    if long is not None:
        LONG_SHARE = long
    if out:
        OUT = out
    r = random.Random(seed)
    parts, size = [], 0
    while size < target_bytes:
        para = " ".join(sentence(r) for _ in range(r.randint(3, 7)))
        parts.append(para)
        size += len(para.encode("utf-8")) + 2
    text = "\n\n".join(parts) + "\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"corpus: {OUT}")
    print(f"  {len(text):,} caractere, {len(set(text))} simboluri distincte")
    print("\nprimele randuri:\n")
    print("\n".join(text.split("\n\n")[:2]))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=420_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    ap.add_argument("--relative_share", type=float, default=None)
    ap.add_argument("--long_share", type=float, default=None)
    a = ap.parse_args()
    main(a.bytes, a.seed, a.out, a.relative_share, a.long_share)
