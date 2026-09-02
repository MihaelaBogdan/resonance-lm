# RezoNet — un model de limbaj construit de la zero

Model de secvență cu arhitectură proprie, antrenat de la inițializare aleatoare.
Nu pornește de la greutățile, tokenizatorul sau structura vreunui model existent
și nu folosește niciun framework de învățare automată: motorul de
autodiferențiere, optimizatorul și cele două nuclee de calcul sunt scrise aici,
peste NumPy.

---

## Ideea

Un transformer își amintește **recitind** tot contextul la fiecare pas. RezoNet
își amintește **rezonând**: fiecare canal este un oscilator amortizat, acordat pe
o frecvență proprie. Tokenul curent îl lovește, iar ce a rămas din loviturile
anterioare este memoria. Trecutul nu se recitește — el continuă să vibreze.

Un bloc are două etaje.

**1. Rezonanță.** Starea fiecărui canal *k* este un număr complex care se rotește
și se stinge:

```
s_t = ρ_t · e^(i·ω_k) · s_{t-1} + (u_t + i·v_t)
```

- `ω_k` — frecvența proprie a canalului, învățată, inițializată logaritmic.
- `ρ_t = exp(−softplus(λ_k)·(1 + g_t))` — cât de mult reține canalul. Termenul
  `g_t` se calculează din tokenul curent, deci **uitarea depinde de conținut**:
  modelul poate șterge starea când textul o cere.
- Recurența este liniară în stare, deci stabilă: `|ρ| < 1` garantează că nimic nu
  explodează, oricât de lungă e secvența.

Citirea folosește partea reală, partea imaginară **și anvelopa** `|s| = √(a²+b²)`.
Anvelopa este partea neliniară a etajului și este invariantă la fază — răspunde
la *cât* de mult a rezonat un tipar, indiferent *unde* a început.

**2. Legare și dezlegare holografică.** În loc de atenție și în loc de un
perceptron, blocul folosește algebra vectorilor distribuiți:

```
leagă:      z₁ = p ⊛ q          (convoluție circulară)
interoghează: z₂ = p ⊛ q̃         (corelație circulară — operația inversă)
```

Convoluția amestecă multiplicativ toate perechile de trăsături în O(D log D), nu
O(D²). Corelația este *interogarea*: extrage din starea suprapusă componenta
asociată unei chei. Amândouă se calculează prin FFT.

### Trei proprietăți care ies din construcție

| | |
|---|---|
| **Fără embedding-uri de poziție** | Faza acumulată a fiecărui oscilator spune de cât timp a intrat un semnal. Poziția nu se adaugă — se măsoară. |
| **Cost constant la generare** | Un token costă la fel indiferent dacă în urmă sunt 10 sau 10.000 de caractere. Atenția costă O(T). |
| **Memorie multi-scală** | Canalele pornesc cu constante de timp de la ~1 la ~1000 de caractere; antrenarea le rearanjează singură. |

### Ce este nou și ce nu

Onest: **recurența liniară cu stare complexă diagonală** aparține aceleiași
familii matematice ca modelele moderne de tip state-space, iar **legarea prin
convoluție circulară** vine din reprezentările holografice reduse (Plate, 1995).
Niciuna nu e inventată aici.

Ce este propriu acestui model: combinația lor într-un singur bloc —
oscilator explicit cu frecvență învățată + uitare dependentă de conținut +
citire prin anvelopă, urmat de un mixer care *și leagă, și dezleagă* —
plus faptul că totul, inclusiv autodiferențierea, este scris de la zero.

---

## Structura

```
rezonet/
  autograd.py    motor de autodiferențiere reverse-mode peste NumPy
  ops.py         osc_scan (rezonanța), circconv (legare), circcorr (dezlegare)
  model.py       arhitectura + calea de inferență în flux
  optim.py       AdamW, tăiere de gradient, program cosinus
  tokenizer.py   tokenizator pe caractere, construit din corpus
  data.py        încărcare corpus și eșantionare
scripts/
  make_corpus.py       generatorul de corpus (gramatică românească cu acorduri)
  train.py             antrenare
  sample.py            generare în flux (--bench pentru costul per token)
  eval_agreement.py    test de acord gramatical, cu intervale de încredere
  inspect_spectrum.py  ce frecvențe și scări de timp a învățat modelul
  gradcheck.py         verificarea gradienților prin diferențe finite
  test_consistency.py  calea paralelă = calea în flux
```

## Utilizare

Generarea corpusului:

```bash
python3 scripts/make_corpus.py --bytes 1700000 --out corpus/ro_big.txt
```

Antrenare:

```bash
python3 scripts/train.py --corpus corpus/ro_big.txt --steps 4000 --out checkpoints/model
```

Generare de text:

```bash
python3 scripts/sample.py --ckpt checkpoints/model --prompt "Dimineața, " --n 400
```

Poți antrena pe textul tău propriu — orice fișier `.txt` merge:

```bash
python3 scripts/train.py --corpus calea/catre/textul_tau.txt --steps 4000
```

## Corectitudine

Două verificări care rulează independent de antrenare:

```bash
python3 scripts/gradcheck.py        # gradienți analitici vs. diferențe finite
python3 scripts/test_consistency.py # antrenare și generare dau același rezultat
```

Gradcheck compară fiecare gradient scris de mână cu diferențe finite în precizie
dublă. Eroarea scade pătratic cu pasul `eps` — semnătura erorii de trunchiere,
deci discrepanța rămasă este numerică, nu o greșeală de derivare.

---

## Rezultate

Toate modelele: 3 blocuri, `d_model=128`, ~615.000 de parametri, antrenate pe
CPU în 7–37 de minute. Vocabular de 45 de caractere; o predicție complet
aleatoare ar costa 5,49 biți/caracter.

| model | corpus | fereastră | dezlegare | biți/caracter |
|---|---|---|---|---|
| `base` | 390 KB | 128 | nu | 0,503 |
| `rezonet` | 390 KB | 128 | **da** | 0,504 |
| `rezonet_v2` | 1,7 MB | 128 | da | 0,471 |
| `rezonet_v3` | 1,7 MB | 256 | da | **0,444** |
| `rezonet_v4` | 1,6 MB* | 256 | da | 0,387* |

\* `rezonet_v4` a fost antrenat pe un corpus cu altă compoziție (propoziții lungi
mai dese), deci cifra lui **nu** este comparabilă direct cu celelalte.

### Test de acord gramatical

Comparăm probabilitatea formei corecte a adjectivului cu cea a formei greșite.
Sunt păstrate doar combinațiile care nu apar în corpusul de antrenare, deci se
măsoară generalizarea. Șansa oarbă este 50%.

| sondă | distanță | `base` | `rezonet` | `rezonet_v2` | `rezonet_v3` | `rezonet_v4` |
|---|---|---|---|---|---|---|
| adjectiv lipit de substantiv | 7 car. | 96,0% | 97,8% | 100% | 100% | 100% |
| peste o relativă + un distractor | 53 car. | 57,0% | 63,4% | 99,8% | **100%** | 99,6% |
| peste o relativă + doi distractori | 95 car. | 48,6% | 45,8% | 52,0% | 49,0% | 54,8% |

(intervale de încredere 95%: ±4,4 puncte la ultimele două linii)

**Ce arată tabelul.** Sonda de la 53 de caractere este proiectată să fie
imposibil de rezolvat prin statistici locale: între substantiv și adjectiv stă
un alt substantiv, de gen opus, iar pronumele „care" nu trădează genul. Un model
care s-ar uita la cuvântul precedent ar da sistematic greșit. RezoNet o rezolvă
complet.

**Ce nu funcționează, și de ce nu știm încă.** La 95 de caractere, cu doi
distractori, modelul este la nivelul hazardului. Am testat patru explicații:

1. *Prea puține date* — de 4x mai mult text: a rezolvat cazul de 53 de caractere,
   nu și pe cel de 95.
2. *Fereastră de antrenare prea scurtă* — dublată la 256: fără efect.
3. *Construcția e prea rară* — de 3,5x mai dese exemplele exact de acel tip:
   fără efect.
4. *Modelul nu stăpânește formele de plural* — infirmat direct: **același** acord
   de gen la plural dă 100% la 8 caractere și hazard la 95.

Rămâne o limită reală de distanță a modelului la această scară, nu un artefact
al datelor. Este exact tipul de rezultat pe care testul a fost construit să-l
prindă.

### Ce a decis singur modelul

`scripts/inspect_spectrum.py` arată constantele de timp învățate. Inițializarea
permitea până la 1024 de caractere de memorie; modelul antrenat cu fereastră de
128 și-a scurtat cea mai lungă constantă la **262 de caractere** și și-a grupat
perioadele în jurul a 25–60 de caractere — aproximativ lungimea unei propoziții.
Nu i s-a spus asta; a găsit singur scările de timp ale textului.

### Cost constant la generare

`python3 scripts/sample.py --bench` măsoară timpul per token pe măsură ce
contextul crește:

```
context generat | ms per token
          200 | 0.312
          600 | 0.307
         1200 | 0.308
```

Plat. Un model cu atenție ar fi crescut de șase ori pe același interval.

### Exemplu de generare

Pornind de la „Pădurea, despre care ":

> Pădurea, despre care copilul a vorbit sub cerul senin, este **frumoasă**.

Substantivul-cap este feminin, distractorul dintre ele („copilul") este masculin,
iar adjectivul de la 40 de caractere distanță este acordat corect.

Sensul este absurd pentru că și corpusul este absurd — gramatica lui este
riguroasă, semantica nu. Modelul a învățat exact ce i s-a dat.

---

## Limite

- Testat doar pe un corpus sintetic, la scară mică (~615.000 de parametri).
  Nimic din ce scrie aici nu spune cum s-ar comporta la scară mare.
- Acordul cedează dincolo de ~50–60 de caractere cu mai mulți distractori.
- Recurența rulează secvențial în NumPy. Fiind liniară, ar putea fi paralelizată
  printr-o baleiere asociativă — nu este implementat aici.
- Fără GPU, fără paralelism pe date.
