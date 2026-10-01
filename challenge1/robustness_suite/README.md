# Suite de robustesse : tester un détecteur sur des données bruitées

Les **200 scènes de test** du challenge 1, chacune dégradée par **un** artefact de mesure de laboratoire que le générateur officiel ne produit jamais : 9 familles, 25 jeux. Les masques et les sticks sont ceux du test : toute baisse vient de l'artefact seul. C'est la suite du § 3.1 du [README](../README.md#31-la-suite-dartefacts), utilisée aussi pour le leave-one-family-out (§ 3.3).

| famille | origine physique | niveaux |
|---|---|---|
| `white_x*` | bruit blanc en plus (intégration plus courte) | × 0,5 / 1 / 2 σ_pix |
| `pink1f_x*` | bruit 1/f le long du temps de scan (bruit de charge, ampli) | × 0,5 / 1 / 2 σ_pix |
| `drift_pp*` | dérive lente du fond (point de fonctionnement du capteur) | crête-crête 1,5 / 3 / 6 |
| `jumps_amp*` | sauts télégraphiques, ~4 par image (piège de charge) | amplitude 1,5 / 3 / 6 |
| `stripes_x*` | décalages ligne à ligne en plus | × 0,5 / 1 / 2 σ_h |
| `lowpass_tau*px` | passe-bas 1 pôle sur l'axe rapide (constante de temps du lock-in) | τ = 0,5 / 1 / 2 px |
| `spikes_*pct` | pixels aberrants ±10 (glitchs, parasites) | 0,1 / 0,5 / 2 % des pixels |
| `saturate_c*` | non-linéarité du capteur c·tanh(x/c) (flanc fini du pic de Coulomb) | c = 10 / 5 / 2,5 |
| `polarity_flip` | signe inversé (capteur sur l'autre flanc) | — |

## Ce qui est versionné, et pourquoi c'est suffisant

Les 25 jeux bruités pèsent 570 Mo. Mais les perturbations sont tirées avec des graines fixes ([`detection/robustness.py`](../detection/robustness.py), `make`). Le dossier ne contient donc que :

| fichier | contenu |
|---|---|
| `clean.npz` | les 200 images de test **exactes** (float32) + masques |
| `clean_sticks.jsonl`, `clean_meta.json` | positions des interdots (pour l'obj F1), paramètres du générateur |
| `val.npz` | 1000 scènes de validation propres (float16, erreur ≤ 0,03 pour un bruit σ = 0,9) + masques, **pour choisir le seuil** |
| `val_sticks.jsonl`, `val_meta.json` | idem pour val |

`unpack` reconstruit les 25 jeux **bit à bit** : c'est vérifié contre les originaux. Il faut numpy seulement, et environ 1 minute.

```bash
uv run python -m detection.robustness unpack                 # -> data/robustness/<jeu>/{images,masks}.npy, sticks.jsonl
```

Commandes à lancer depuis `challenge1/` (là où se trouve le paquet `detection`), ou depuis la racine avec `PYTHONPATH=challenge1`.

## Tester son propre détecteur

Écrire une fonction : images brutes `(N, 150, 150)` float32 en entrée, carte de score dans `[0, 1]` de même forme en sortie. Le modèle [`example_detector.py`](example_detector.py) branche notre U-Net ; remplacez le corps de `detect`.

```bash
PYTHONPATH=robustness_suite uv run python -m detection.robustness eval-fn --fn example_detector:detect
```

Le harnais :
1. choisit le seuil **sur val uniquement**, au critère (obj F1 + tol F1)/2, comme toutes nos méthodes ;
2. le fige, puis score `clean` et les 25 jeux ;
3. écrit `challenge1/results/robustness_external/robustness_eval.json` et affiche le **score de robustesse**, la moyenne de obj F1(perturbé) / obj F1(clean).

Repères (mêmes jeux, seuil choisi sur val) :

| détecteur | obj F1 clean | robustesse |
|---|---|---|
| régression logistique (partie 1) | 0,921 | 0,935 |
| U-Net lowsnr (partie 2, `example_detector.py` tel quel) | 0,981 | 0,872 |

**Règles.** C'est un jeu de **rapport** : on n'y choisit ni seuil, ni checkpoint, ni hyperparamètre. Sinon il devient un jeu d'entraînement et ne mesure plus la robustesse. Pour un test honnête d'une famille, il ne faut pas l'avoir randomisée à l'entraînement : c'est le principe du leave-one-family-out.

Pour les modèles de ce repo (checkpoints `runs/<nom>/best.pt`), `detection.evaluate --eval-dir data/robustness` fait la même chose ; `test/` y est alors une copie de `clean/`.
