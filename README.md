# Hackathon C12 : calibration automatique d'un double point quantique

Fork de [c12-hackathon/challenge](https://github.com/c12-hackathon/challenge). Le simulateur génère des diagrammes de stabilité de charge (CSD) d'un double point quantique à 5 grilles. Il y a deux défis : **détecter** les interdots, puis **régler** les barrières pour maximiser leur contraste.

![Masques prédits](challenge1/figures/fig2_masques_test.png)

## Organisation du repo

```
hackathon/     ← code fourni par les organisateurs (fork, non modifié)
  README.md      énoncé original du hackathon
  csd/           générateur et simulateur
  starter/       scripts de démarrage
challenge1/    ← notre travail : détection des interdots   ✅ résultats
challenge2/    ← notre travail : optimisation du contraste 🚧 en cours
tests/         tests (pare-feu, suivi, a priori)
docs/          PDF de cours et de contexte (non versionnés)
data/, runs/   données générées (non versionnées)
```

## Challenge 1 : détection, en bref → [challenge1/README.md](challenge1/README.md)

Une **régression logistique** sur deux cartes (filtre adapté + lissage large) bat le meilleur filtre classique, le **filtre adapté** :

| test, 400 scènes | Régression logistique | Filtre adapté |
|---|---|---|
| obj F1 (interdots trouvés) | **0,924** [0,909 ; 0,937] | 0,863 [0,845 ; 0,881] |
| tol F1 (pixel ±1 px) | **0,954** | 0,868 |
| frames du challenge 2 (obj F1) | **0,62** | 0,14 |

Le poids négatif de la feature large lui apprend à rejeter les lignes de charge. Le README du dossier détaille la démarche, les métriques (Dice, tol F1, obj F1 et pourquoi l'obj F1 sert à choisir), les masques prédits, la robustesse et la limite de détection.

## Challenge 2 : optimisation → [challenge2/README.md](challenge2/README.md)

Le protocole et le code sont en place, les premiers runs pilotes tournent. Rien n'est encore figé.

## Installation

```bash
uv sync
uv run pytest -q
```

`uv sync` installe en mode éditable les paquets `csd` (dans `hackathon/`), `detection` (dans `challenge1/`), `optimization`, `evaluation`, `training` et `transfer_m5` (dans `challenge2/`). Les imports et les `python -m ...` marchent donc depuis la racine. Les commandes de l'énoncé original restent valables, à condition d'ajouter le préfixe `hackathon/` aux chemins `starter/...`.
