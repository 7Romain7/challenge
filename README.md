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

## Challenge 1 : détection → [challenge1/README.md](challenge1/README.md)

Trois étapes, chacune motivée par la limite de la précédente :

1. **Régression logistique** sur deux cartes physiques : robuste, mais plafonne à bas SNR.
2. **U-Net** sur un générateur infini : nettement meilleur, mais il s'effondre sur les artefacts qu'il n'a jamais vus.
3. **U-Net + familles d'artefacts**, testé en **leave-one-family-out** : on entraîne sans une famille, puis on teste sur elle. La randomisation protège ce qu'elle couvre, pas un artefact nouveau.

| test, 400 scènes | Régression logistique | U-Net (lowsnr) |
|---|---|---|
| obj F1 (interdots trouvés) | 0,924 | **0,979** |
| frames du challenge 2 (obj F1) | 0,62 | **0,96** |
| robustesse aux artefacts (1 = insensible) | **0,935** | 0,872 |

## Challenge 2 : optimisation → [challenge2/README.md](challenge2/README.md)

Le protocole et le code sont en place, les premiers runs pilotes tournent. Rien n'est encore figé.

## Installation

```bash
uv sync
uv run pytest -q
```

`uv sync` installe en mode éditable les paquets `csd` (dans `hackathon/`), `detection` (dans `challenge1/`), `optimization`, `evaluation`, `training` et `transfer_m5` (dans `challenge2/`). Les imports et les `python -m ...` marchent donc depuis la racine. Les commandes de l'énoncé original restent valables, à condition d'ajouter le préfixe `hackathon/` aux chemins `starter/...`.
