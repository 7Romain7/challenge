# Challenge 2 : maximiser le contraste des interdots

> **En cours.** Ce dossier contient le protocole, le code et les premiers runs pilotes. Les résultats ne sont pas encore figés : rien ici n'est un chiffre final.

**Tâche.** Trouver les tensions de barrières (g1, g3, g5) qui maximisent le contraste des interdots d'un dispositif simulé inconnu, avec un budget de mesure limité et sans accès à l'état caché du simulateur.

## Ce qui existe

| chemin | contenu |
|---|---|
| [`PROTOCOL.md`](PROTOCOL.md) | protocole écrit **avant** les méthodes : faits mesurés sur le simulateur, pièges physiques, garde-fous, splits de seeds, tests d'abandon |
| [`optimization/`](optimization/) | `BlindExperiment` (pare-feu : `measure`, `scan_1d` et budget en pixels seulement), perception, suivi de la dérive des sticks, score, méthodes (`random`, `bo`, variantes `*_m5` avec le détecteur du challenge 1) |
| [`evaluation/`](evaluation/) | le seul code qui touche la vérité (`reveal`), runner parallèle, splits de seeds |
| [`training/`](training/) | génération de frames et a priori pour les méthodes apprises prévues |
| [`transfer_m5/`](transfer_m5/README.md) | branchement de la régression logistique du challenge 1 dans la perception, balayage du seuil |
| [`results/`](results/) | runs pilotes (`pilot_dev30`, `m5_dev100`) |

## Premiers constats (split dev, non définitifs)

- La recherche aléatoire est une baseline forte. La BO avec suivi des sticks fait mieux qu'elle à budget élevé (R médian @1M px : 0,19 contre 0,74 sur 100 dispositifs).
- Brancher la régression logistique du challenge 1 n'apporte rien. Au plancher de contraste, la limite est le SNR des interdots, pas la qualité du détecteur ([`transfer_m5/`](transfer_m5/README.md)).

```bash
uv run python -m evaluation.run --methods random bo --split dev --n 30 --budget 1000000 --workers 7 --out challenge2/results/pilot.jsonl
```
