# Transfert du détecteur M5_min (challenge 1) vers le front-end du challenge 2

Question : le meilleur détecteur du challenge 1, M5_min, améliore-t-il l'optimisation du contraste si on le branche dans la perception du challenge 2 ?

```bash
uv run python -m detection.export_m5     # gèle M5_min (8 s, poids identiques au protocole v2)
uv run python -m transfer_m5.sweep       # balayage du seuil, stage 1 vs stage 2 (~50 s)
uv run python -m evaluation.run --methods bo bo_m5 bo_m5_fa5 bo_m5_fa20 --split dev --n 100 \
    --budget 1000000 --workers 15 --out challenge2/transfer_m5/results/m5_thr_dev100.jsonl
```

Le benchmark a tourné sur `truite` (16 cœurs) en environ 2 min. La vérité cachée ne sert qu'à l'évaluation.

## 1. Le score de M5_min est indépendant du dataset

Distribution du logit sur les pixels de fond (`results/sweep.txt`) :

| jeu | médiane | σ (MAD) | p99,9 |
|---|---|---|---|
| val (stage 1) | −5,29 | 1,01 | +5,36 |
| test (stage 1) | −5,29 | 1,01 | +5,28 |
| ood_stage2 | −5,33 | 0,98 | −2,17 |
| null (sans stick) | −5,33 | 0,98 | −2,28 |

Le fond est le même partout. Les deux features (matched, s2) sont blanchies par le bruit de chaque image, donc **un seuil fixe correspond déjà à un taux de fausses alarmes constant** (CFAR). L'hypothèse « le seuil dépend du dataset » est fausse, et un seuil CFAR ne changerait rien.

## 2. L'écart entre stage 1 et stage 2 vient du rapport signal/bruit

| seuil | fausses alarmes / scène vide | F1 objets test | F1 ood_stage2 | rappel ood_stage2 |
|---|---|---|---|---|
| −2,50 | 19,2 | 0,65 | 0,44 | 0,73 |
| −2,00 | 4,5 | 0,86 | 0,63 | 0,62 |
| −1,75 | 1,7 | 0,90 | **0,66** | 0,55 |
| **−1,35** (choisi sur val) | ≈ 0,3 | **0,92** | 0,64 | ≈ 0,47 |
| −1,00 | 0,1 | 0,93 | 0,55 | 0,38 |

Aucun seuil ne ramène le stage 2 au niveau du stage 1 : le meilleur F1 stage 2 est 0,66, contre 0,64 au seuil d'origine. Au point de départ du challenge 2, les sticks sont proches du plancher de contraste (|i| ≈ 3) : le p99,9 du fond y est le même que sur les scènes vides. C'est une **limite physique de détection**, pas un défaut de calibration.

## 3. Effet sur l'optimisation (BO, 100 appareils dev, budget 1 M px, apparié)

| front-end | R médian @500k | R médian @1M | succès R ≤ 0,05 | échecs de suivi / run | ΔR vs `bo` [IC 95 %] | Wilcoxon p |
|---|---|---|---|---|---|---|
| filtre actuel (`bo`) | **0,50** | **0,19** | **18 %** | 1,6 | — | — |
| M5_min, seuil −1,35 (`bo_m5`) | 0,79 | 0,28 | 15 % | 8,3 | +0,023 [−0,002 ; +0,087] | 0,013 |
| M5_min, seuil −2,0 (`bo_m5_fa5`) | 0,56 | 0,21 | 14 % | 3,1 | +0,008 [−0,008 ; +0,037] | 0,35 |
| M5_min, seuil −2,5 (`bo_m5_fa20`) | 0,61 | 0,21 | 14 % | 1,8 | +0,004 [−0,016 ; +0,046] | 0,48 |

ΔR > 0 signifie que la variante est pire que `bo`.

- Au seuil du challenge 1, M5_min est **significativement pire**. Il est trop conservateur pour le suivi, qui a besoin d'au moins 3 sticks appariés : il rate la moitié des sticks et le suivi décroche (8,3 échecs par run).
- Avec un seuil plus permissif, fixé par le taux de fausses alarmes sur des scènes vides (ce qu'on peut mesurer en labo dans une zone sans transition), les échecs de suivi retombent au niveau du filtre et la BO retrouve ses performances. Mais M5_min **ne fait pas mieux** que le filtre (p > 0,3).

## Conclusion

M5_min transfère : son score est le même sur les deux distributions. Pour le challenge 2, il suffit de choisir un point de fonctionnement orienté rappel (le recalage tolère les faux positifs). Il n'apporte pourtant aucun gain, parce que la limite est le rapport signal/bruit des sticks au plancher, pas la qualité du détecteur. Le levier pour le challenge 2 est donc ailleurs : l'estimation du score sur les sticks suivis (T3, ρ ≈ 0,4) et l'exploration.

## Limites

- Split dev uniquement : les seuils −2,0 et −2,5 ont été choisis a priori par le taux de fausses alarmes, pas réglés sur ces runs, mais ce n'est pas un test gelé.
- Une seule graine d'algorithme.
- Seuls les seuils ont été balayés ; l'union M5 ∪ filtre n'a pas été testée.
