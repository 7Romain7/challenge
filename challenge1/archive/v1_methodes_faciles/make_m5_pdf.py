"""M5 implementation write-up -> M5_implementation.pdf (needs reportlab; run after make_m5_figs.py)."""
import json
from pathlib import Path

from PIL import Image as PI
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle

H = Path(__file__).parent
F = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("A", str(F / "arial.ttf")))
pdfmetrics.registerFont(TTFont("AB", str(F / "arialbd.ttf")))
pdfmetrics.registerFont(TTFont("AI", str(F / "ariali.ttf")))
pdfmetrics.registerFont(TTFont("C", str(F / "consola.ttf")))
pdfmetrics.registerFontFamily("A", normal="A", bold="AB", italic="AI", boldItalic="AB")
W = json.loads((H / "m5_weights.json").read_text())
R = json.loads((H / "baselines.json").read_text())["M5_logreg"]
T, O = R["test"], R["ood_noise"]

body = ParagraphStyle("b", fontName="A", fontSize=9.6, leading=13.6, spaceAfter=5)
h1 = ParagraphStyle("h1", fontName="AB", fontSize=19, leading=23, spaceAfter=4)
h2 = ParagraphStyle("h2", fontName="AB", fontSize=12.5, leading=16, spaceBefore=11, spaceAfter=4,
                    textColor=colors.HexColor("#1f3a5f"))
small = ParagraphStyle("s", parent=body, fontSize=8.2, leading=11, textColor=colors.HexColor("#444444"))
code = ParagraphStyle("c", fontName="C", fontSize=7.9, leading=10, backColor=colors.HexColor("#f3f4f6"),
                      borderPadding=5, spaceBefore=3, spaceAfter=9)
cell = ParagraphStyle("t", parent=body, fontSize=8.6, leading=11, spaceAfter=0)
M = "<font name='C'>%s</font>"


def P(t, s=body):
    return Paragraph(t, s)


def C(t):
    return Preformatted(t.strip("\n"), code)


def tbl(rows, widths):
    rows = [[P(str(c), cell) for c in r] for r in rows]
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#c8ccd2")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6ecf5"))]))
    return t


def img(p, w):
    iw, ih = PI.open(p).size
    return Image(str(p), width=w, height=w * ih / iw)


s = []
s += [P("Méthode M5 — régression logistique sur cartes multi-échelles", h1),
      P("Challenge 1 · détection des interdots · implémentation dans " + M % "detection/baselines.py", small), Spacer(1, 6)]

s += [P("1. Idée", h2),
      P("M1 à M4 appliquent <i>un seul</i> filtre puis un seuil. Chacun a un défaut propre : le lissage (M1) prend une ligne de charge pour un stick, "
        "la courbure (M3) suit la crête des lignes de charge, les bâtonnets (M2) sont corrects mais figés dans une forme. "
        "M5 ne choisit pas : elle calcule six cartes complémentaires pour chaque pixel et <b>apprend la combinaison linéaire</b> "
        "qui sépare au mieux « pixel d'interdot » de « pixel de fond ». Le modèle est un classifieur par pixel à 7 paramètres "
        "(6 poids + 1 biais), entraîné sur le masque officiel du générateur. Pas de réseau de neurones, pas de torch : tout est numpy/scipy.")]

s += [P("2. Pipeline complet", h2),
      P("Image brute (150×150, bruit blanc + rayures horizontales) → <b>prétraitement</b> → <b>6 cartes</b> → <b>standardisation</b> → "
        "<b>logit = w·x + b</b> → <b>seuil choisi sur val</b> → masque binaire.")]

s += [P("2.1 Prétraitement " + M % "prep()" + " : image en unités de σ du bruit", h2),
      C("""
x = x - np.median(x, axis=2, keepdims=True)                  # (1) retire les rayures (médiane par ligne)
mad = np.median(np.abs(x - np.median(x, axis=(1,2), keepdims=True)), axis=(1,2), keepdims=True)
z = -x / (1.4826 * mad + 1e-6)                               # (2) signe  (3) normalisation MAD
"""),
      P("(1) Le bruit en rayures est constant le long de l'axe de scan rapide ; les sticks sont rares, donc la médiane d'une ligne est robuste et ne les absorbe pas. "
        "(2) Les sticks sont négatifs : on inverse le signe pour qu'un stick soit un pic positif. "
        "(3) MAD × 1,4826 estime σ du bruit car le fond couvre presque toute l'image ; un z-score classique serait gonflé par les sticks brillants et effacerait le SNR absolu. "
        "Conséquence : un même seuil vaut pour toutes les scènes.")]

s += [P("2.2 Les six cartes " + M % "features()", h2),
      tbl([["#", "carte", "calcul", "ce qu'elle apporte"],
           ["1", "z", "image prétraitée telle quelle", "position exacte du pixel (résolution sous-pixel conservée)"],
           ["2", "smooth σ=1", "gaussien σ=1 px, puis re-blanchiment MAD", "réduit le bruit blanc, SNR local du stick"],
           ["3", "smooth σ=2", "gaussien σ=2 px, puis re-blanchiment MAD", "contexte plus large, sticks faibles"],
           ["4", "matched", "max sur 3 gabarits (θ = π/4 et π/4 ± 0,2), longueur 5 px, largeur 1 px, moyenne nulle, norme 1, par corrélation",
            "filtre adapté à la forme attendue du stick"],
           ["5", "ridge", "−λ<sub>min</sub> de la Hessienne du lissé (σ=1,2), × σ²", "détecte les structures étroites (crêtes)"],
           ["6", "écart-type local", "√(moyenne locale de z² sur 7×7)", "énergie locale"]],
          [.6 * cm, 2.3 * cm, 6.4 * cm, 7.9 * cm]),
      Spacer(1, 3),
      P("Le re-blanchiment (" + M % "_whiten" + " : soustraction de la médiane, division par 1,4826·MAD de la carte filtrée) remet chaque carte en unités de σ de <i>son</i> fond. "
        "Il est nécessaire après le max sur les gabarits, qui biaise le fond vers le haut.")]

s += [P("2.3 Entraînement " + M % "LogReg.fit()", h2),
      C("""
F = features(z).reshape(-1, 6);  y = m.reshape(-1)                   # un échantillon = un pixel
pos = np.flatnonzero(y)                                              # tous les positifs
neg = rng.choice(np.flatnonzero(~y), 20 * len(pos), replace=False)   # 20 négatifs par positif
X, t = F[idx], y[idx].astype(float)
self.mu, self.sd = X.mean(0), X.std(0) + 1e-6                        # standardisation
X = np.c_[(X - self.mu) / self.sd, np.ones(len(X))]                  # colonne de 1 = biais
w = np.zeros(7)
for _ in range(25):                                                  # IRLS (Newton) pour la logistique
    p = 1 / (1 + np.exp(-X @ w))
    H = (X * (p*(1-p))[:, None]).T @ X + 1e-3*np.eye(7)              # Hessienne + ridge 1e-3
    w += np.linalg.solve(H, X.T @ (t - p))
"""),
      P(f"<b>Données</b> : les 500 premières images de {M % 'data/train'}, soit {W['n_pix']:,} pixels dont <b>{W['n_pos']:,} positifs ({100 * W['n_pos'] / W['n_pix']:.2f} %)</b>. "
        f"Le sous-échantillonnage garde {W['n_pos']:,} positifs et {20 * W['n_pos']:,} négatifs. "
        f"<b>Optimisation</b> : IRLS (équivalent Newton), 25 itérations, régularisation L2 de 10<super>−3</super> sur la Hessienne. Le fit prend {R['fit_s']:.0f} s, "
        "ce qui inclut le calcul des cartes sur les 500 images."),
      P("<b>Pourquoi sous-échantillonner ?</b> Les positifs ne font que 0,35 % des pixels ; sans cela, le classifieur prédirait « fond » partout. "
        "Prix à payer : les probabilités sont fausses par construction (le biais est décalé d'environ ln 20). "
        "On ne les utilise jamais comme probabilités : on compare le <b>logit</b> à un seuil réglé ensuite sur val, ce qui absorbe ce décalage.")]

s += [P("2.4 Inférence et choix du seuil", h2),
      C("""
def scores(self, z):
    F = (features(z) - self.mu) / self.sd
    return F @ self.w[:-1] + self.w[-1]          # logit par pixel
# seuil : balayage sur data/val, on garde celui qui maximise le tol-F1, puis il est gelé
"""),
      P(f"Le seuil retenu est <b>{W['thr']}</b> (en logit). Il est choisi sur {M % 'data/val'} uniquement, jamais sur le test ni sur l'OOD (piège de fuite de sélection). "
        "Le masque final est simplement " + M % "logit &gt; seuil" + ", sans post-traitement morphologique.")]

rows = [["carte", "μ", "σ", "poids w"]]
for n, mu, sd, w in zip(W["names"], W["mu"], W["sd"], W["w"]):
    rows.append([n, f"{mu:.2f}", f"{sd:.2f}", f"{w:+.2f}"])
rows.append(["biais b", "", "", f"{W['b']:+.2f}"])
s += [P("3. Paramètres appris", h2), tbl(rows, [6.2 * cm, 2.8 * cm, 2.8 * cm, 3.2 * cm]), Spacer(1, 4),
      P("Les poids portent sur des features <i>standardisées</i>, donc leur amplitude est comparable. Lecture (mon interprétation, non démontrée par ablation) :"),
      P("• <b>z (+11,8) et smooth σ=1 (−12,1)</b> : de signes opposés et presque égaux. La combinaison ≈ z − lissé(z) est un passe-haut : le modèle réclame un pixel qui ressort "
        "<i>par rapport à son voisinage</i>, ce qui localise le stick au pixel près. C'est probablement ce qui lui donne une IoU strict de 0,81, là où M1 à M4 plafonnent à 0,15–0,30.<br/>"
        "• <b>matched (+17,1)</b> : le plus fort poids positif ; la forme attendue est le principal indice de présence.<br/>"
        "• <b>ridge (−4,2) et écart-type local (−3,5)</b> : poids négatifs. Une ligne de charge est longue et fine, donc très « crête » et très énergétique sur 7×7 ; "
        "le modèle pénalise ces réponses, ce qui est cohérent avec la réduction des faux positifs sur les lignes de charge, que le masque officiel ne labellise pas.")]

s += [KeepTogether([P("4. Ce que voit le modèle sur une scène (test #172)", h2), img(H / "m5_features.png", 17 * cm),
                    P("Les six cartes (haut et milieu), puis le logit et le masque après seuil. Le titre de chaque carte donne son poids.", small)]),
      KeepTogether([img(H / "m5_contributions.png", 17.4 * cm),
                    P("Contribution w·z de chaque carte au logit (rouge : pousse vers « stick », bleu : vers « fond »).", small)])]

s += [P("5. Résultats", h2),
      tbl([["jeu", "tol-F1", "IoU strict", "précision objet", "rappel objet", "F1 objet"],
           ["val (seuil réglé ici)", f"{R['val']['tol_f1']:.3f}", f"{R['val']['iou']:.3f}", f"{R['val']['obj_precision']:.3f}",
            f"{R['val']['obj_recall']:.3f}", f"{R['val']['obj_f1']:.3f}"],
           ["test (400 scènes, seed 2025)", f"{T['tol_f1']:.3f}", f"{T['iou']:.3f}", f"{T['obj_precision']:.3f}",
            f"{T['obj_recall']:.3f}", f"{T['obj_f1']:.3f}"],
           ["test + bruit supplémentaire (OOD)", f"{O['tol_f1']:.3f}", f"{O['iou']:.3f}", f"{O['obj_precision']:.3f}",
            f"{O['obj_recall']:.3f}", f"{O['obj_f1']:.3f}"]],
          [5.6 * cm, 2 * cm, 2.2 * cm, 2.8 * cm, 2.5 * cm, 2 * cm]),
      Spacer(1, 6)]
ra, rb = T["recall_by_amplitude"], O["recall_by_amplitude"]
s += [tbl([["rappel objet vs |amplitude|"] + list(ra)] +
          [["test"] + [f"{v[0]:.2f} (n={v[1]})" for v in ra.values()]] +
          [["test + bruit"] + [f"{v[0]:.2f}" for v in rb.values()]], [4.2 * cm] + [2.6 * cm] * 5),
      Spacer(1, 4),
      P(f"Coût : {T['ms_per_image']:.0f} ms par image sur CPU. "
        "Sous bruit supplémentaire, la baisse vient du rappel (0,93 → 0,87) : les sticks d'amplitude 2 à 8 sont perdus, tandis que la précision monte car le seuil figé devient plus sévère en pratique. "
        "Face à la meilleure méthode sans apprentissage (M2 : tol-F1 0,850, F1 objet 0,829), M5 gagne surtout sur les sticks faibles : "
        "rappel 0,55 contre 0,01 dans [2,4), et 0,96 contre 0,32 dans [4,8).")]

s += [P("6. Limites et précautions", h2),
      P("• <b>Apprise sur le masque du simulateur.</b> Ce masque est un seuil à 0,5 d'un rectangle flouté, fragmenté en environ 2,4 px par composante. M5 apprend ces artefacts de rastérisation, "
        "d'où l'IoU strict de 0,81 ; une partie de l'avance en IoU reproduit donc le générateur plutôt que la physique. Les métriques tolérantes et par objet sont plus fiables.<br/>"
        "• <b>Dépendance au générateur.</b> μ, σ et les poids sont fixés pour ce simulateur (pente ≈ π/4, niveau de bruit, pas de 2 mV). Un autre dispositif ou une autre résolution demanderait de réentraîner ; "
        "le test OOD du dépôt ne couvre que du bruit supplémentaire.<br/>"
        "• <b>Faux positifs sur les lignes de charge.</b> Le modèle linéaire les réduit mais ne les supprime pas : sur la scène #150 il laisse des pointillés le long des lignes (précision objet 0,53 sur cette scène).<br/>"
        "• <b>Pas d'ablation.</b> L'interprétation des poids (section 3) est une lecture, pas une mesure ; retirer une carte et réentraîner la confirmerait.<br/>"
        "• <b>Poids opposés presque égaux (z et smooth σ=1).</b> Signe de colinéarité entre cartes : les poids isolés ne sont pas interprétables, seule leur combinaison l'est.")]

SimpleDocTemplate(str(H / "M5_implementation.pdf"), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                  topMargin=1.7 * cm, bottomMargin=1.6 * cm, title="M5 - implémentation").build(s)
print("ok")
