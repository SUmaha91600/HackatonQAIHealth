import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import cv2

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score

from skimage import measure
from skimage.morphology import remove_small_objects
from skimage.filters import threshold_multiotsu

st.set_page_config(page_title="Classic/Quantum Health Predictor", page_icon="🩺", layout="wide")

# ------------------------------------------------------------------------------------------
# Données & entraînement modèles (5 features)
# ------------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    data = load_breast_cancer()
    return data.data, data.target, data.feature_names, data

X_full, y, feature_names, data_obj = load_data()

# 5 features : mean radius, mean texture, mean perimeter, mean area, mean compactness (index 5)
FEAT_IDX = [0, 1, 2, 3, 5]
CHOSEN_FEATURES = feature_names[FEAT_IDX]

@st.cache_resource(show_spinner=False)
def train_models(X, y, feat_idx):
    Xsel = X[:, feat_idx]
    X_train, X_test, y_train, y_test = train_test_split(
        Xsel, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
    Xs_train = scaler.transform(X_train)
    Xs_test  = scaler.transform(X_test)

    logreg = LogisticRegression(max_iter=1000).fit(Xs_train, y_train)
    svc    = SVC(kernel="rbf", probability=True).fit(Xs_train, y_train)

    yp_lr  = logreg.predict(Xs_test)
    yp_svm = svc.predict(Xs_test)

    scores = {
        "acc_lr": accuracy_score(y_test, yp_lr),
        "f1_lr":  f1_score(y_test, yp_lr),
        "acc_svm": accuracy_score(y_test, yp_svm),
        "f1_svm":  f1_score(y_test, yp_svm),
    }

    pca = PCA(n_components=2).fit(Xs_train)
    return scaler, logreg, svc, scores, (X_train, X_test, y_train, y_test, Xs_train, Xs_test), pca

scaler, logreg, svc, scores, splits, pca = train_models(X_full, y, FEAT_IDX)
X_train, X_test, y_train, y_test, Xs_train, Xs_test = splits

# ------------------------------------------------------------------------------------------
# UI
# ------------------------------------------------------------------------------------------
st.title(f"🩺 Classic Health Predictor — Breast Cancer ({len(CHOSEN_FEATURES)} features)")
tabs = st.tabs(["🔢 Sliders (dataset)", "🖼️ Upload image (démo)"])

# ==========================================================================================
# Onglet 1 — Sliders (dataset sklearn)
# ==========================================================================================
with tabs[0]:
    st.subheader("Données & Modèles (sur dataset sklearn)")

    with st.expander("Voir les mesures utilisées", expanded=False):
        for i, n in enumerate(CHOSEN_FEATURES, 1):
            st.markdown(f"**{i}. {n}** *(valeurs issues d’images microscopiques ; adimensionnelles)*")

    colA, colB = st.columns(2)
    with colA:
        st.metric("Logistic Regression — Accuracy", f"{scores['acc_lr']:.3f}")
        st.metric("Logistic Regression — F1", f"{scores['f1_lr']:.3f}")
    with colB:
        st.metric("SVM RBF — Accuracy", f"{scores['acc_svm']:.3f}")
        st.metric("SVM RBF — F1", f"{scores['f1_svm']:.3f}")

    st.divider()
    st.subheader("🧪 Tester une observation (glissez les curseurs)")
    X_sel_all = X_full[:, FEAT_IDX]
    mins = X_sel_all.min(axis=0); maxs = X_sel_all.max(axis=0); meds = np.median(X_sel_all, axis=0)

    sliders = []
    cols = st.columns(len(CHOSEN_FEATURES))
    for i, name in enumerate(CHOSEN_FEATURES):
        sliders.append(cols[i].slider(name, float(mins[i]), float(maxs[i]), float(meds[i])))

    x_raw = np.array([sliders], dtype=float)
    x_scaled = scaler.transform(x_raw)

    p_lr  = float(logreg.predict_proba(x_scaled)[0,1])
    p_svm = float(svc.predict_proba(x_scaled)[0,1])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔵 Logistic Regression")
        st.write(f"Probabilité **malin** : **{p_lr:.2f}**")
        st.success("Prédiction : **Bénin**") if p_lr < 0.5 else st.error("Prédiction : **Malin**")
    with col2:
        st.markdown("### 🟣 SVM (RBF)")
        st.write(f"Probabilité **malin** : **{p_svm:.2f}**")
        st.success("Prédiction : **Bénin**") if p_svm < 0.5 else st.error("Prédiction : **Malin**")

    st.caption("⚠️ Démo pédagogique : ce n’est **pas un outil médical**. Consulter un professionnel de santé.")

    st.divider()
    st.subheader("🧠 Importance des features (LogReg |coef|)")
    coefs = np.abs(logreg.coef_[0]); imp = coefs / (coefs.sum() + 1e-12)
    fig_imp, ax_imp = plt.subplots()
    ax_imp.barh([str(n) for n in CHOSEN_FEATURES], imp)
    ax_imp.set_xlabel("Importance relative"); ax_imp.invert_yaxis()
    st.pyplot(fig_imp)

# ==========================================================================================
# Onglet 2 — Upload image (démo non médicale)
# ==========================================================================================
with tabs[1]:
    st.subheader("Upload d'image (démo éducative — non médical)")

    col_up1, col_up2 = st.columns([3,1])
    with col_up1:
        up = st.file_uploader("Charge une image PNG/JPG", type=["png","jpg","jpeg"])
    with col_up2:
        use_demo = st.checkbox("Image de démo", value=False)

    # Réglages de base
    st.markdown("**Réglages (si besoin)**")
    cA, cB, cC = st.columns(3)
    blur_ks   = cA.slider("Flou (Gaussian ksize)", 3, 11, 5, step=2)
    min_area  = cB.slider("Aire min (px²)", 50, 10000, 300)
    circ_min  = cC.slider("Circularité min (0–1)", 0.30, 0.98, 0.50)

    # Réglages avancés
    cD, cE, cF = st.columns(3)
    perc_thr   = cD.slider("Percentile de seuil (%)", 70, 99, 90)
    ring_w     = cE.slider("Largeur de l’anneau (px)", 3, 30, 12)
    border_m   = cF.slider("Marge à éviter (px)", 0, 40, 10)

    # Charger image
    if use_demo:
        img = np.zeros((300, 480), np.uint8)
        cv2.circle(img, (140,150), 40, 220, -1)      # lésion 1
        cv2.circle(img, (320,140), 18, 240, -1)      # lésion 2
        cv2.ellipse(img, (260,210), (45,25), 20, 0, 360, 210, -1)  # lésion irrégulière
    elif up is not None:
        file_bytes = np.asarray(bytearray(up.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    else:
        img = None

    if img is None:
        st.info("Charge une image ou coche **Image de démo**.")
        st.stop()

    H, W = img.shape

    # ========= 1) Prétraitements =========
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    eq = clahe.apply(img)
    small = cv2.GaussianBlur(eq, (5,5), 0)
    large = cv2.GaussianBlur(eq, (31,31), 0)
    band  = cv2.normalize(cv2.subtract(small, large), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    den   = cv2.GaussianBlur(band, (blur_ks, blur_ks), 0)

    # ========= 2) Multi-seuillage (percentile OU Multi-Otsu) =========
    try:
        otsu_levels = threshold_multiotsu(den, classes=3)
        th_mo = (den >= otsu_levels[-1]).astype(np.uint8) * 255
    except Exception:
        th_mo = None

    thr_val = int(np.percentile(den, perc_thr))
    th_pct  = (den >= thr_val).astype(np.uint8) * 255
    th = cv2.bitwise_or(th_pct, th_mo) if th_mo is not None else th_pct

    # Nettoyage (ouvrir/fermer) + remplissage de trous
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8), iterations=1)

    # ========= 3) Étiquetage et features par lésion =========
    th_fuse = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=1)
    lab = measure.label(th_fuse, connectivity=2)
    lab = remove_small_objects(lab, min_size=max(20, int(min_area*0.2)))
    regions = measure.regionprops(lab, intensity_image=img)

    lesions = []
    for r in regions:
        A = float(r.area)
        if A < min_area:
            continue

        # pénalisation bord
        minr, minc, maxr, maxc = r.bbox
        touches_border = (minc <= border_m or maxc >= W-1-border_m or
                          minr <= border_m or maxr >= H-1-border_m)
        border_pen = 0.5 if touches_border else 1.0

        P = float(r.perimeter) if r.perimeter is not None else 0.0
        if P <= 0:
            continue
        circ = 4*np.pi*A/(P*P)

        eq_radius = float(np.sqrt(A/np.pi))
        mean_int  = float(r.mean_intensity) if r.mean_intensity is not None else float(img[r.coords[:,0], r.coords[:,1]].mean())
        std_int   = float(img[r.coords[:,0], r.coords[:,1]].std())
        compactness = (P*P)/(4*np.pi*A) if A>0 else np.nan

        ecc      = float(r.eccentricity)
        solidity = float(r.solidity) if r.solidity is not None else np.nan
        extent   = float(r.extent) if r.extent is not None else np.nan
        maj_ax   = float(r.major_axis_length) if r.major_axis_length is not None else np.nan
        min_ax   = float(r.minor_axis_length) if r.minor_axis_length is not None else np.nan
        orient   = float(r.orientation) if r.orientation is not None else np.nan

        # contraste d’anneau
        fill = np.zeros((H,W), np.uint8); fill[r.coords[:,0], r.coords[:,1]] = 1
        dil  = cv2.dilate(fill, np.ones((ring_w, ring_w), np.uint8), iterations=1).astype(bool)
        ring = np.logical_and(dil, np.logical_not(fill))
        if ring.sum() > 0:
            ring_mean = float(img[ring].mean())
            ring_contrast = max(mean_int - ring_mean, 0.0)
        else:
            ring_contrast = 0.0

        eps = 1e-3
        shape_weight = max(circ, 0.4)  # on ne tue pas complètement les formes irrégulières
        score = mean_int * A * (ring_contrast + eps) * border_pen * shape_weight

        lesions.append({
            "label": r.label,
            "score": score,
            "area_px2": A,
            "perimeter_px": P,
            "equiv_radius_px": eq_radius,
            "mean_intensity": mean_int,
            "std_intensity": std_int,     # = mean texture
            "circularity": circ,
            "compactness": compactness,
            "eccentricity": ecc,
            "solidity": solidity,
            "extent": extent,
            "major_axis": maj_ax,
            "minor_axis": min_ax,
            "orientation": orient,
            "ring_contrast": ring_contrast,
            "touches_border": touches_border,
            "coords": r.coords
        })

    if not lesions:
        st.warning("Aucune lésion détectée avec les réglages actuels. "
                   "Baisse le percentile, diminue l’aire min, ou réduit la marge bord.")
        st.stop()

    lesions.sort(key=lambda d: d["score"], reverse=True)

    # ========= 4) Overlays (contours + heatmap) =========
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for idx, L in enumerate(lesions, 1):
        cc = L["coords"]
        mask_i = np.zeros((H,W), np.uint8); mask_i[cc[:,0], cc[:,1]] = 255
        cnts, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        color = (0,255,0) if idx==1 else (0,180,255)
        if cnts:
            cv2.drawContours(overlay, cnts, -1, color, 2)
            M = cv2.moments(cnts[0])
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
                cv2.putText(overlay, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    score_map = np.zeros((H, W), np.float32)
    max_score = max(L["score"] for L in lesions) if lesions else 1.0
    for L in lesions:
        val = float(L["score"] / max_score) if max_score > 0 else 0.0
        rrcc = L["coords"]
        score_map[rrcc[:,0], rrcc[:,1]] = np.maximum(score_map[rrcc[:,0], rrcc[:,1]], val)

    score_map = cv2.GaussianBlur(score_map, (0,0), sigmaX=3, sigmaY=3)
    score_map = np.clip(score_map, 0, 1)
    heat_uint8 = (score_map * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    img_bgr    = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay_heat = cv2.addWeighted(img_bgr, 1.0, heat_color, 0.45, 0)

    c1, c2 = st.columns(2)
    with c1:
        st.image(img, caption="Image (grayscale)", use_container_width=True)
    with c2:
        st.image(overlay, caption="Lésions détectées (toutes formes)", use_container_width=True)

    st.image(overlay_heat, caption="Heatmap des scores (plus chaud = plus suspect)", use_container_width=True)

    # ========= 5) Tableaux + export CSV =========
    cols_show = ["label","score","area_px2","perimeter_px","equiv_radius_px",
                 "mean_intensity","std_intensity","circularity","compactness",
                 "eccentricity","solidity","extent","major_axis","minor_axis",
                 "orientation","ring_contrast","touches_border"]
    df_lesions = pd.DataFrame([{k: L[k] for k in cols_show} for L in lesions])

    st.subheader("📋 Caractéristiques par lésion")
    st.dataframe(df_lesions.style.format({
        "score":"{:.1f}", "area_px2":"{:.0f}", "perimeter_px":"{:.1f}",
        "equiv_radius_px":"{:.2f}", "mean_intensity":"{:.1f}", "std_intensity":"{:.2f}",
        "circularity":"{:.3f}", "compactness":"{:.3f}",
        "eccentricity":"{:.3f}", "solidity":"{:.3f}", "extent":"{:.3f}",
        "major_axis":"{:.1f}", "minor_axis":"{:.1f}", "orientation":"{:.2f}",
        "ring_contrast":"{:.2f}"
    }), use_container_width=True)

    # ========= PCA 2D en bas de l'onglet Upload =========
    st.subheader("🗺️ Carte 2D (PCA) — Patients vs. lésions (image upload)")

    # 1) Construire X_lesions (5 features) depuis la table des features par lésion (df_feat)
    #    -> adapter les noms si les colonnes diffèrent chez toi
    feat_map = {
        "mean radius":      "equiv_radius_px",
         "mean texture":     "std_intensity",     # (σ d'intensité ≈ texture)
        "mean perimeter":   "perimeter_px",
        "mean area":        "area_px2",
        "mean compactness": "compactness",
    }
    cols_lesions = [feat_map[name] for name in CHOSEN_FEATURES]
    X_lesions = df_lesions[cols_lesions].to_numpy(dtype=float)

    # 2) Clip aux bornes du dataset de référence (évite les saturations)
    mins = X_sel_all.min(axis=0)   # X_sel_all = features (train+test) déjà construites au début
    maxs = X_sel_all.max(axis=0)
    X_lesions_clip = np.clip(X_lesions, mins, maxs)

    # 3) Même scaler + même PCA que les modèles
    X_lesions_scaled = scaler.transform(X_lesions_clip)
    Z_les = pca.transform(X_lesions_scaled)

    # 4) Projeter aussi les patients (déjà scalés : Xs_train, Xs_test)
    Z_tr = pca.transform(Xs_train)
    Z_te = pca.transform(Xs_test)

    # 5) Plot
    fig, ax = plt.subplots()
    ax.scatter(Z_tr[y_train==0,0], Z_tr[y_train==0,1], s=12, alpha=0.25, label="Bénin (train)")
    ax.scatter(Z_tr[y_train==1,0], Z_tr[y_train==1,1], s=12, alpha=0.25, label="Malin (train)")
    ax.scatter(Z_te[y_test==0,0],  Z_te[y_test==0,1],  s=12, alpha=0.25, label="Bénin (test)")
    ax.scatter(Z_te[y_test==1,0],  Z_te[y_test==1,1],  s=12, alpha=0.25, label="Malin (test)")

    ax.scatter(Z_les[:,0], Z_les[:,1], s=90, marker="*", edgecolor="k", linewidth=0.6,
               label="Lésions (upload)")

    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("Carte 2D (PCA) des patients + lésions détectées")
    ax.legend(loc="best")
    st.pyplot(fig)

    # Export CSV (features)
    csv_lesions = df_lesions.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Télécharger les features (CSV)", data=csv_lesions,
                       file_name="lesions_features.csv", mime="text/csv")

    # ========= 6) Prédictions par lésion (mêmes 5 features) + export =========
    st.subheader("🔮 Prédictions par lésion (démo non médicale)")
    X_lesions = []
    for L in lesions:
        X_lesions.append([
            L["equiv_radius_px"],   # mean radius ≈ rayon équiv.
            L["std_intensity"],     # mean texture ≈ écart-type
            L["perimeter_px"],      # mean perimeter
            L["area_px2"],          # mean area
            L["compactness"],       # mean compactness
        ])
    X_lesions = np.array(X_lesions, dtype=float)
    X_lesions_scaled = scaler.transform(X_lesions)

    p_lr  = logreg.predict_proba(X_lesions_scaled)[:,1]
    p_svm = svc.predict_proba(X_lesions_scaled)[:,1]

    df_pred = df_lesions[["label","area_px2","perimeter_px","equiv_radius_px","compactness"]].copy()
    df_pred["prob_malin_LR"]  = np.round(p_lr,  3)
    df_pred["prob_malin_SVM"] = np.round(p_svm, 3)
    st.dataframe(df_pred, use_container_width=True)

    # Export CSV (predictions)
    csv_pred = df_pred.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Télécharger les prédictions (CSV)", data=csv_pred,
                       file_name="lesions_predictions.csv", mime="text/csv")

    st.caption("⚠️ Démo pédagogique. Pour un usage clinique : images DICOM, segmentation validée et modèles entraînés sur des données médicales réelles.")
