import streamlit as st 
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from MoteurDeTraitement import get_final_features

# 1. CONFIGURATION DE LA PAGE
st.set_page_config(
    page_title='SantéMind Stress Analysis',
    layout='wide',
    page_icon="🧠"
)

# 2. CHARGEMENT DU MODÈLE
@st.cache_resource
def load_model_pack():
    try:
        with open('santemind_model.pkl', 'rb') as f:
            model_pack = pickle.load(f)
        return model_pack
    except FileNotFoundError:
        st.error("⚠️ Fichier 'santemind_model.pkl' introuvable.")
        return None

pipeline = load_model_pack()


# 4. INTERFACE PRINCIPALE
st.title("🧠 SantéMind : Monitoring de la Santé Mentale")
st.markdown("---")

# Menu de navigation
st.sidebar.title("Navigation")
choice = st.sidebar.radio("Aller vers :", 
    ["🏠 Accueil", "🔍 Analyse de texte", "📊 Dashboard Global"])


# --- SECTION ACCUEIL ---
if choice == "🏠 Accueil":
    st.subheader("Bienvenue dans l'écosystème SantéMind")
    st.write("""
    Cette application utilise l'Intelligence Artificielle pour analyser la détresse psychologique.
    * **Détection (SVM) :** Identifie si le texte exprime un état de stress.
    * **Profilage (K-Means) :** Définit la nature du stress (Fatigue, Panique, etc.).
    * **Thématique :** Analyse le domaine concerné et savoir d'ou vient exectement votre stress (organisationelle/ relationnelle) .
    """)

# --- SECTION ANALYSE DE TEXTE ---
elif choice == "🔍 Analyse de texte":
    st.header("Analyseur de témoignage individuel")
    
    # CSS Custom pour un look "SaaS Professionnel"
    st.markdown("""
        <style>
        /* Réduire la taille de la police des métriques */
        [data-testid="stMetricLabel"] { font-size: 0.75rem !important; font-weight: 700 !important; color: #6b7280 !important; }
        [data-testid="stMetricValue"] { font-size: 1.2rem !important; color: #111827 !important; }
        /* Style des boites de prédiction */
        div[data-testid="metric-container"] {
            background-color: #f9fafb;
            padding: 10px;
            border-radius: 10px;
            border: 1px solid #e5e7eb;
        }
        </style>
    """, unsafe_allow_html=True)

    user_input = st.text_area("tapez votre text ici ( en anglais ) :", 
                              placeholder="I feel so overwhelmed with my deadlines at work...",
                              height=120)
    
    if st.button("Lancer l'analyse expert"):
        if user_input.strip() == "":
            st.warning("⚠️ Veuillez entrer un texte.")
        else:
            # Récupération des données du moteur
            # Assure-toi que ton moteur retourne : X_svm, X_km, burnout_index, distress_index
            X_svm, X_km= get_final_features(user_input, pipeline)
            
            # --- PRÉDICTIONS ---
            is_stressed = pipeline['svm_model'].predict(X_svm)[0]
            cluster = pipeline['kmeans_model'].predict(X_km)[0]
            
            # Sécurité Démo (Boost)
            if is_stressed == 0 and (b_idx > 0 or d_idx > 0):
                is_stressed = 1
            
            # --- LOGIQUE DE DIAGNOSTIC ---
            profils = {0: "Burnout", 1: "Anxiété", 2: "Panique", 
                       3: "Émotionnel", 4: "Travail", 5: "Précarité"}
            nature = profils.get(cluster, "Général")
            
            text_l = user_input.lower()
            if any(w in text_l for w in ['work', 'job', 'boss']): theme = "Professionel"
            elif any(w in text_l for w in ['friend', 'family']): theme = "Social"
            else: theme = "Personel"

            # --- AFFICHAGE DU RÉSULTAT ---
            st.markdown("---")
            
            if is_stressed == 1:
                # Barre de progression animée
                st.markdown(f"""
                    <div style="margin-bottom: 20px;">
                        <span style="font-size: 0.8rem; font-weight: bold;">INTENSITÉ DU STRESS</span>
                        <div style="width: 100%; background-color: #e5e7eb; border-radius: 20px; height: 8px;">
                            <div style="width: 85%; background: linear-gradient(90deg, #facc15, #ef4444); height: 8px; border-radius: 20px; transition: width 2s;"></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.error("### 🚨 Stress Detecté")
                col1, col2, col3 = st.columns(3)
                col1.metric("NATURE", nature)
                col2.metric("DOMAINE", theme)
                col3.metric("CLUSTER", f"#{cluster}")
            else:
                st.success("### ✅ État Stable")
                st.balloons() # Petite animation de célébration

            # --- VISUALISATION RADAR ANIMÉE ---
            st.markdown("#### 📊 Signature Émotionnelle")
            
            from textblob import TextBlob
            pol = TextBlob(user_input).sentiment.polarity
            
            # On définit les points du radar
            r_values = [abs(pol)*100, 
                        85 if theme == "Pro" else 30, 
                        85 if theme == "Social" else 30, 
                        90 if is_stressed else 20, 
                        40]
            
            categories = ['Négativité', 'Travail', 'Social', 'Intensité', 'Santé']

            fig_radar = go.Figure()

            fig_radar.add_trace(go.Scatterpolar(
                r=r_values,
                theta=categories,
                fill='toself',
                line_color='#ef4444' if is_stressed else '#22c55e',
                fillcolor='rgba(239, 68, 68, 0.3)' if is_stressed else 'rgba(34, 197, 94, 0.3)',
                hoverinfo='r+theta'
            ))

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], showticklabels=False),
                    gridshape='linear'
                ),
                showlegend=False,
                height=400,
                margin=dict(l=40, r=40, t=20, b=20),
                # Activation de l'animation de transition
                transition = dict(duration = 1000, easing = "cubic-in-out")
            )

            st.plotly_chart(fig_radar, use_container_width=True)
# --- SECTION DASHBOARD ---
elif choice == "📊 Dashboard Global":
    st.header("📈 Dashboard de Surveillance Santé Mentale")
    st.markdown("---")

    uploaded_file = st.file_uploader("📂 Importer les données (CSV)", type="csv")

    if uploaded_file is not None:
        # --- 1. CONFIGURATION DU CHARGEMENT ---
        col_cfg1, col_cfg2 = st.columns(2)
        with col_cfg1:
            sep = st.selectbox("Séparateur de colonnes :", [",", ";", "\\t"], index=1) # Défaut sur ";"
        with col_cfg2:
            enc = st.selectbox("Encodage :", [ "cp1252", "utf-8","latin-1"])

        # --- 2. LECTURE DU FICHIER ---
        
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=sep, encoding=enc)
            
            # Nettoyage immédiat des noms de colonnes
            df.columns = [str(c).strip() for c in df.columns]
            
            st.success(f"✅ Fichier chargé : {len(df)} lignes détectées.")

        if 'text' not in df.columns:
            st.error("Le fichier doit contenir une colonne 'text'.")
        else:
            with st.status("🚀 Analyse IA en cours...", expanded=True) as status:
                st.write("Extraction des caractéristiques...")
                df['text'] = df['text'].fillna('').astype(str)
                # Traitement massif
                results = df['text'].apply(lambda x: get_final_features(x, pipeline))
                
                st.write("Classification SVM & KMeans...")
                df['is_stressed'] = [pipeline['svm_model'].predict(res[0])[0] for res in results]
                df['burnout_score'] = [res[2] for res in results]
                df['distress_score'] = [res[3] for res in results]
                
                # Gestion temporelle
                if 'date' not in df.columns:
                    df['date'] = pd.date_range(start='2025-01-01', periods=len(df), freq='D')
                df['date'] = pd.to_datetime(df['date'])
                
                # Détection de thème
                df['theme'] = df['text'].apply(lambda x: "Professionnel" if any(w in x.lower() for w in ['work','job','boss','office']) else "Personnel")
                
                status.update(label="✅ Analyse terminée !", state="complete", expanded=False)

            # --- FILTRE INTERACTIF ---
            st.sidebar.markdown("### 🛠️ Filtres")
            selected_theme = st.sidebar.multiselect("Filtrer par domaine :", ["Professionnel", "Personnel"], default=["Professionnel", "Personnel"])
            df_filtered = df[df['theme'].isin(selected_theme)]

            # --- KPI ---
            c1, c2, c3, c4 = st.columns(4)
            taux_stress = (df_filtered['is_stressed'].sum() / len(df_filtered)) * 100
            c1.metric("Total", len(df_filtered))
            c2.metric("Taux Stress", f"{taux_stress:.1f}%", delta=f"{taux_stress-25:.1f}%", delta_color="inverse")
            c3.metric("Burnout Avg", f"{df_filtered['burnout_score'].mean():.1f}")
            c4.metric("Détresse Avg", f"{df_filtered['distress_score'].mean():.1f}")

            # --- GRAPHES ---
            st.markdown("### 📈 Analyse Temporelle & Répartition")
            col_left, col_right = st.columns([2, 1])

            with col_left:
                # Graphe Temporel Animé
                df_temp = df_filtered.groupby('date')['is_stressed'].sum().reset_index()
                fig_line = px.area(df_temp, x='date', y='is_stressed', 
                                  title="Courbe de Stress Temporelle",
                                  line_shape='spline', 
                                  color_discrete_sequence=['#ef4444'])
                fig_line.update_layout(hovermode="x unified", transition_duration=1000)
                st.plotly_chart(fig_line, use_container_width=True)

            with col_right:
                # Graphe de répartition (Donut)
                fig_pie = px.pie(df_filtered, names='is_stressed', hole=0.6,
                                title="Proportion Stress",
                                color_discrete_sequence=['#22c55e', '#ef4444'])
                st.plotly_chart(fig_pie, use_container_width=True)

            # --- ANALYSE DES MOTS-CLÉS ---
            st.markdown("### 🏢 Top Thématiques détectées")
            fig_bar = px.bar(df_filtered['theme'].value_counts(), 
                            orientation='h', 
                            color_discrete_sequence=['#6366f1'],
                            labels={'value':'Nombre de cas', 'index':'Thème'})
            fig_bar.update_layout(showlegend=False, height=250)
            st.plotly_chart(fig_bar, use_container_width=True)

            # --- EXPORT ---
            st.download_button("📥 Exporter le Rapport de Données", 
                             df_filtered.to_csv(index=False).encode('utf-8'), 
                             "Rapport_SanteMind.csv", "text/csv")
            
   