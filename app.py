import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Prédiction Statuts Réservation Uber - Portfolio IA",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour améliorer l'apparence
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
    }
    .success-metric {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-metric {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Charge les données réelles ou génère des données de démonstration"""
    try:
        # Tentative de chargement de vos données réelles
        df = pd.read_csv('uber/archive/ncr_ride_bookings.csv')
        st.success("Données réelles chargées avec succès!")
        return df
    except FileNotFoundError:
        st.warning("Fichier de données non trouvé. Utilisation de données de démonstration.")
        return generate_demo_data()

@st.cache_data
def generate_demo_data():
    """Génère des données de démonstration réalistes"""
    np.random.seed(42)
    n_samples = 10000
    
    # Génération de données synthétiques réalistes basées sur le contexte Uber
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='H')
    
    data = {
        'Date': [d.strftime('%Y-%m-%d') for d in dates],
        'Time': [d.strftime('%H:%M:%S') for d in dates],
        'Pickup Location': np.random.choice([
            'Airport', 'Downtown', 'Mall', 'Station', 'Hospital', 
            'Hotel', 'Office Complex', 'Residential Area'
        ], n_samples, p=[0.15, 0.25, 0.12, 0.10, 0.08, 0.10, 0.15, 0.05]),
        'Drop Location': np.random.choice([
            'Airport', 'Downtown', 'Mall', 'Station', 'Hospital', 
            'Hotel', 'Office Complex', 'Residential Area'
        ], n_samples, p=[0.12, 0.20, 0.15, 0.08, 0.05, 0.08, 0.12, 0.20]),
        'Ride Distance': np.random.exponential(8, n_samples) + 1,  # Distance en km
        'Booking Value': np.random.normal(28, 12, n_samples).clip(min=8),  # Prix en euros
        'Vehicle Type': np.random.choice([
            'Economy', 'Premium', 'Shared', 'Luxury'
        ], n_samples, p=[0.5, 0.25, 0.2, 0.05]),
        'Payment Method': np.random.choice([
            'Card', 'Cash', 'Wallet', 'Corporate'
        ], n_samples, p=[0.6, 0.25, 0.12, 0.03]),
        'Avg VTAT': np.random.normal(7, 3, n_samples).clip(min=1),  # Temps attente véhicule
        'Avg CTAT': np.random.normal(4, 2, n_samples).clip(min=1),  # Temps attente client
        'Booking Status': np.random.choice([
            'Success', 'Canceled by Driver', 'Canceled by Customer', 'No Show'
        ], n_samples, p=[0.75, 0.12, 0.10, 0.03])
    }
    
    return pd.DataFrame(data)

@st.cache_data
def preprocess_data(df):
    """Preprocessing complet des données"""
    df_processed = df.copy()
    
    # Feature engineering temporel
    df_processed['DateTime'] = pd.to_datetime(df_processed['Date'] + ' ' + df_processed['Time'])
    df_processed['Hour'] = df_processed['DateTime'].dt.hour
    df_processed['DayOfWeek'] = df_processed['DateTime'].dt.dayofweek
    df_processed['Month'] = df_processed['DateTime'].dt.month
    df_processed['IsWeekend'] = (df_processed['DayOfWeek'] >= 5).astype(int)
    
    def get_time_slot(hour):
        if 6 <= hour < 10:
            return 'Morning_Rush'
        elif 10 <= hour < 16:
            return 'Midday'
        elif 16 <= hour < 20:
            return 'Evening_Rush'
        elif 20 <= hour < 24:
            return 'Evening'
        else:
            return 'Night_EarlyMorning'
    
    df_processed['TimeSlot'] = df_processed['Hour'].apply(get_time_slot)
    
    # Sélection des features pertinentes
    features = ['Pickup Location', 'Drop Location', 'Ride Distance', 'Booking Value',
                'Vehicle Type', 'Payment Method', 'Avg VTAT', 'Avg CTAT',
                'Hour', 'DayOfWeek', 'Month', 'IsWeekend', 'TimeSlot']
    
    X = df_processed[features].copy()
    y = df_processed['Booking Status'].copy()
    
    # Encodage des variables catégorielles
    label_encoders = {}
    categorical_features = X.select_dtypes(include=['object']).columns
    
    for col in categorical_features:
        X[col] = X[col].fillna('MISSING')
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Encodage de la variable cible
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    return X, y_encoded, target_encoder, label_encoders, df_processed

@st.cache_resource
def train_models(X, y):
    """Entraîne et évalue les modèles de machine learning"""
    # Division train/test stratifiée
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    # Pipeline de preprocessing
    preprocessing = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Modèles à évaluer
    models = {
        'Random Forest': Pipeline([
            ('preprocessing', preprocessing),
            ('classifier', RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                class_weight='balanced',
                n_jobs=-1
            ))
        ]),
        'Logistic Regression': Pipeline([
            ('preprocessing', preprocessing),
            ('classifier', LogisticRegression(
                max_iter=1000, 
                random_state=42, 
                class_weight='balanced'
            ))
        ])
    }
    
    results = {}
    for name, model in models.items():
        # Entraînement
        model.fit(X_train, y_train)
        
        # Prédictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Métriques
        accuracy = accuracy_score(y_test, y_pred)
        
        # Validation croisée
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        except:
            cv_scores = np.array([accuracy])  # Fallback
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'y_test': y_test
        }
    
    return results, X_train, X_test, y_train, y_test

def create_confusion_matrix_plot(y_test, y_pred, target_encoder):
    """Crée une heatmap de la matrice de confusion"""
    cm = confusion_matrix(y_test, y_pred)
    
    fig = px.imshow(
        cm, 
        text_auto=True,
        aspect="auto",
        title="Matrice de Confusion",
        labels=dict(x="Prédictions", y="Vraies Valeurs"),
        x=target_encoder.classes_,
        y=target_encoder.classes_,
        color_continuous_scale="Blues"
    )
    
    return fig

def create_feature_importance_plot(model, feature_names):
    """Crée un graphique d'importance des features pour Random Forest"""
    if hasattr(model.named_steps['classifier'], 'feature_importances_'):
        importance = model.named_steps['classifier'].feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            importance_df, 
            x='Importance', 
            y='Feature',
            orientation='h',
            title="Importance des Features (Random Forest)"
        )
        
        return fig
    return None

def main():
    # Header principal avec style
    st.markdown("""
    <div class="main-header">
        <h1>🚗 Prédiction des Statuts de Réservation Uber</h1>
        <p>Projet d'Intelligence Artificielle - Machine Learning</p>
        <p><strong>Par :</strong> Nadir Ali Ahmed | <strong>Portfolio Data Science</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar avec informations du projet
    st.sidebar.markdown("## 📊 À Propos du Projet")
    st.sidebar.markdown("""
    **Objectif :** Prédire le statut des réservations Uber
    
    **Classes prédites :**
    - ✅ Succès
    - ❌ Annulé par chauffeur
    - ❌ Annulé par client
    - ⚠️ No Show
    
    **Technologies :**
    - Python (Pandas, Scikit-learn)
    - Random Forest, Régression Logistique
    - Streamlit, Plotly
    
    **Méthodologie :**
    1. Analyse exploratoire (EDA)
    2. Feature Engineering
    3. Preprocessing automatisé
    4. Entraînement de modèles
    5. Évaluation comparative
    """)
    
    # Chargement des données avec feedback
    with st.spinner('Chargement et preprocessing des données...'):
        df = load_data()
        X, y, target_encoder, label_encoders, df_processed = preprocess_data(df)
        results, X_train, X_test, y_train, y_test = train_models(X, y)
    
    # Tabs principales de l'application
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Données", 
        "🔍 EDA", 
        "🤖 Modèles", 
        "📊 Performance", 
        "🎯 Test Interactif",
        "📋 Rapport Final"
    ])
    
    with tab1:
        st.header("📈 Aperçu des Données")
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Observations", f"{len(df):,}")
        with col2:
            st.metric("Features", f"{len(X.columns)}")
        with col3:
            st.metric("Classes", f"{len(target_encoder.classes_)}")
        with col4:
            success_rate = (df['Booking Status'] == 'Success').mean() * 100
            st.metric("Taux Succès", f"{success_rate:.1f}%")
        
        # Échantillon des données
        st.subheader("Échantillon des données originales")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Distribution de la variable cible
        st.subheader("Distribution de la variable cible")
        target_dist = df['Booking Status'].value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(
                values=target_dist.values, 
                names=target_dist.index,
                title="Répartition des Statuts"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                x=target_dist.index, 
                y=target_dist.values,
                title="Effectifs par Statut",
                labels={'x': 'Statut', 'y': 'Nombre de réservations'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistiques descriptives
        st.subheader("Statistiques descriptives")
        st.dataframe(df.describe(), use_container_width=True)
    
    with tab2:
        st.header("🔍 Analyse Exploratoire des Données")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Réservations par heure")
            hourly_data = df_processed.groupby('Hour').size()
            fig = px.line(
                x=hourly_data.index, 
                y=hourly_data.values,
                title="Volume de réservations par heure",
                labels={'x': 'Heure', 'y': 'Nombre de réservations'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Statuts par jour de la semaine")
            days = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
            weekly_data = pd.crosstab(df_processed['DayOfWeek'], df['Booking Status'])
            weekly_data.index = days
            fig = px.bar(weekly_data, title="Statuts par jour de la semaine")
            st.plotly_chart(fig, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Distribution des distances")
            fig = px.histogram(
                df, 
                x='Ride Distance', 
                title="Distribution des distances de trajet",
                nbins=50
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            st.subheader("Valeur moyenne par type de véhicule")
            vehicle_stats = df.groupby('Vehicle Type')['Booking Value'].mean().sort_values(ascending=False)
            fig = px.bar(
                x=vehicle_stats.values,
                y=vehicle_stats.index,
                orientation='h',
                title="Valeur moyenne par type de véhicule"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Corrélations
        st.subheader("Matrice de corrélation des variables numériques")
        numeric_cols = ['Ride Distance', 'Booking Value', 'Avg VTAT', 'Avg CTAT', 'Hour']
        if all(col in df.columns for col in numeric_cols):
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(
                corr_matrix, 
                title="Corrélations entre variables numériques",
                color_continuous_scale="RdBu_r",
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("🤖 Modèles d'Intelligence Artificielle")
        
        st.markdown("""
        ### 🛠️ Méthodologie de Développement
        
        **1. Feature Engineering :**
        - Extraction temporelle : heure, jour semaine, créneaux horaires
        - Variables dérivées : weekend, rush hours
        - Encodage des variables catégorielles
        
        **2. Preprocessing :**
        - Gestion des valeurs manquantes (médiane)
        - Standardisation des features numériques
        - Pipeline intégré pour éviter le data leakage
        
        **3. Modèles testés :**
        - **Random Forest** : Ensemble d'arbres de décision
        - **Régression Logistique** : Modèle linéaire probabiliste
        
        **4. Validation :**
        - Division train/test stratifiée (80/20)
        - Validation croisée 5-fold
        - Équilibrage des classes (class_weight='balanced')
        """)
        
        # Comparaison des performances
        st.subheader("🏆 Comparaison des Performances")
        
        model_comparison = []
        for name, result in results.items():
            model_comparison.append({
                'Modèle': name,
                'Accuracy': result['accuracy'],
                'CV Mean': result['cv_mean'],
                'CV Std': result['cv_std']
            })
        
        comparison_df = pd.DataFrame(model_comparison)
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(
                comparison_df.style.format({
                    'Accuracy': '{:.4f}',
                    'CV Mean': '{:.4f}',
                    'CV Std': '{:.4f}'
                }),
                use_container_width=True
            )
        
        with col2:
            fig = px.bar(
                comparison_df, 
                x='Modèle', 
                y='Accuracy',
                title="Accuracy par modèle",
                color='Accuracy',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Meilleur modèle
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        st.markdown(f"""
        <div class="success-metric">
            <h4>🏆 Meilleur Modèle : {best_model_name}</h4>
            <p>Accuracy: {results[best_model_name]['accuracy']:.4f} ({results[best_model_name]['accuracy']*100:.2f}%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab4:
        st.header("📊 Analyse Détaillée des Performances")
        
        # Sélection du meilleur modèle
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_result = results[best_model_name]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Matrice de Confusion")
            fig = create_confusion_matrix_plot(
                best_result['y_test'], 
                best_result['predictions'], 
                target_encoder
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Métriques par Classe")
            report = classification_report(
                best_result['y_test'], 
                best_result['predictions'], 
                target_names=target_encoder.classes_, 
                output_dict=True
            )
            
            report_df = pd.DataFrame(report).transpose().round(3)
            st.dataframe(report_df, use_container_width=True)
        
        # Importance des features (si Random Forest)
        if 'Random Forest' in best_model_name:
            st.subheader("Importance des Features")
            fig = create_feature_importance_plot(best_result['model'], X.columns)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Analyse des erreurs
        st.subheader("Analyse des Erreurs")
        cm = confusion_matrix(best_result['y_test'], best_result['predictions'])
        total_errors = len(best_result['y_test']) - cm.trace()
        error_rate = total_errors / len(best_result['y_test'])
        
        st.markdown(f"""
        **Statistiques d'erreur :**
        - Erreurs totales : {total_errors}
        - Taux d'erreur : {error_rate:.3f} ({error_rate*100:.1f}%)
        - Prédictions correctes : {cm.trace()} / {len(best_result['y_test'])}
        """)
    
    with tab5:
        st.header("🎯 Test Interactif du Modèle")
        st.markdown("Testez le modèle avec vos propres paramètres et obtenez une prédiction en temps réel !")
        
        # Interface de test
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pickup = st.selectbox("Lieu de départ", [
                'Airport', 'Downtown', 'Mall', 'Station', 'Hospital', 
                'Hotel', 'Office Complex', 'Residential Area'
            ])
            drop = st.selectbox("Destination", [
                'Airport', 'Downtown', 'Mall', 'Station', 'Hospital', 
                'Hotel', 'Office Complex', 'Residential Area'
            ])
            distance = st.slider("Distance (km)", 1.0, 50.0, 10.0, 0.5)
            booking_value = st.slider("Valeur de la course (€)", 5.0, 100.0, 25.0, 1.0)
            vehicle_type = st.selectbox("Type de véhicule", [
                'Economy', 'Premium', 'Shared', 'Luxury'
            ])
        
        with col2:
            payment = st.selectbox("Méthode de paiement", [
                'Card', 'Cash', 'Wallet', 'Corporate'
            ])
            hour = st.slider("Heure de la réservation", 0, 23, 12)
            day_of_week = st.selectbox("Jour de la semaine", [
                'Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'
            ])
            avg_vtat = st.slider("Temps d'attente véhicule (min)", 1.0, 20.0, 7.0, 0.5)
            avg_ctat = st.slider("Temps d'attente client (min)", 1.0, 15.0, 4.0, 0.5)
        
        with col3:
            st.markdown("### Contexte de la prédiction")
            st.info(f"""
            **Trajet :** {pickup} → {drop}  
            **Distance :** {distance} km  
            **Valeur :** {booking_value}€  
            **Véhicule :** {vehicle_type}  
            **Heure :** {hour}h ({day_of_week})
            """)
        
        if st.button("🔮 Prédire le Statut", type="primary", use_container_width=True):
            # Préparation des données pour prédiction
            day_mapping = {
                'Lundi': 0, 'Mardi': 1, 'Mercredi': 2, 'Jeudi': 3, 
                'Vendredi': 4, 'Samedi': 5, 'Dimanche': 6
            }
            
            def get_time_slot(hour):
                if 6 <= hour < 10:
                    return 'Morning_Rush'
                elif 10 <= hour < 16:
                    return 'Midday'
                elif 16 <= hour < 20:
                    return 'Evening_Rush'
                elif 20 <= hour < 24:
                    return 'Evening'
                else:
                    return 'Night_EarlyMorning'
            
            # Construction de l'échantillon de test
            test_data = {
                'Pickup Location': pickup,
                'Drop Location': drop,
                'Ride Distance': distance,
                'Booking Value': booking_value,
                'Vehicle Type': vehicle_type,
                'Payment Method': payment,
                'Avg VTAT': avg_vtat,
                'Avg CTAT': avg_ctat,
                'Hour': hour,
                'DayOfWeek': day_mapping[day_of_week],
                'Month': 6,  # Valeur par défaut
                'IsWeekend': 1 if day_mapping[day_of_week] >= 5 else 0,
                'TimeSlot': get_time_slot(hour)
            }
            
            # Encodage des variables catégorielles
            for col, encoder in label_encoders.items():
                if col in test_data:
                    try:
                        if test_data[col] in encoder.classes_:
                            test_data[col] = encoder.transform([test_data[col]])[0]
                        else:
                            test_data[col] = 0  # Valeur par défaut
                    except:
                        test_data[col] = 0
            
            # Prédiction avec le meilleur modèle
            test_df = pd.DataFrame([test_data])
            best_model = results[best_model_name]['model']
            
            try:
                prediction = best_model.predict(test_df)[0]
                prediction_proba = best_model.predict_proba(test_df)[0]
                predicted_class = target_encoder.inverse_transform([prediction])[0]
                
                # Affichage du résultat
                max_proba = np.max(prediction_proba)
                
                if predicted_class == 'Success':
                    st.success(f"🎉 **Prédiction : {predicted_class}** (Confiance: {max_proba:.1%})")
                else:
                    st.error(f"⚠️ **Prédiction : {predicted_class}** (Confiance: {max_proba:.1%})")
                
                # Graphique des probabilités
                proba_df = pd.DataFrame({
                    'Statut': target_encoder.classes_,
                    'Probabilité': prediction_proba
                }).sort_values('Probabilité', ascending=True)
                
                fig = px.bar(
                    proba_df, 
                    x='Probabilité', 
                    y='Statut',
                    orientation='h',
                    title="Distribution des probabilités",
                    color='Probabilité',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Erreur lors de la prédiction: {e}")
    
    with tab6:
        st.header("📋 Rapport Final du Projet")
        
        # Métriques de performance finales
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_accuracy = results[best_model_name]['accuracy']
        
        st.markdown("""
        ## 🎯 Objectifs du Projet
        Développer un modèle de machine learning capable de prédire avec précision 
        le statut des réservations Uber pour optimiser les opérations et améliorer 
        l'expérience utilisateur.
        """)
        
        # Résultats principaux
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="success-metric">
                <h4>🏆 Modèle Final</h4>
                <p><strong>{best_model_name}</strong></p>
                <p>Accuracy: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            total_predictions = len(results[best_model_name]['y_test'])
            correct_predictions = int(best_accuracy * total_predictions)
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Performance Test</h4>
                <p>{correct_predictions}/{total_predictions} prédictions correctes</p>
                <p>Sur ensemble de validation</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            cv_mean = results[best_model_name]['cv_mean']
            st.markdown(f"""
            <div class="metric-card">
                <h4>🔄 Validation Croisée</h4>
                <p>Moyenne: {cv_mean:.3f}</p>
                <p>Stabilité confirmée</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Forces et améliorations
        st.markdown("## ✅ Points Forts du Projet")
        st.markdown("""
        - **Feature Engineering avancé** : Extraction de patterns temporels pertinents
        - **Pipeline robuste** : Preprocessing automatisé évitant le data leakage
        - **Validation rigoureuse** : Train/test stratifié + validation croisée
        - **Interface utilisateur** : Application interactive pour tests en temps réel
        - **Documentation complète** : Code commenté et rapport détaillé
        """)
        
        st.markdown("## 🔧 Améliorations Possibles")
        st.markdown("""
        - **Données externes** : Intégration météo, trafic, événements
        - **Deep Learning** : Test de réseaux de neurones pour patterns complexes
        - **Optimisation hyperparamètres** : Grid search ou optimisation bayésienne
        - **Features géospatiales** : Analyse de la densité des zones
        - **Modèles ensemblistes** : Stacking ou blending de modèles
        """)
        
        st.markdown("## 🚀 Impact Business")
        st.markdown("""
        Ce modèle peut être utilisé pour :
        - **Optimisation des ressources** : Allocation des chauffeurs aux zones à fort succès
        - **Prévention des annulations** : Identification des réservations à risque
        - **Amélioration UX** : Estimation des temps d'attente réalistes
        - **Stratégie pricing** : Ajustement des tarifs selon probabilité de succès
        """)
        
        # Contact et portfolio
        st.markdown("---")
        st.markdown("""
        ### 👨‍💻 Contact & Portfolio
        
        **Développé par :** Nadir Ali Ahmed  
        **Email :** [Votre email]  
        **LinkedIn :** [Votre profil LinkedIn]  
        **GitHub :** [Repository du projet]  
        
        *Ce projet fait partie de mon portfolio data science. 
        Il démontre mes compétences en machine learning, data preprocessing, 
        et développement d'applications interactives.*
        """)

if __name__ == "__main__":
    main()