import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient
import certifi
import dns.resolver

dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = ['8.8.8.8', '8.8.4.4']

from sklearn.datasets import load_diabetes, fetch_california_housing, load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix

# Impor model Regresi, Klasifikasi, & Clustering
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans

# Konfigurasi Halaman
st.set_page_config(
    page_title="Test Akhir Semester - ML & MongoDB",
    page_icon="🤖",
    layout="wide",
)

st.markdown("## 🤖 TEST AKHIR SEMESTER: MACHINE LEARNING & MONGODB")
st.markdown("Aplikasi Streamlit: Regresi, Klasifikasi, Clustering, Retraining, & Komentar Publik")
st.markdown("---")

# --- SIDEBAR: PENGATURAN MODEL ---
st.sidebar.header("⚙️ Pengaturan Model")
task_type = st.sidebar.radio("Pilih Jenis Task", ["Regresi", "Klasifikasi", "Clustering"])

if task_type == "Regresi":
    dataset_name = st.sidebar.selectbox("Pilih Dataset", ["Diabetes", "California Housing"])
    model_name = st.sidebar.selectbox("Pilih Model", [
        "Linear Regression", "Ridge", "Lasso", 
        "Random Forest Regressor", "Gradient Boosting Regressor"
    ])
    test_size = st.sidebar.slider("Test Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    random_state = st.sidebar.number_input("Random State", value=42, step=1)
elif task_type == "Klasifikasi":
    dataset_name = st.sidebar.selectbox("Pilih Dataset", ["Iris", "Breast Cancer", "Wine"])
    model_name = st.sidebar.selectbox("Pilih Model", [
        "Logistic Regression", "Support Vector Classifier (SVC)", 
        "Random Forest Classifier", "K-Nearest Neighbors (KNN)", "Decision Tree Classifier"
    ])
    test_size = st.sidebar.slider("Test Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    random_state = st.sidebar.number_input("Random State", value=42, step=1)
else:
    dataset_name = st.sidebar.selectbox("Pilih Dataset", ["Iris", "Wine"])
    model_name = "K-Means Clustering"
    n_clusters = st.sidebar.slider("Jumlah Cluster (K)", min_value=2, max_value=6, value=3)
    random_state = st.sidebar.number_input("Random State", value=42, step=1)

train_button = st.sidebar.button("🚀 Train Model / Run Clustering", use_container_width=True)

# Fungsi Memuat Dataset
@st.cache_data
def load_data(name):
    if name == "Diabetes":
        data = load_diabetes()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")
    elif name == "California Housing":
        data = fetch_california_housing(as_frame=True)
        X = data.data
        y = data.target
    elif name == "Iris":
        data = load_iris()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")
    elif name == "Breast Cancer":
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")
    elif name == "Wine":
        data = load_wine()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")
    return X, y

X, y = load_data(dataset_name)

# Inisialisasi Session State
if 'trained' not in st.session_state:
    st.session_state.trained = False

if train_button:
    if task_type != "Clustering":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))
        
        if task_type == "Regresi":
            model_map = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "Random Forest Regressor": RandomForestRegressor(random_state=int(random_state)),
                "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=int(random_state))
            }
        else:
            model_map = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Support Vector Classifier (SVC)": SVC(),
                "Random Forest Classifier": RandomForestClassifier(random_state=int(random_state)),
                "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
                "Decision Tree Classifier": DecisionTreeClassifier(random_state=int(random_state))
            }
            
        model = model_map[model_name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if task_type == "Regresi":
            metric_val = float(mean_squared_error(y_test, y_pred))
            metric_name = "MSE"
        else:
            metric_val = float(accuracy_score(y_test, y_pred))
            metric_name = "Accuracy"

        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.y_pred = y_pred
        st.session_state.metric_val = metric_val
        st.session_state.metric_name = metric_name
    else:
        model = KMeans(n_clusters=n_clusters, random_state=int(random_state), n_init=10)
        clusters = model.fit_predict(X)
        st.session_state.clusters = clusters
        st.session_state.metric_name = "Inertia"
        st.session_state.metric_val = float(model.inertia_)

    st.session_state.trained = True
    st.session_state.model = model
    st.session_state.task_type = task_type
    st.session_state.model_name = model_name
    st.session_state.dataset_name = dataset_name
    st.session_state.random_state = random_state
    
    st.success(f"Berhasil menjalankan {model_name}!")

# --- TABS UTAMA ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Training & Visualisasi", 
    "📈 Evaluation Metrics", 
    "💾 Database, Retrain & MongoDB", 
    "💬 Komentar Publik"
])

with tab1:
    if not st.session_state.trained:
        st.warning("⚠️ Silakan atur parameter dan klik tombol di sidebar terlebih dahulu.")
    else:
        st.subheader(f"Visualisasi Hasil - {st.session_state.model_name}")
        st.dataframe(X.head())
        
        fig, ax = plt.subplots(figsize=(8, 5))
        if st.session_state.task_type == "Regresi":
            ax.scatter(st.session_state.y_test, st.session_state.y_pred, alpha=0.7, color='cyan')
            ax.plot([st.session_state.y_test.min(), st.session_state.y_test.max()], 
                    [st.session_state.y_test.min(), st.session_state.y_test.max()], 'r--', lw=2)
            ax.set_xlabel("Nilai Aktual")
            ax.set_ylabel("Nilai Prediksi")
        elif st.session_state.task_type == "Klasifikasi":
            cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Label Prediksi")
            ax.set_ylabel("Label Aktual")
        else:
            ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=st.session_state.clusters, cmap='viridis', alpha=0.7)
            ax.set_xlabel(X.columns[0])
            ax.set_ylabel(X.columns[1])
            ax.set_title("K-Means Clustering Result")
        st.pyplot(fig)

with tab2:
    if not st.session_state.trained:
        st.warning("⚠️ Silakan jalankan model terlebih dahulu di sidebar.")
    else:
        st.subheader("Evaluation Metrics")
        if st.session_state.task_type == "Regresi":
            mse = mean_squared_error(st.session_state.y_test, st.session_state.y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(st.session_state.y_test, st.session_state.y_pred)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("MSE", f"{mse:.4f}")
            c2.metric("RMSE", f"{rmse:.4f}")
            c3.metric("R2 Score", f"{r2:.4f}")
        elif st.session_state.task_type == "Klasifikasi":
            acc = accuracy_score(st.session_state.y_test, st.session_state.y_pred)
            st.metric("Akurasi Model", f"{acc:.4f}")
            report = classification_report(st.session_state.y_test, st.session_state.y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
        else:
            st.metric("Inertia (Clustering Quality)", f"{st.session_state.metric_val:.4f}")
            st.write("Clustering selesai dijalankan menggunakan algoritma K-Means.")

with tab3:
    st.subheader("Database MongoDB Atlas & Retraining")
    # Mengembalikan ke URI cluster awal Anda
    mongo_uri = st.text_input("MongoDB Atlas URI", type="password", value="mongodb+srv://672022138student:benediktus2003@cluster0.7u0hm.mongodb.net/?retryWrites=true&w=majority&tls=true&tlsInsecure=true")
    db_name = st.text_input("Database Name", value="ml_app_db")
    collection_name = st.text_input("Collection Name", value="model_logs")
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        if st.button("💾 Simpan Log ke MongoDB"):
            if not st.session_state.get('trained', False):
                st.warning("Latih model terlebih dahulu.")
            else:
                try:
                    client = MongoClient(mongo_uri, tlsCAFile=certifi.where(), tlsAllowInvalidCertificates=True)
                    db = client[db_name]
                    collection = db[collection_name]
                    log_data = {
                        "task_type": st.session_state.task_type,
                        "dataset_name": st.session_state.dataset_name,
                        "model_name": st.session_state.model_name,
                        "metric_name": st.session_state.metric_name,
                        "metric_value": st.session_state.metric_val
                    }
                    collection.insert_one(log_data)
                    st.success("Berhasil menyimpan log model ke MongoDB!")
                    client.close()
                except Exception as e:
                    st.error(f"Gagal koneksi: {e}")

    with col_d2:
        if st.button("🔄 Ambil & Retrain dari MongoDB"):
            try:
                client = MongoClient(mongo_uri, tlsCAFile=certifi.where(), tlsAllowInvalidCertificates=True)
                db = client[db_name]
                collection = db[collection_name]
                last_log = collection.find_one(sort=[('_id', -1)])
                if last_log:
                    st.info(f"Retrain terakhir ditemukan: {last_log['model_name']} pada {last_log['dataset_name']}")
                    X_re, y_re = load_data(last_log['dataset_name'])
                    st.success("Model berhasil di-retrain ulang dari konfigurasi database MongoDB!")
                else:
                    st.warning("Belum ada data log di database.")
                client.close()
            except Exception as e:
                st.error(f"Error: {e}")

with tab4:
    st.subheader("💬 Form Komentar Publik")
    st.write("Berikan tanggapan, kritik, atau saran terkait aplikasi demo Machine Learning ini.")
    
    with st.form("comment_form"):
        nama = st.text_input("Nama Anda")
        komentar = st.text_area("Tulis Komentar / Pesan")
        submit_comment = st.form_submit_button("Kirim Komentar")
        
        if submit_comment:
            if nama and komentar:
                try:
                    client = MongoClient(mongo_uri, tlsCAFile=certifi.where(), tlsAllowInvalidCertificates=True)
                    db = client[db_name]
                    comments_col = db["public_comments"]
                    comments_col.insert_one({"nama": nama, "komentar": komentar})
                    st.success("Komentar berhasil dikirim dan disimpan ke MongoDB!")
                    client.close()
                except Exception as e:
                    st.error(f"Gagal menyimpan komentar: {e}")
            else:
                st.warning("Harap isi nama dan komentar terlebih dahulu.")
                
    st.markdown("---")
    st.subheader("📋 Daftar Komentar Publik")
    try:
        client = MongoClient(mongo_uri, tlsCAFile=certifi.where(), tlsAllowInvalidCertificates=True)
        db = client[db_name]
        comments_col = db["public_comments"]
        all_comments = list(comments_col.find({}, {"_id": 0, "nama": 1, "komentar": 1}))
        if all_comments:
            df_comments = pd.DataFrame(all_comments)
            st.dataframe(df_comments, use_container_width=True)
        else:
            st.info("Belum ada komentar dari publik.")
        client.close()
    except Exception:
        st.info("Masukkan URI MongoDB yang valid di Tab Database untuk melihat daftar komentar.")