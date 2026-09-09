import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Customer Churn Monitoring Dashboard", layout="wide")

st.title("📊 Dashboard Theo Dõi & Dự Đoán Khách Hàng Rời Bỏ (Churn)")
st.markdown(
    "Hệ thống trực quan hóa xác suất rời bỏ khách hàng, phân loại rủi ro và hỗ trợ trích xuất danh sách can thiệp.")


# Tạo dữ liệu mẫu mô phỏng kết quả dự đoán từ mô hình XGBoost
@st.cache_data
def load_data():
    np.random.seed(42)
    n_samples = 2000
    data = pd.DataFrame({
        'CustomerID': range(10000, 10000 + n_samples),
        'Age': np.random.randint(18, 70, size=n_samples),
        'Balance': np.random.uniform(0, 150000, size=n_samples),
        'NumOfProducts': np.random.randint(1, 4, size=n_samples),
        'IsActiveMember': np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6]),
        'Churn_Probability': np.random.uniform(0, 1, size=n_samples)
    })

    def get_risk_level(prob):
        if prob >= 0.7:
            return 'Nguy cơ cao (High)'
        elif prob >= 0.4:
            return 'Nguy cơ trung bình (Medium)'
        else:
            return 'Nguy cơ thấp (Low)'

    data['Risk_Level'] = data['Churn_Probability'].apply(get_risk_level)
    return data


df = load_data()

# --- SIDEBAR: BỘ LỌC ---
st.sidebar.header("Bộ lọc thông tin")
selected_risk = st.sidebar.multiselect(
    "Chọn mức độ rủi ro:",
    options=df['Risk_Level'].unique(),
    default=['Nguy cơ cao (High)', 'Nguy cơ trung bình (Medium)']
)

filtered_df = df[df['Risk_Level'].isin(selected_risk)]

# --- HIỂN THỊ CHỈ SỐ TỔNG QUAN (METRICS) ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Tổng khách hàng khảo sát", len(df))
col2.metric("Khách hàng Nguy cơ cao", len(df[df['Risk_Level'] == 'Nguy cơ cao (High)']))
col3.metric("Tỷ lệ Churn trung bình", f"{df['Churn_Probability'].mean() * 100:.2f}%")
col4.metric("Khách hàng đang hiển thị", len(filtered_df))

st.markdown("---")

# --- KHU VỰC VẼ BIỂU ĐỒ CHUYÊN NGHIỆP (PLOTLY) ---
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("📈 Phân phối Xác suất Churn")
    # Biểu đồ Histogram kết hợp KDE mô phỏng bằng Plotly
    fig_hist = px.histogram(
        df, x="Churn_Probability", color="Risk_Level",
        marginal="box",  # Thêm biểu đồ boxplot ở trên cho chuyên nghiệp
        color_discrete_map={
            'Nguy cơ cao (High)': '#ef4444',
            'Nguy cơ trung bình (Medium)': '#f59e0b',
            'Nguy cơ thấp (Low)': '#10b981'
        },
        labels={'Churn_Probability': 'Xác suất Rời bỏ', 'count': 'Số lượng khách hàng'}
    )
    fig_hist.update_layout(bargap=0.1, template="plotly_white")
    st.plotly_chart(fig_hist, use_container_width=True)

with chart_col2:
    st.subheader("📊 Tầm quan trọng Đặc trưng (SHAP/Model)")
    # Biểu đồ thanh ngang thể hiện các yếu tố ảnh hưởng mạnh
    feature_importance = pd.DataFrame({
        'Feature': ['NumOfProducts', 'Age', 'IsActiveMember', 'Balance', 'Geography_Germany'],
        'Importance': [0.35, 0.28, 0.18, 0.12, 0.07]
    }).sort_values('Importance', ascending=True)

    fig_bar = px.bar(
        feature_importance, x='Importance', y='Feature', orientation='h',
        color='Importance', color_continuous_scale='Viridis',
        labels={'Importance': 'Mức độ tác động', 'Feature': 'Đặc trưng'}
    )
    fig_bar.update_layout(template="plotly_white")
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# --- TRÌNH BÀY DỮ LIỆU & XUẤT FILE ---
st.subheader("📋 Danh sách khách hàng cần can thiệp theo phân khúc rủi ro")
st.dataframe(filtered_df.sort_values(by='Churn_Probability', ascending=False), use_container_width=True)

# Nút tải xuống danh sách khẩn cấp
csv = filtered_df[filtered_df['Risk_Level'] == 'Nguy cơ cao (High)'].to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Tải xuống danh sách khách hàng Nguy cơ cao cần can thiệp khẩn cấp (CSV)",
    data=csv,
    file_name='high_risk_customers_to_intervention.csv',
    mime='text/csv',
)