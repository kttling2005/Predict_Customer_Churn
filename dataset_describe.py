import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('Churn_Modelling.csv')

for col in df.columns: #duyet qua tung cot
  fig, (ax_plot, ax_text) = plt.subplots(
#Tạo khung hiển thị (plt.subplots): Thiết lập một khung ảnh gồm 2 ô con cạnh nhau (ax_plot bên trái để vẽ biểu đồ, ax_text bên phải để in thông số) với tỷ lệ chiều rộng là 2:1.
      1, 2, figsize=(10, 3.5), gridspec_kw={'width_ratios': [2, 1]}
  )

  # Kiểm tra nếu là cột số
  if pd.api.types.is_numeric_dtype(df[col]):
      ax_plot.hist(df[col].dropna(), bins=30, color='#1f77b4', edgecolor='none')

      valid_count = df[col].count()
      missing_count = df[col].isnull().sum()
      mean_val = df[col].mean()
      std_val = df[col].std()
      q = df[col].quantile([0, 0.25, 0.5, 0.75, 1])

      stats_text = (
          f'Valid: {valid_count:,} (100%)\n'
          f'Missing: {missing_count} (0%)\n\n'
          f'Mean: {mean_val:,.2f}\n'
          f'Std. Deviation: {std_val:,.2f}\n\n'
          f'Min: {q[0]:,.2f}\n'
          f'25%: {q[0.25]:,.2f}\n'
          f'50%: {q[0.5]:,.2f}\n'
          f'75%: {q[0.75]:,.2f}\n'
          f'Max: {q[1]:,.2f}'
      )

      # Nếu là cột chữ (Categorical)
  else:
      top_counts = df[col].value_counts().head(5)
      top_counts.plot(kind='bar', ax=ax_plot, color='#ff7f0e')
      ax_plot.tick_params(axis='x', rotation=30)

      valid_count = df[col].count()
      missing_count = df[col].isnull().sum()
      unique_count = df[col].nunique()
      top_val = df[col].mode()[0] if not df[col].mode().empty else 'N/A'
      freq_val = df[col].value_counts().iloc[0] if unique_count > 0 else 0

      stats_text = (
          f'Valid: {valid_count:,} (100%)\n'
          f'Missing: {missing_count} (0%)\n\n'
          f'Unique: {unique_count}\n'
          f'Top: {top_val}\n'
          f'Freq: {freq_val:,}'
      )

  ax_plot.set_title(f'# {col}', fontweight='bold', loc='left', fontsize=12)
  ax_plot.spines['top'].set_visible(False)
  ax_plot.spines['right'].set_visible(False)

  ax_text.axis('off')
  ax_text.text(
      0.05,
      0.5,
      stats_text,
      fontsize=10,
      verticalalignment='center',
      family='monospace',
  )
"""
  # plt.tight_layout()
  # plt.show()"""

# 1. Lọc ra các cột số và tính ma trận tương quan
numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()

# 2. Vẽ biểu đồ nhiệt (Heatmap)
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    linewidths=0.5,
    cbar=True,
)

# Thiết lập tiêu đề và hiển thị
plt.title(
    'Ma trận tương quan giữa các biến số (Correlation Matrix)',
    fontsize=14,
    fontweight='bold',
    pad=15,
)
plt.tight_layout()
plt.show()