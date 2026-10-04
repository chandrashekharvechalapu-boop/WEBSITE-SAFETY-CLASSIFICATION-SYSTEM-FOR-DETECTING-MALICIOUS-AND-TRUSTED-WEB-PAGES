"""
Website Safety Classification System
Classifies websites as trusted or malicious using machine learning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

def generate_website_data(n_samples=1500, random_state=42):
    """Generate synthetic website feature data"""
    np.random.seed(random_state)
    
    data = []
    
    for i in range(n_samples):
        # Determine if website is malicious (0=trusted, 1=malicious)
        is_malicious = np.random.choice([0, 1], p=[0.6, 0.4])
        
        # Generate features based on website type
        if is_malicious:
            # Malicious websites typically have these characteristics
            url_length = np.random.randint(50, 200)
            domain_age_days = np.random.randint(1, 180)
            has_ssl_certificate = np.random.choice([0, 1], p=[0.7, 0.3])
            ssl_validity_days = np.random.randint(1, 365) if has_ssl_certificate else 0
            page_load_time = np.random.uniform(3, 15)
            external_links_count = np.random.randint(20, 100)
            internal_links_count = np.random.randint(0, 20)
            special_chars_in_url = np.random.randint(5, 20)
            redirect_count = np.random.randint(2, 10)
            suspicious_keywords = np.random.randint(3, 15)
            phishing_score = np.random.uniform(0.6, 1.0)
            malware_indicators = np.random.randint(2, 8)
        else:
            # Trusted websites typically have these characteristics
            url_length = np.random.randint(10, 60)
            domain_age_days = np.random.randint(365, 7300)
            has_ssl_certificate = np.random.choice([0, 1], p=[0.05, 0.95])
            ssl_validity_days = np.random.randint(100, 365) if has_ssl_certificate else 0
            page_load_time = np.random.uniform(0.5, 3)
            external_links_count = np.random.randint(5, 50)
            internal_links_count = np.random.randint(20, 100)
            special_chars_in_url = np.random.randint(0, 5)
            redirect_count = np.random.randint(0, 2)
            suspicious_keywords = np.random.randint(0, 3)
            phishing_score = np.random.uniform(0.0, 0.4)
            malware_indicators = np.random.randint(0, 2)
        
        # Additional features
        dns_record_age_days = np.random.randint(1, 7300)
        page_rank = np.random.uniform(0, 10)
        alexa_rank = np.random.randint(1, 1000000) if not is_malicious else np.random.randint(500000, 10000000)
        https_percentage = np.random.uniform(0, 100) if not is_malicious else np.random.uniform(0, 40)
        
        data.append({
            'Website_ID': f'WEB{i+1:05d}',
            'URL_Length': url_length,
            'Domain_Age_Days': domain_age_days,
            'Has_SSL_Certificate': has_ssl_certificate,
            'SSL_Validity_Days': ssl_validity_days,
            'Page_Load_Time': page_load_time,
            'External_Links_Count': external_links_count,
            'Internal_Links_Count': internal_links_count,
            'Special_Chars_In_URL': special_chars_in_url,
            'Redirect_Count': redirect_count,
            'Suspicious_Keywords': suspicious_keywords,
            'Phishing_Score': phishing_score,
            'Malware_Indicators': malware_indicators,
            'DNS_Record_Age_Days': dns_record_age_days,
            'Page_Rank': page_rank,
            'Alexa_Rank': alexa_rank,
            'HTTPS_Percentage': https_percentage,
            'Website_Safety': is_malicious
        })
    
    return pd.DataFrame(data)

def prepare_features(df):
    """Prepare features for modeling"""
    df_copy = df.copy()
    
    feature_cols = ['URL_Length', 'Domain_Age_Days', 'Has_SSL_Certificate', 'SSL_Validity_Days',
                   'Page_Load_Time', 'External_Links_Count', 'Internal_Links_Count',
                   'Special_Chars_In_URL', 'Redirect_Count', 'Suspicious_Keywords',
                   'Phishing_Score', 'Malware_Indicators', 'DNS_Record_Age_Days',
                   'Page_Rank', 'Alexa_Rank', 'HTTPS_Percentage']
    
    X = df_copy[feature_cols]
    y = df_copy['Website_Safety']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    
    return X_scaled, y, scaler

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple classification models"""
    models = {}
    results = []
    predictions = {}
    
    # Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_proba_lr = lr.predict_proba(X_test)[:, 1]
    
    models['Logistic Regression'] = lr
    predictions['Logistic Regression'] = y_pred_lr
    results.append({
        'Model': 'Logistic Regression',
        'Accuracy': accuracy_score(y_test, y_pred_lr),
        'Precision': precision_score(y_test, y_pred_lr, zero_division=0),
        'Recall': recall_score(y_test, y_pred_lr, zero_division=0),
        'F1_Score': f1_score(y_test, y_pred_lr, zero_division=0),
        'ROC_AUC': roc_auc_score(y_test, y_proba_lr)
    })
    
    # Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    
    models['Random Forest'] = rf
    predictions['Random Forest'] = y_pred_rf
    results.append({
        'Model': 'Random Forest',
        'Accuracy': accuracy_score(y_test, y_pred_rf),
        'Precision': precision_score(y_test, y_pred_rf, zero_division=0),
        'Recall': recall_score(y_test, y_pred_rf, zero_division=0),
        'F1_Score': f1_score(y_test, y_pred_rf, zero_division=0),
        'ROC_AUC': roc_auc_score(y_test, y_proba_rf)
    })
    
    # Gradient Boosting
    print("Training Gradient Boosting...")
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    y_proba_gb = gb.predict_proba(X_test)[:, 1]
    
    models['Gradient Boosting'] = gb
    predictions['Gradient Boosting'] = y_pred_gb
    results.append({
        'Model': 'Gradient Boosting',
        'Accuracy': accuracy_score(y_test, y_pred_gb),
        'Precision': precision_score(y_test, y_pred_gb, zero_division=0),
        'Recall': recall_score(y_test, y_pred_gb, zero_division=0),
        'F1_Score': f1_score(y_test, y_pred_gb, zero_division=0),
        'ROC_AUC': roc_auc_score(y_test, y_proba_gb)
    })
    
    return models, pd.DataFrame(results), predictions, y_test

def plot_website_distribution(df):
    """Plot website safety distribution and characteristics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Website safety distribution
    safety_counts = df['Website_Safety'].value_counts()
    colors_safety = ['#2ecc71', '#e74c3c']
    axes[0, 0].bar(['Trusted', 'Malicious'], [safety_counts[0], safety_counts[1]], 
                   color=colors_safety, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0, 0].set_ylabel('Count', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Website Safety Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate([safety_counts[0], safety_counts[1]]):
        axes[0, 0].text(i, v + 10, str(v), ha='center', fontweight='bold')
    
    # Domain age vs phishing score
    for safety_status in [0, 1]:
        subset = df[df['Website_Safety'] == safety_status]
        label = 'Trusted' if safety_status == 0 else 'Malicious'
        color = '#2ecc71' if safety_status == 0 else '#e74c3c'
        axes[0, 1].scatter(subset['Domain_Age_Days'], subset['Phishing_Score'], 
                          label=label, alpha=0.6, s=50, color=color, edgecolor='black', linewidth=0.5)
    axes[0, 1].set_xlabel('Domain Age (Days)', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Phishing Score', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Domain Age vs Phishing Score', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # SSL certificate distribution
    ssl_by_safety = df.groupby('Website_Safety')['Has_SSL_Certificate'].mean() * 100
    axes[1, 0].bar(['Trusted', 'Malicious'], [ssl_by_safety[0], ssl_by_safety[1]], 
                   color=['#3498db', '#e67e22'], alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1, 0].set_ylabel('SSL Certificate Percentage (%)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('SSL Certificate Adoption by Website Type', fontsize=12, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # URL length distribution
    df[df['Website_Safety'] == 0]['URL_Length'].hist(bins=20, alpha=0.6, label='Trusted', 
                                                      ax=axes[1, 1], color='#2ecc71', edgecolor='black')
    df[df['Website_Safety'] == 1]['URL_Length'].hist(bins=20, alpha=0.6, label='Malicious', 
                                                      ax=axes[1, 1], color='#e74c3c', edgecolor='black')
    axes[1, 1].set_xlabel('URL Length', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('URL Length Distribution by Website Type', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/website_safety_distribution.png', dpi=300, bbox_inches='tight')
    print("Saved: website_safety_distribution.png")
    plt.close()

def plot_model_comparison(results_df):
    """Plot model performance comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
    
    for idx, metric in enumerate(metrics):
        row = idx // 2
        col = idx % 2
        
        axes[row, col].bar(results_df['Model'], results_df[metric], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        axes[row, col].set_ylabel(metric, fontsize=11, fontweight='bold')
        axes[row, col].set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        axes[row, col].set_ylim([0, 1.1])
        axes[row, col].grid(axis='y', alpha=0.3)
        axes[row, col].tick_params(axis='x', rotation=45)
        
        for i, v in enumerate(results_df[metric]):
            axes[row, col].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/website_model_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: website_model_comparison.png")
    plt.close()

def plot_confusion_matrices(models, X_test, y_test):
    """Plot confusion matrices"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (model_name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], 
                   xticklabels=['Trusted', 'Malicious'], 
                   yticklabels=['Trusted', 'Malicious'], cbar=False)
        axes[idx].set_title(f'{model_name}\nConfusion Matrix', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('True Label', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/website_confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("Saved: website_confusion_matrices.png")
    plt.close()

def plot_feature_importance(models):
    """Plot feature importance"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    feature_names = ['URL_Length', 'Domain_Age_Days', 'Has_SSL_Certificate', 'SSL_Validity_Days',
                    'Page_Load_Time', 'External_Links_Count', 'Internal_Links_Count',
                    'Special_Chars_In_URL', 'Redirect_Count', 'Suspicious_Keywords',
                    'Phishing_Score', 'Malware_Indicators', 'DNS_Record_Age_Days',
                    'Page_Rank', 'Alexa_Rank', 'HTTPS_Percentage']
    
    # Random Forest
    rf_model = models['Random Forest']
    importances = rf_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:8]
    axes[0].barh(np.array(feature_names)[sorted_idx], importances[sorted_idx], color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].set_xlabel('Importance', fontsize=11, fontweight='bold')
    axes[0].set_title('Random Forest Feature Importance', fontsize=12, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Gradient Boosting
    gb_model = models['Gradient Boosting']
    importances = gb_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:8]
    axes[1].barh(np.array(feature_names)[sorted_idx], importances[sorted_idx], color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1].set_xlabel('Importance', fontsize=11, fontweight='bold')
    axes[1].set_title('Gradient Boosting Feature Importance', fontsize=12, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/website_feature_importance.png', dpi=300, bbox_inches='tight')
    print("Saved: website_feature_importance.png")
    plt.close()

def plot_security_indicators(df):
    """Plot security indicators analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Malware indicators by website type
    malware_by_safety = df.groupby('Website_Safety')['Malware_Indicators'].mean()
    axes[0, 0].bar(['Trusted', 'Malicious'], [malware_by_safety[0], malware_by_safety[1]], 
                   color=['#2ecc71', '#e74c3c'], alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0, 0].set_ylabel('Average Malware Indicators', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Average Malware Indicators by Website Type', fontsize=12, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Redirect count distribution
    df[df['Website_Safety'] == 0]['Redirect_Count'].hist(bins=10, alpha=0.6, label='Trusted', 
                                                         ax=axes[0, 1], color='#2ecc71', edgecolor='black')
    df[df['Website_Safety'] == 1]['Redirect_Count'].hist(bins=10, alpha=0.6, label='Malicious', 
                                                         ax=axes[0, 1], color='#e74c3c', edgecolor='black')
    axes[0, 1].set_xlabel('Redirect Count', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Redirect Count Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Page load time by website type
    axes[1, 0].boxplot([df[df['Website_Safety'] == 0]['Page_Load_Time'], 
                        df[df['Website_Safety'] == 1]['Page_Load_Time']], 
                       labels=['Trusted', 'Malicious'], patch_artist=True)
    axes[1, 0].set_ylabel('Page Load Time (seconds)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Page Load Time Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Suspicious keywords by website type
    keywords_by_safety = df.groupby('Website_Safety')['Suspicious_Keywords'].mean()
    axes[1, 1].bar(['Trusted', 'Malicious'], [keywords_by_safety[0], keywords_by_safety[1]], 
                   color=['#3498db', '#f39c12'], alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1, 1].set_ylabel('Average Suspicious Keywords', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Average Suspicious Keywords by Website Type', fontsize=12, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/website_security_indicators.png', dpi=300, bbox_inches='tight')
    print("Saved: website_security_indicators.png")
    plt.close()

def main():
    print("="*80)
    print("WEBSITE SAFETY CLASSIFICATION SYSTEM")
    print("="*80)
    
    # Generate dataset
    print("\n[1] Generating website feature data...")
    df = generate_website_data(n_samples=1500)
    df.to_csv('/home/ubuntu/website_safety_data.csv', index=False)
    print(f"Dataset generated: {len(df)} websites")
    print(f"Malicious rate: {df['Website_Safety'].mean()*100:.2f}%")
    
    # Prepare features
    print("\n[2] Preparing features...")
    X, y, scaler = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Train models
    print("\n[3] Training classification models...")
    models, results_df, predictions, y_test = train_models(X_train, X_test, y_train, y_test)
    results_df.to_csv('/home/ubuntu/website_model_results.csv', index=False)
    print("\nModel Results:")
    print(results_df.to_string(index=False))
    
    # Generate visualizations
    print("\n[4] Generating visualizations...")
    plot_website_distribution(df)
    plot_model_comparison(results_df)
    plot_confusion_matrices(models, X_test, y_test)
    plot_feature_importance(models)
    plot_security_indicators(df)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE - All visualizations saved")
    print("="*80)

if __name__ == "__main__":
    main()
