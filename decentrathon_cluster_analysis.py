import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from typing import Dict, List, Any
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from openai import OpenAI
import json

class DecentrathonClusterAnalyzer:
    """
    Класс для анализа и кластеризации транзакционных данных.
    Использует DBSCAN для кластеризации и ChatGPT для генерации описаний кластеров.
    """
    def __init__(self, eps=1.0, min_samples=3, openai_api_key=None):
        """
        Инициализация анализатора кластеров.
        
        Параметры:
        eps (float): Максимальное расстояние между точками для формирования кластера
        min_samples (int): Минимальное количество точек для формирования кластера
        openai_api_key (str): Ключ API для доступа к ChatGPT
        """
        self.scaler = StandardScaler()  # Стандартизация признаков
        self.model = DBSCAN(eps=eps, min_samples=min_samples)  # Инициализация DBSCAN
        if openai_api_key:
            self.client = OpenAI(api_key=openai_api_key)  # Инициализация клиента OpenAI
        else:
            self.client = None
        
    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Извлечение признаков из транзакционных данных.
        
        Создает следующие признаки для каждой карты:
        - Медианная сумма транзакций
        - Средняя сумма транзакций
        - Стандартное отклонение сумм
        - Количество транзакций
        - Наличие высокочастотных транзакций
        - Наличие международных транзакций
        - Наличие транзакций с высокой суммой
        - Паттерн фиксированных сумм
        - Паттерн рабочих часов
        """
        # Оптимизация использования памяти путем выбора только нужных колонок
        needed_columns = ['card_id', 'transaction_amount_kzt', 'transaction_currency', 'transaction_timestamp']
        df = df[needed_columns].copy()
        
        # Преобразование временной метки в datetime
        df['transaction_timestamp'] = pd.to_datetime(df['transaction_timestamp'])
        
        # Оптимизация типов данных
        df['transaction_amount_kzt'] = df['transaction_amount_kzt'].astype('float32')
        
        # Группировка по card_id для признаков на уровне карты
        card_groups = df.groupby('card_id')
        
        # Создание DataFrame с признаками
        features = pd.DataFrame(index=card_groups.groups.keys())
        
        # Признаки суммы транзакций
        features['median_amount'] = card_groups['transaction_amount_kzt'].median().astype('float32')
        features['mean_amount'] = card_groups['transaction_amount_kzt'].mean().astype('float32')
        features['amount_std'] = card_groups['transaction_amount_kzt'].std().astype('float32')
        
        # Признаки частоты транзакций
        features['transaction_count'] = card_groups.size().astype('int32')
        features['high_frequency_tx'] = (features['transaction_count'] > 50).astype('int8')
        
        # Признаки международных транзакций
        features['international_tx'] = (df.groupby('card_id')['transaction_currency'].apply(
            lambda x: (x != 'KZT').any())).astype('int8')
        
        # Признаки транзакций с высокой суммой
        features['high_value_tx'] = (df.groupby('card_id')['transaction_amount_kzt'].max() > 50000).astype('int8')
        
        # Паттерн фиксированных сумм
        features['fixed_amount_pattern'] = (features['amount_std'] < features['amount_std'].median()).astype('int8')
        
        # Временные признаки
        df['hour'] = df['transaction_timestamp'].dt.hour.astype('int8')
        features['work_hours_tx'] = (df.groupby('card_id')['hour'].apply(
            lambda x: ((x >= 9) & (x <= 18)).mean() > 0.5)).astype('int8')
        
        return features

    def _get_cluster_name(self, characteristics: Dict[str, Any]) -> str:
        """
        Generate a descriptive name for a cluster based on its characteristics.
        """
        # Extract key metrics
        avg_amount = characteristics['avg_transaction_amount']
        median_amount = characteristics['median_transaction_amount']
        card_count = characteristics['card_count']
        intl_ratio = characteristics['international_tx_ratio']
        
        # Get top categories
        top_categories = list(characteristics['top_mcc_categories'].keys())
        top_cities = list(characteristics['top_cities'].keys())
        
        # Determine cluster type based on characteristics
        if card_count > 1000 and avg_amount > 50000:
            return "Крупная корпорация"
        elif card_count > 500 and avg_amount > 20000:
            return "Средний бизнес"
        elif card_count > 100 and avg_amount > 10000:
            return "Малый бизнес"
        elif intl_ratio > 0.3:
            return "Международный бизнес"
        elif avg_amount < 5000 and 'Grocery' in str(top_categories):
            return "Экономный потребитель"
        elif avg_amount < 10000 and 'Entertainment' in str(top_categories):
            return "Студент"
        elif avg_amount < 8000 and 'Education' in str(top_categories):
            return "Школьник"
        elif avg_amount > 30000 and 'Travel' in str(top_categories):
            return "Путешественник"
        elif avg_amount > 20000 and 'Luxury' in str(top_categories):
            return "Премиум клиент"
        else:
            return "Стандартный клиент"

    def _get_cluster_description_from_gpt(self, characteristics: Dict[str, Any]) -> str:
        """
        Get cluster description from ChatGPT API.
        
        Args:
            characteristics: Dictionary containing cluster characteristics
            
        Returns:
            String containing cluster description
        """
        if not self.client:
            return {
                'name': 'Unknown Segment',
                'description': 'OpenAI API key not provided',
                'insights': [],
                'opportunities': []
            }

        # Prepare prompt for ChatGPT
        prompt = f"""Analyze these transaction cluster characteristics and provide a detailed description of the customer segment in Russian language:

Cluster Characteristics:
- Average Transaction Amount: {characteristics['avg_transaction_amount']:.2f} KZT
- Median Transaction Amount: {characteristics['median_transaction_amount']:.2f} KZT
- Number of Cards: {characteristics['card_count']}
- Total Transactions: {characteristics['total_transactions']}
- International Transaction Ratio: {characteristics['international_tx_ratio']:.2f}
- Transaction Amount Statistics:
  * Standard Deviation: {characteristics['std_amount']:.2f}
  * Minimum: {characteristics['min_amount']:.2f}
  * Maximum: {characteristics['max_amount']:.2f}
- Diversity Metrics:
  * Unique Categories: {characteristics['unique_categories']}
  * Unique Cities: {characteristics['unique_cities']}
  * Unique Transaction Types: {characteristics['unique_transaction_types']}
- Time Patterns:
  * Average Hour: {characteristics['avg_hour']:.2f}
  * Hour Standard Deviation: {characteristics['std_hour']:.2f}
  * Weekend Transaction Ratio: {characteristics['weekend_ratio']:.2f}

Top MCC Categories:
{json.dumps(characteristics['top_mcc_categories'], indent=2)}

Top Cities:
{json.dumps(characteristics['top_cities'], indent=2)}

Transaction Types:
{json.dumps(characteristics['transaction_types'], indent=2)}

Payment Methods:
{json.dumps(characteristics['payment_methods'], indent=2)}

Please provide a response in the following JSON format:
{{
    "name": "Название сегмента",
    "description": "Подробное описание сегмента",
    "insights": [
        "Инсайт 1",
        "Инсайт 2",
        "Инсайт 3"
    ],
    "opportunities": [
        "Возможность 1",
        "Возможность 2",
        "Возможность 3"
    ]
}}

Make sure the response is a valid JSON object with exactly these keys."""

        try:
            # Call ChatGPT API using the new format
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial analyst specializing in customer segmentation. Always respond in Russian language with valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            # Get the response content
            content = response.choices[0].message.content.strip()
            
            # Try to parse the JSON response
            try:
                description = json.loads(content)
                # Validate required keys
                required_keys = ['name', 'description', 'insights', 'opportunities']
                if not all(key in description for key in required_keys):
                    raise ValueError("Missing required keys in response")
                return description
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response: {str(e)}")
                print(f"Raw response: {content}")
                return {
                    'name': 'Ошибка формата',
                    'description': 'Не удалось обработать ответ от ChatGPT',
                    'insights': ['Проверьте формат ответа'],
                    'opportunities': ['Попробуйте еще раз']
                }
            
        except Exception as e:
            print(f"Error getting description from ChatGPT: {str(e)}")
            return {
                'name': 'Ошибка API',
                'description': 'Не удалось получить описание от ChatGPT',
                'insights': ['Проверьте подключение к API'],
                'opportunities': ['Попробуйте позже']
            }

    def analyze_clusters(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Анализ кластеров с использованием DBSCAN.
        
        Процесс:
        1. Извлечение и масштабирование признаков
        2. Применение DBSCAN для кластеризации
        3. Расчет метрик качества кластеризации
        4. Анализ характеристик каждого кластера
        
        Возвращает словарь с метриками, характеристиками кластеров и их распределением.
        """
        # Извлечение и масштабирование признаков
        features = self._extract_features(df)
        feature_columns = ['median_amount', 'international_tx', 'high_value_tx', 
                         'high_frequency_tx', 'fixed_amount_pattern', 'work_hours_tx']
        
        # Вывод статистики признаков
        print("\nСтатистика признаков перед масштабированием:")
        print(features[feature_columns].describe())
        
        # Масштабирование признаков
        scaled_features = self.scaler.fit_transform(features[feature_columns])
        
        # Применение DBSCAN и получение меток кластеров
        labels = self.model.fit_predict(scaled_features)
        
        # Вывод результатов кластеризации
        unique_labels = np.unique(labels)
        print("\nРезультаты кластеризации:")
        print(f"Количество кластеров: {len(unique_labels) - (1 if -1 in unique_labels else 0)}")
        print(f"Количество шумовых точек: {np.sum(labels == -1)}")
        print("\nТочки по кластерам:")
        for label in unique_labels:
            count = np.sum(labels == label)
            print(f"Кластер {label}: {count} точек")
        
        # Добавление меток кластеров к признакам
        features['cluster'] = labels
        
        # Расчет метрик качества кластеризации
        metrics = {}
        if len(np.unique(labels)) > 1:
            metrics = {
                'silhouette_score': silhouette_score(scaled_features, labels),
                'calinski_harabasz_score': calinski_harabasz_score(scaled_features, labels),
                'davies_bouldin_score': davies_bouldin_score(scaled_features, labels)
            }
        
        # Анализ характеристик кластеров
        characteristics = {}
        for cluster in np.unique(labels):
            if cluster == -1:  # Пропуск шумовых точек
                continue
                
            # Получение данных для текущего кластера
            cluster_data = df[df['card_id'].isin(features[features['cluster'] == cluster].index)]
            
            # Добавление часа, если его нет
            if 'hour' not in cluster_data.columns:
                cluster_data['hour'] = pd.to_datetime(cluster_data['transaction_timestamp']).dt.hour
            
            # Расчет характеристик кластера
            char = {
                'avg_transaction_amount': cluster_data['transaction_amount_kzt'].mean(),
                'median_transaction_amount': cluster_data['transaction_amount_kzt'].median(),
                'top_mcc_categories': cluster_data['mcc_category'].value_counts().head(5).to_dict(),
                'top_cities': cluster_data['merchant_city'].value_counts().head(5).to_dict(),
                'transaction_types': cluster_data['transaction_type'].value_counts().to_dict(),
                'payment_methods': cluster_data['pos_entry_mode'].value_counts().to_dict(),
                'international_tx_ratio': (cluster_data['transaction_currency'] != 'KZT').mean(),
                'card_count': cluster_data['card_id'].nunique(),
                'total_transactions': len(cluster_data),
                'std_amount': cluster_data['transaction_amount_kzt'].std(),
                'min_amount': cluster_data['transaction_amount_kzt'].min(),
                'max_amount': cluster_data['transaction_amount_kzt'].max(),
                'unique_categories': cluster_data['mcc_category'].nunique(),
                'unique_cities': cluster_data['merchant_city'].nunique(),
                'unique_transaction_types': cluster_data['transaction_type'].nunique(),
                'avg_hour': cluster_data['hour'].mean(),
                'std_hour': cluster_data['hour'].std(),
                'weekend_ratio': (pd.to_datetime(cluster_data['transaction_timestamp']).dt.dayofweek >= 5).mean()
            }
            
            # Добавление названия кластера
            char['cluster_name'] = self._get_cluster_name(char)
            characteristics[cluster] = char
        
        return {
            'metrics': metrics,
            'characteristics': characteristics,
            'cluster_distribution': pd.Series(labels).value_counts().to_dict()
        }

    def print_cluster_descriptions(self, results: Dict[str, Any]):
        """
        Print detailed descriptions of each cluster using ChatGPT.
        
        Args:
            results: Dictionary containing cluster analysis results
        """
        characteristics = results['characteristics']
        total_cards = sum(char['card_count'] for char in characteristics.values())
        
        print("\nПодробное описание кластеров:")
        for cluster, char in characteristics.items():
            print(f"\nКластер {cluster}")
            print(f"Размер кластера: {char['card_count']} ({char['card_count']/total_cards*100:.1f}%)")
            
            # Get description from ChatGPT
            description = self._get_cluster_description_from_gpt(char)
            
            print(f"\nНазвание сегмента: {description['name']}")
            print(f"\nОписание:")
            print(description['description'])
            
            print("\nКлючевые инсайты:")
            for insight in description['insights']:
                print(f"- {insight}")
            
            print("\nВозможности для бизнеса:")
            for opportunity in description['opportunities']:
                print(f"- {opportunity}")
            
            print("\nТехнические характеристики:")
            for key, value in char.items():
                if key not in ['cluster_name', 'top_mcc_categories', 'top_cities', 
                             'transaction_types', 'payment_methods']:
                    print(f"- {key}: {value:.2f}")
            
            print("\nТоп категории MCC (с количеством транзакций):")
            for category, count in char['top_mcc_categories'].items():
                print(f"- {category}: {count:,} транзакций")
            
            print("\nТоп города (с количеством транзакций):")
            for city, count in char['top_cities'].items():
                print(f"- {city}: {count:,} транзакций")
            
            print("\nРаспределение типов транзакций:")
            total_tx = sum(char['transaction_types'].values())
            for tx_type, count in char['transaction_types'].items():
                percentage = (count / total_tx) * 100
                print(f"- {tx_type}: {count:,} транзакций ({percentage:.1f}%)")
            
            print("\nМетоды оплаты:")
            total_payments = sum(char['payment_methods'].values())
            for method, count in char['payment_methods'].items():
                percentage = (count / total_payments) * 100
                print(f"- {method}: {count:,} транзакций ({percentage:.1f}%)")
            
            print("\nГеографическое распределение:")
            print(f"- Количество уникальных городов: {char['unique_cities']}")
            print(f"- Основные города: {', '.join(char['top_cities'].keys())}")
            
            print("\nКатегории покупок:")
            print(f"- Количество уникальных категорий: {char['unique_categories']}")
            print(f"- Основные категории: {', '.join(char['top_mcc_categories'].keys())}")
            
            print("\nВременные паттерны:")
            print(f"- Средний час транзакций: {char['avg_hour']:.1f}")
            print(f"- Доля транзакций в выходные: {char['weekend_ratio']*100:.1f}%")
            
            print("\nФинансовые показатели:")
            print(f"- Средняя сумма транзакции: {char['avg_transaction_amount']:,.2f} KZT")
            print(f"- Медианная сумма транзакции: {char['median_transaction_amount']:,.2f} KZT")
            print(f"- Минимальная сумма: {char['min_amount']:,.2f} KZT")
            print(f"- Максимальная сумма: {char['max_amount']:,.2f} KZT")
            print(f"- Стандартное отклонение: {char['std_amount']:,.2f} KZT")
            
            print("-" * 80)

    def visualize_cluster_characteristics(self, results: Dict[str, Any], save_path: str = None):
        """
        Create visualizations for cluster characteristics.
        
        Args:
            results: Dictionary containing cluster analysis results
            save_path: Optional path to save the plots
        """
        characteristics = results['characteristics']
        
        # Create subplots for different metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Average transaction amount
        clusters = list(characteristics.keys())
        avg_amounts = [char['avg_transaction_amount'] for char in characteristics.values()]
        axes[0, 0].bar(clusters, avg_amounts)
        axes[0, 0].set_title('Average Transaction Amount by Cluster')
        axes[0, 0].set_xlabel('Cluster')
        axes[0, 0].set_ylabel('Amount (KZT)')
        
        # Card count
        card_counts = [char['card_count'] for char in characteristics.values()]
        axes[0, 1].bar(clusters, card_counts)
        axes[0, 1].set_title('Number of Cards by Cluster')
        axes[0, 1].set_xlabel('Cluster')
        axes[0, 1].set_ylabel('Number of Cards')
        
        # International transaction ratio
        intl_ratios = [char['international_tx_ratio'] for char in characteristics.values()]
        axes[1, 0].bar(clusters, intl_ratios)
        axes[1, 0].set_title('International Transaction Ratio by Cluster')
        axes[1, 0].set_xlabel('Cluster')
        axes[1, 0].set_ylabel('Ratio')
        
        # Cluster distribution
        cluster_dist = results['cluster_distribution']
        axes[1, 1].bar(cluster_dist.keys(), cluster_dist.values())
        axes[1, 1].set_title('Cluster Distribution')
        axes[1, 1].set_xlabel('Cluster')
        axes[1, 1].set_ylabel('Number of Points')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def create_cluster_pie_chart(self, results: Dict[str, Any], save_path: str = None):
        """
        Create a pie chart showing the distribution of clusters with their ChatGPT-generated names.
        
        Args:
            results: Dictionary containing cluster analysis results
            save_path: Optional path to save the plot
        """
        characteristics = results['characteristics']
        
        # Prepare data for pie chart
        labels = []
        sizes = []
        total_cards = sum(char['card_count'] for char in characteristics.values())
        
        for cluster, char in characteristics.items():
            # Get cluster name from ChatGPT
            description = self._get_cluster_description_from_gpt(char)
            cluster_name = description['name']
            
            # Calculate percentage
            percentage = (char['card_count'] / total_cards) * 100
            
            labels.append(f"{cluster_name}\n({percentage:.1f}%)")
            sizes.append(char['card_count'])
        
        # Create pie chart
        plt.figure(figsize=(12, 8))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('Распределение клиентов по сегментам', pad=20, fontsize=14)
        
        # Add legend
        plt.legend(labels, title="Сегменты", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()

    def save_clustering_results(self, df: pd.DataFrame, results: Dict[str, Any], output_path: str = 'clustered_transactions.parquet'):
        """
        Сохранение результатов кластеризации в parquet файл.
        
        Процесс:
        1. Извлечение признаков и получение меток кластеров
        2. Создание маппинга между card_id и кластерами
        3. Генерация названий кластеров с помощью ChatGPT
        4. Объединение информации о кластерах с исходными данными
        5. Сохранение результатов в parquet файл
        """
        print("Подготовка к сохранению результатов кластеризации...")
        
        # Извлечение признаков и получение меток кластеров
        features = self._extract_features(df)
        feature_columns = ['median_amount', 'international_tx', 'high_value_tx', 
                         'high_frequency_tx', 'fixed_amount_pattern', 'work_hours_tx']
        scaled_features = self.scaler.fit_transform(features[feature_columns])
        labels = self.model.fit_predict(scaled_features)
        
        # Создание DataFrame с маппингом
        cluster_mapping = pd.DataFrame({
            'card_id': features.index,
            'cluster': labels
        })
        
        # Создание маппинга номеров кластеров на их названия от ChatGPT
        cluster_names = {}
        for cluster, char in results['characteristics'].items():
            description = self._get_cluster_description_from_gpt(char)
            cluster_names[cluster] = description['name']
            print(f"Сгенерировано название для кластера {cluster}: {description['name']}")
        
        print("Названия кластеров сгенерированы, подготовка данных...")
        
        # Объединение информации о кластерах с исходными данными
        df = df.merge(cluster_mapping, on='card_id', how='left')
        df['cluster_name'] = df['cluster'].map(cluster_names)
        
        # Вывод примера результатов
        print("\nПример кластеризованных данных:")
        print(df[['card_id', 'cluster', 'cluster_name']].head(10))
        
        # Сохранение в parquet
        print("\nСохранение в parquet...")
        df.to_parquet(output_path, index=False)
        
        print(f"\nРезультаты сохранены в: {output_path}")
        print(f"Добавлены колонки:")
        print("- cluster: номер кластера")
        print("- cluster_name: название кластера от ChatGPT")

def main():
    # Example usage
    analyzer = DecentrathonClusterAnalyzer(eps=1.0, min_samples=3, openai_api_key='OPENAI_API_KEY')
    
    try:
        # Load and process your data
        print("Loading data...")
        df = pd.read_csv('data/DECENTRATHON_3.0.csv')
        print("Data loaded successfully. Shape:", df.shape)
        
        # Analyze clusters
        print("\nAnalyzing clusters...")
        results = analyzer.analyze_clusters(df)
        
        # Print metrics
        print("\nClustering Metrics:")
        for metric_name, value in results['metrics'].items():
            print(f"{metric_name}: {value:.3f}")
        
        # Print detailed cluster descriptions
        print("\nGenerating cluster descriptions...")
        analyzer.print_cluster_descriptions(results)
        
        # Visualize results
        print("\nCreating visualizations...")
        analyzer.visualize_cluster_characteristics(results, save_path='dbscan_cluster_analysis.png')
        analyzer.create_cluster_pie_chart(results, save_path='cluster_distribution_pie.png')
        
        # Save clustering results to parquet
        print("\nSaving clustering results to parquet...")
        analyzer.save_clustering_results(df, results, 'clustered_transactions.parquet')
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 