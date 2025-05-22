import flet as ft
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import io
from PIL import Image
import base64
import matplotlib

matplotlib.use('Agg')  # Устанавливаем non-interactive backend


def main(page: ft.Page):
    # Настройки страницы
    page.title = "Анализ оттока клиентов банка"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 20
    page.scroll = ft.ScrollMode.AUTO
    page.update()

    # Прогресс-бар загрузки
    progress_bar = ft.ProgressBar(width=400, visible=True)
    page.add(
        ft.Text("Загрузка данных и построение графиков...", size=16),
        progress_bar
    )
    page.update()

    # Функция для создания и конвертации графиков
    def create_plot_image(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        return img_byte_arr

    # Загрузка данных
    try:
        data = pd.read_csv('Bank Customer Churn Prediction.csv')

        # Предобработка данных
        label_encoder = LabelEncoder()
        categorical_cols = ['Geography', 'Gender']
        for col in categorical_cols:
            data[col] = label_encoder.fit_transform(data[col])

        data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, errors='ignore')

        X = data.drop('Exited', axis=1)
        y = data['Exited']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        # Обучение модели
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)

        # Очищаем страницу перед добавлением результатов
        page.clean()

        # Заголовок
        page.add(
            ft.Text("Анализ оттока клиентов банка", size=24, weight=ft.FontWeight.BOLD),
            ft.Divider()
        )

        # 1. Распределение оттока клиентов
        fig1 = plt.figure(figsize=(10, 5))
        sns.countplot(x='Exited', data=data)
        plt.title('Распределение оттока клиентов (0 - остаются, 1 - уходят)')
        img1 = create_plot_image(fig1)
        plt.close(fig1)

        page.add(
            ft.Text("1. Распределение оттока клиентов", size=18, weight=ft.FontWeight.BOLD),
            ft.Image(src_base64=base64.b64encode(img1).decode("utf-8"),
                     width=800, height=500),
            ft.Text("Вывод: Виден дисбаланс классов - ушедших клиентов значительно меньше.")
        )
        page.update()

        # 2. Корреляционная матрица
        fig2 = plt.figure(figsize=(12, 8))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Корреляционная матрица')
        img2 = create_plot_image(fig2)
        plt.close(fig2)

        page.add(
            ft.Text("2. Корреляционная матрица", size=18, weight=ft.FontWeight.BOLD),
            ft.Image(src_base64=base64.b64encode(img2).decode("utf-8"),
                     width=800, height=500),
            ft.Text("Вывод: Наибольшая корреляция с оттоком у Age (0.29), IsActiveMember (-0.15).")
        )
        page.update()

        # 3. Распределение возраста
        fig3 = plt.figure(figsize=(10, 5))
        sns.histplot(data['Age'], bins=30, kde=True)
        plt.title('Распределение возраста клиентов')
        img3 = create_plot_image(fig3)
        plt.close(fig3)

        page.add(
            ft.Text("3. Распределение возраста клиентов", size=18, weight=ft.FontWeight.BOLD),
            ft.Image(src_base64=base64.b64encode(img3).decode("utf-8"),
                     width=800, height=500),
            ft.Text("Вывод: Основная масса клиентов 30-40 лет, есть пик около 35 лет.")
        )
        page.update()

        # 4. Возраст vs Отток
        fig4 = plt.figure(figsize=(10, 5))
        sns.boxplot(x='Exited', y='Age', data=data)
        plt.title('Распределение возраста по оттоку клиентов')
        img4 = create_plot_image(fig4)
        plt.close(fig4)

        page.add(
            ft.Text("4. Влияние возраста на отток", size=18, weight=ft.FontWeight.BOLD),
            ft.Image(src_base64=base64.b64encode(img4).decode("utf-8"),
                     width=800, height=500),
            ft.Text("Вывод: Ушедшие клиенты в среднем старше (медиана ~45 лет против ~35).")
        )
        page.update()

        # 5. Важность признаков
        feature_importances = pd.DataFrame(rf.feature_importances_,
                                           index=X.columns,
                                           columns=['importance']).sort_values('importance', ascending=False)
        fig5 = plt.figure(figsize=(10, 5))
        sns.barplot(x=feature_importances.importance, y=feature_importances.index)
        plt.title('Важность признаков в модели')
        img5 = create_plot_image(fig5)
        plt.close(fig5)

        page.add(
            ft.Text("5. Важность признаков", size=18, weight=ft.FontWeight.BOLD),
            ft.Image(src_base64=base64.b64encode(img5).decode("utf-8"),
                     width=800, height=500),
            ft.Text("Вывод: Возраст (Age) и баланс (Balance) - ключевые факторы оттока.")
        )
        page.update()

        # 6. Матрица ошибок
        cm = confusion_matrix(y_test, y_pred_rf)
        fig6 = plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Матрица ошибок')
        plt.xlabel('Предсказанные')
        plt.ylabel('Фактические')
        img6 = create_plot_image(fig6)
        plt.close(fig6)

        page.add(
            ft.Text("6. Матрица ошибок модели", size=18, weight=ft.FontWeight.BOLD),
            ft.Image(src_base64=base64.b64encode(img6).decode("utf-8"),
                     width=800, height=500),
            ft.Text(f"Точность модели: {accuracy_score(y_test, y_pred_rf):.2f}")
        )
        page.update()

        # Рекомендации
        page.add(
            ft.Text("Рекомендации по удержанию клиентов:", size=20, weight=ft.FontWeight.BOLD),
            ft.ListView([
                ft.ListTile(title=ft.Text("1. Программы лояльности для клиентов 40+ лет")),
                ft.ListTile(title=ft.Text("2. Персонализированные предложения для неактивных клиентов")),
                ft.ListTile(title=ft.Text("3. Специальные условия для клиентов с высокими балансами")),
                ft.ListTile(title=ft.Text("4. Региональные маркетинговые стратегии"))
            ])
        )

    except Exception as e:
        page.clean()
        page.add(
            ft.Text("Ошибка при выполнении анализа:", size=20, color=ft.colors.RED),
            ft.Text(str(e), size=16),
            ft.Text("Проверьте наличие файла данных и корректность его формата.", size=16)
        )


ft.app(target=main)
