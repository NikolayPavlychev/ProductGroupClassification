Проект ProductGroupPrediction

Сортировка товаров по группам на базе описания товарной карточки.

Участники: Николай Павлычев, Кусимова Екатерина, Татаурова Анастасия

CLI.

1. Препроцессинг
<путь до интерпретатора> preprocessing.py <путь до .csv файла> <режим>

<путь до .csv файла> файл должен иметь атрибуты ['artical', 'brend_code', 'desc', 'guid', 'group_code']

<режим>:
train - переобучение модели на данных из .csv файла
inference - инференс модели на данных из .csv файла
inference_production - инференс предобученной модели на данных из .csv файла

2. Обучение, инференс и оценка модели
<путь до интерпретатора> training_inference.py <путь до .csv файла> <режим>

<путь до .csv файла> файл должен иметь атрибуты ['artical', 'brend_code', 'desc', 'guid', 'group_code']

<режим>:
train - переобучение модели на данных из .csv файла
inference - инференс модели на данных из .csv файла
inference_production - инференс предобученной модели на данных из .csv файла

Артефакт с предсказаниями продуктовых групп:
Атрибуты: ['artical','group_code','group_code_predict']
<имя .csv файла>_predictions'+'_inference.csv',sep=';',index=False)
	
Пример запуска:
	
/home/pavlychev/anaconda3/bin/python preprocessing.py dataset_groups_before_analyse_sub.csv inference_production
/home/pavlychev/anaconda3/bin/python training_inference.py dataset_groups_before_analyse_sub.csv inference_production

3. Cлучайная выборка 

<путь до интерпретатора> random_samples.py 

Формируется dataset_groups_before_analyse_sub.csv, содержащий случаные 100000 записей из dataset_groups_before_analyse.csv
