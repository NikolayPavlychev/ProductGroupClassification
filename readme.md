Проект ProductGroupPrediction

Сортировка товаров по группам на базе описания товарной карточки.

--------------------------------------------------------------------
Участники: Николай Павлычев, Кусимова Екатерина, Татаурова Анастасия
--------------------------------------------------------------------

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

--------------------------------------------------------------------

Описание экспериментов.

1. В ходе экспериментов изменялся объем словаря, составленного из токенов записей, путем изменения порога по tf-idf.
Было выявлено, что точность растет по мере роста словаря, но асимпотически стремится к предельному значению accuracy = 0.87 при объеме словаря ~30 тыс. токенов.
2. Были также помимо векторов tf-idf опробованы skip-gram эмбеддинги, но они не дали значимого прироста, а производительность снижали.

--------------------------------------------------------------------

Оборудование.

Работа скриптов настроена по умолчанию на конфигурацию: 32 потоков CPU, 64 Gb RAM. 

--------------------------------------------------------------------

Метрики качества.

Целевая БМ - accuracy.
Качество модели для production : accuracy 0.87

--------------------------------------------------------------------

Низкочастотные и классы с низкими метриками.

Из 839 классов - 315 классов с accuracy<0.8, но их вес относительно мал.
Для каждого такого класса мы искали уникальные токены и формировали регулярные выражения, которые позволят эти классы предсказывать rule based.

index	target	unique_tokens	                                        created_regex
0	17	{'15913830', 'топливомасломер', 'искрового', '...	15913830|топливомасломер|искрового|abft01|ваку...
1	54	{'1438998', 'компрессоров', 'kompressorenoil',...	1438998|компрессоров|kompressorenoil|238990659...
2	57	{'ssf', 'supervis', '310000110', '253331607', ...	ssf|supervis|310000110|253331607|832904|354l|b...
3	70	{'противоотка', '9094205042', '120х80х70', 'не...	противоотка|9094205042|120х80х70|несим|202097|...
4	93	{'183141', 'позолотой', 'цепочке', 'фликер', '...	183141|позолотой|цепочке|фликер|ypn
5	103	{'избыток', 'поя', 'azn06', 'бамажная', 'azn11...	избыток|поя|azn06|бамажная|azn11|инвалид|кубок...
6	116	{'заборник', '03783'}	заборник|03783
7	123	{'onyx', 'vub503660', '8450009377', 'пoдhoжkи'...	onyx|vub503660|8450009377|пoдhoжkи|769523970r|...
8	128	{'111см', '133х111см', 'npa00t31621', 'rtm', '...	111см|133х111см|npa00t31621|rtm|133х111|погруз...
9	130	{'13803001', '1пр', 'c59', 'ванночки', '69х', ...	13803001|1пр|c59|ванночки|69х|15804003|5038см|...




