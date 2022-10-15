# intelligent-placer
# Постановка задачи
Создать программу, которая принимает на вход объекты и многоугольник и определяет возможность уместить объекты в многоугольник без наложений.
## Входные данные
*Вход:* Программа получает изображение в формате *.jpg. На изображении должен быть лист белой бумаги с расположенными на нем объектами и многоугольником. 
*Выход* Программа выведет True в случае, если возможно разместить объекты в многоугольник без пересечений. Программа выведет False, если изображение не будет соответствовать требованиям, или если объекты не помещаются в многоугольник
## Требования
+ На изображении обязательно должен присутствовать многоугольник и минимум один объект
+ Многоугольник должен быть замкнутым и выпуклым
+ Многоугольник должен быть отчетливо нарисован на белой бумаге черным маркером с линией не тоньше 1мм
+ Многоугольник не должен иметь более 7 вершин
+ Объекты должны быть только из заданного тестового [набора](objects)
+ Объекты должны располагаться на белой бумаге вне многоугольника
+ На фоне не должно быть ничего кроме белой бумаги
+ Объекты должны быть хорошо различимы
+ Объекты и многоугольник должны умещаться в листе бумаги
+ Объекты не должны повторяться и пересекаться
+ Объекты будут рассматриваться только по внешнему контуру, объект не может быть помещен в отверстие другого объекта
+ Тень не должна мешать четко распознать границы объектов
+ Угол между направлением камеры и перпендикуляром к поверхности должен быть не более 15°
# План решения задачи
## Нахождение объектов, многоугольника и проверка входных данных
+ С помощью бинаризации и морфологических операций закрытия можно легко отделить объекты и многоугольник от фона при выполненных требованиях к обектам и фону.
+ Многоугольник должен содержать в себе белую область немного меньшего размера, которая по форме повторяет его. Пользуясь этим можно отличить многоугольник от объектов.
+ Найдя особые точки всех найденных объектов, можно сравнить их с тестовым набором данных и, исключая принадлежащие этому набору объекты, найти многоугольник, а также проверить его количество вершин и выпуклость
## Алгоритм нахождения оптимального расположения
+ Вписываем объекты в минимальные по площади многоугольники. В дальниейшем вместо объектов будут рассматриваться их оболочки в виде многоугольников, так как с ними гораздо быстрее работать
+ Проверяем, что сумма площадей объектов меньше чем площадь многоугольников, если это не так, то выводим False
+ Заносим все объекты в список и сортируем его по убыванию площадей.
+ Берем объекты по очереди из списка и перебираем расположение их центрв и поворот так, чтобы ни одна его сторона не пересекала сторон многоугольника и предыдущих расположенных объектов.
+ В случае если объект не вмещается то перемещаем предыдущие объекты
+ Если объекты удалось разместить печатаем True, иначе - False
### Возможные улучшения производительности
+ При размещении объекта из его вершин проводить перпендикуляры к сторонам многоугольника, которые расположены достаточно близко (этот параметр можно определять на основе размеров данных объектов, например - наименьшая из ширин всех объектов). Разрезая многоугольник этими небольшими перпендикулярами и удаляя область, занимаемую объектами, получим множество многоугольников. Благодаря этому можно анализировать оптимальность расположения объекта, составить функцию оценки положения и поставить задачу ее максимизации при расположении очередного объекта. Простейшей функцией оценки может выступать размер наибольшего многоугольника минус их количество, умноженное на некий коэффициент. 
+ Пользуясь методом выше, можно перебирать сразу несколько последовательностей предметов (не только по убыванию площади) и на некотором этапе отбрасывать те, которые менее эффективно используют пространство многоугольника


# Данные
+ [Объекты](objects)
+ [Тесты](tests)
