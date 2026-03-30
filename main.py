"""
Прототип системы детекции уборки столиков по видео.
Гибридный подход: YOLO + детекция движения.
Разбиение области на сетку 3x3 для лучшей детекции.
"""

import argparse
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import os
import logging
from collections import deque
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


def ensure_dirs():
    """Создает необходимые директории для логов, отчетов и результатов."""
    os.makedirs("logs", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("results", exist_ok=True)


def setup_logger(video_path):
    """
    Настраивает логгер для обработки видео.
    
    Args:
        video_path: Путь к видеофайлу
        
    Returns:
        tuple: (logger, log_filename)
    """
    base_name = get_base_name(video_path)
    log_filename = os.path.join("logs", f"detection_{base_name}.log")

    logger = logging.getLogger(base_name)
    logger.setLevel(logging.INFO)

    logger.handlers.clear()

    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger, log_filename


def format_timestamp(video_seconds):
    """
    Преобразует секунды видео в формат HH:MM:SS.
    
    Args:
        video_seconds: Время в секундах
        
    Returns:
        str: Время в формате HH:MM:SS
    """
    hours = int(video_seconds // 3600)
    minutes = int((video_seconds % 3600) // 60)
    seconds = int(video_seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def get_default_output_path(video_path):
    """
    Генерирует путь для выходного видео по умолчанию.
    
    Args:
        video_path: Путь к исходному видео
        
    Returns:
        str: Путь для сохранения результата
    """
    video_name = os.path.basename(video_path)
    name_without_ext = os.path.splitext(video_name)[0]
    extension = os.path.splitext(video_name)[1]
    output_name = f"result_{name_without_ext}{extension}"
    return os.path.join("results", output_name)


def get_base_name(video_path):
    """
    Извлекает имя файла без расширения из пути к видео.
    
    Args:
        video_path: Путь к видеофайлу
        
    Returns:
        str: Имя файла без расширения
    """
    video_name = os.path.basename(video_path)
    return os.path.splitext(video_name)[0]


class Table:
    """Класс, представляющий отдельный столик с его состоянием и историей."""
    
    def __init__(self, table_id, contour, mask, roi, cells):
        """
        Инициализирует объект столика.
        
        Args:
            table_id: Идентификатор столика
            contour: Контур полигона столика
            mask: Маска области столика
            roi: Прямоугольная область (x, y, w, h)
            cells: Список ячеек сетки 3x3
        """
        self.id = table_id
        self.contour = contour
        self.mask = mask
        self.roi = roi
        self.cells = cells
        self.events = []
        self.is_occupied = False
        self.occupation_start_time = None
        self.last_motion_time = None
        self.no_motion_start = None
        self.motion_history = deque(maxlen=30)
        self.cell_motion = [deque(maxlen=30) for _ in range(9)]
        self.prev_frame_gray_cells = [None for _ in range(9)]

    def get_status_text(self):
        """
        Возвращает текстовое описание текущего состояния столика.
        
        Returns:
            str: Текстовый статус столика
        """
        if self.is_occupied:
            return f"Стол {self.id}: 🔴 ЗАНЯТ"
        elif self.occupation_start_time is not None:
            return f"Стол {self.id}: 🟡 ОБНАРУЖЕНА АКТИВНОСТЬ"
        elif self.no_motion_start is not None:
            return f"Стол {self.id}: 🟡 ОЖИДАНИЕ УХОДА"
        else:
            return f"Стол {self.id}: 🟢 СВОБОДЕН"

    def get_color(self):
        """
        Возвращает цвет для отрисовки столика в зависимости от состояния.
        
        Returns:
            tuple: Цвет в формате BGR
        """
        # Если стол занят, но таймер ухода уже запущен — желтый
        if self.is_occupied and self.no_motion_start is not None:
            return (0, 255, 255)  # желтый
        elif self.is_occupied:
            return (0, 0, 255)  # красный
        elif self.occupation_start_time is not None or self.no_motion_start is not None:
            return (0, 255, 255)  # желтый
        else:
            return (0, 255, 0)  # зеленый


class MultiTableDetectionSystem:
    """Основной класс системы детекции занятости столиков."""
    
    def __init__(self, video_path, output_path=None, skip_frames=2, 
                 conf_threshold=0.2, stabilization_time=5.0, 
                 motion_threshold=300, no_motion_timeout=8.0):
        """
        Инициализирует систему детекции.
        
        Args:
            video_path: Путь к видеофайлу
            output_path: Путь для сохранения выходного видео
            skip_frames: Количество пропускаемых кадров между обработками
            conf_threshold: Порог уверенности для YOLO
            stabilization_time: Время непрерывного присутствия для фиксации посадки (сек)
            motion_threshold: Порог движения для одной ячейки (пикселей)
            no_motion_timeout: Время без движения для фиксации ухода (сек)
        """
        self.video_path = video_path
        self.output_path = output_path if output_path else get_default_output_path(video_path)
        self.skip_frames = skip_frames
        self.conf_threshold = conf_threshold
        self.stabilization_time = stabilization_time
        self.motion_threshold = motion_threshold
        self.no_motion_timeout = no_motion_timeout
        self.yolo_interval = 2
        self.logger, self.log_filename = setup_logger(video_path)
        
        self.model = None
        self.tables = []
        self.detection_scale = 1
        self.fps = 30

    def draw_mask_outline(self, frame, cell, color):
        """
        Рисует контур маски ячейки на кадре.
        
        Args:
            frame: Кадр для отрисовки
            cell: Словарь с маской ячейки
            color: Цвет контура
        """
        contours, _ = cv2.findContours(cell['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, color, 1)

    def init_model(self):
        """Инициализирует модель YOLO для детекции людей."""
        if self.model is None:
            self.logger.info("Загрузка модели YOLOv8n...")
            self.model = YOLO('yolov8n.pt')
            self.logger.info("Модель загружена")

    def create_grid_cells_in_polygon(self, polygon, frame_shape, rows=3, cols=3):
        """
        Создаёт сетку rows x cols прямо внутри полигона.
        Каждая ячейка — это маска с формой полигона.
        
        Args:
            polygon: Массив точек полигона
            frame_shape: Форма кадра (height, width)
            rows: Количество строк сетки
            cols: Количество столбцов сетки
            
        Returns:
            list: Список словарей с ячейками, содержащими rect и mask
        """
        x, y, w, h = cv2.boundingRect(polygon)
        cells = []

        # Маска всего стола
        table_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        cv2.fillPoly(table_mask, [polygon], 255)

        # Размер ячеек по boundingRect
        cell_w = w / cols
        cell_h = h / rows

        for i in range(rows):
            for j in range(cols):
                # Создаём маску ячейки
                cell_mask = np.zeros(frame_shape[:2], dtype=np.uint8)

                # Прямоугольник ячейки в пределах boundingRect
                cx1 = int(x + j * cell_w)
                cy1 = int(y + i * cell_h)
                cx2 = int(x + (j+1) * cell_w)
                cy2 = int(y + (i+1) * cell_h)

                # Вырезаем "квадрат" и пересекаем с полигоном стола
                cv2.rectangle(cell_mask, (cx1, cy1), (cx2, cy2), 255, -1)
                cell_mask = cv2.bitwise_and(cell_mask, table_mask)

                cells.append({
                    "rect": (cx1, cy1, cx2-cx1, cy2-cy1),
                    "mask": cell_mask
                })

        return cells

    def select_polygon_roi(self, frame, table_num):
        """
        Позволяет пользователю выбрать область столика в виде многоугольника.
        
        Args:
            frame: Кадр для выбора области
            table_num: Номер столика
            
        Returns:
            dict or None: Словарь с данными столика или None при отмене
        """
        print(f"\n--- Выбор столика #{table_num} ---")
        print("Инструкция:")
        print("1. Обведите область стола (можно большую область)")
        print("2. Система автоматически разобьет её на 9 квадратов")
        print("3. После выбора всех углов нажмите ENTER")
        print("4. Нажмите ESC для отмены")

        clone = frame.copy()
        points = []

        def draw_polygon(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
                if len(points) > 1:
                    cv2.line(clone, points[-2], points[-1], (0, 255, 0), 2)
                cv2.imshow(f"Столик #{table_num} - обведите область стола", clone)

        cv2.namedWindow(f"Столик #{table_num} - обведите область стола")
        cv2.setMouseCallback(f"Столик #{table_num} - обведите область стола", draw_polygon)

        while True:
            cv2.imshow(f"Столик #{table_num} - обведите область стола", clone)
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # ENTER
                break
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                return None

        cv2.destroyAllWindows()

        if len(points) < 3:
            print("❌ Нужно минимум 3 точки!")
            return None

        pts = np.array(points, np.int32)
        x, y, w, h = cv2.boundingRect(pts)

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        # Создаем сетку 3x3 внутри полигона
        cells = self.create_grid_cells_in_polygon(pts, frame.shape, rows=3, cols=3)

        # Визуализация: контур стола + сетка + номера ячеек
        preview = frame.copy()
        cv2.polylines(preview, [pts], True, (0, 255, 0), 3)
        for idx, cell in enumerate(cells):
            self.draw_mask_outline(preview, cell, (0, 255, 255))
            cx, cy, cw, ch = cell['rect']
            cv2.putText(preview, f"{idx+1}", (cx + cw//2 - 10, cy + ch//2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("Сетка внутри полигона", preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(f"✅ Столик #{table_num} добавлен (область {w}x{h}, разбита на 9 секций)")
        return {'contour': pts, 'mask': mask, 'roi': (x, y, w, h), 'cells': cells}

    def select_single_table(self, frame):
        """
        Выбирает один столик на видео.
        
        Args:
            frame: Первый кадр видео
            
        Returns:
            list: Список с данными выбранного столика
        """
        print("\n" + "="*60)
        print("ВЫБОР СТОЛА")
        print("="*60)

        result = self.select_polygon_roi(frame, table_num=1)

        if result is None:
            raise ValueError("❌ Стол не выбран!")

        return [{
            'id': 1,
            'contour': result['contour'],
            'mask': result['mask'],
            'roi': result['roi'],
            'cells': result['cells']
        }]

    def point_in_mask(self, x, y, mask):
        """
        Проверяет, находится ли точка внутри маски.
        
        Args:
            x: Координата X
            y: Координата Y
            mask: Маска области
            
        Returns:
            bool: True если точка внутри маски
        """
        if mask is None:
            return True
        h, w = mask.shape
        return 0 <= x < w and 0 <= y < h and mask[y, x] == 255

    def detect_motion_in_cell(self, frame, cell, prev_gray):
        """
        Детектирует движение в отдельной ячейке.
        
        Args:
            frame: Текущий кадр
            cell: Словарь с данными ячейки
            prev_gray: Предыдущий灰度 кадр ячейки
            
        Returns:
            tuple: (количество движущихся пикселей, обновленный prev_gray)
        """
        x, y, w, h = cell['rect']
        mask = cell['mask']
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return 0, prev_gray

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            return 0, gray

        diff = cv2.absdiff(gray, prev_gray)
        _, thresh = cv2.threshold(diff, 45, 255, cv2.THRESH_BINARY)
        thresh = cv2.medianBlur(thresh, 3)

        # Обрезаем маску под ROI
        roi_mask = mask[y:y+h, x:x+w]

        motion_pixels = cv2.countNonZero(cv2.bitwise_and(thresh, roi_mask))

        # плавное обновление фона
        new_prev = cv2.addWeighted(prev_gray, 0.8, gray, 0.2, 0)
        return motion_pixels, new_prev

    def detect_activity(self, frame, table, frame_count):
        """
        Детектирует активность на столике с помощью движения и YOLO.
        
        Args:
            frame: Текущий кадр
            table: Объект столика
            frame_count: Номер текущего кадра
            
        Returns:
            tuple: (has_activity, active_cells, total_motion, yolo_people)
        """
        active_cells = 0
        total_motion = 0
        
        # Детекция движения по каждой ячейке
        for idx, cell in enumerate(table.cells):
            motion, new_prev = self.detect_motion_in_cell(frame, cell, table.prev_frame_gray_cells[idx])
            table.prev_frame_gray_cells[idx] = new_prev
            table.cell_motion[idx].append(motion)
            
            if motion > self.motion_threshold:
                active_cells += 1
            total_motion += motion

        # базовая логика движения
        has_motion_activity = active_cells >= 2
        yolo_people = 0

        # YOLO только если есть движение и не каждый кадр
        if frame_count % self.yolo_interval == 0:
            x, y, w, h = table.roi

            if w > 0 and h > 0:
                scale = self.detection_scale

                if scale != 1.0:
                    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
                    roi = cv2.resize(frame[y:y+h, x:x+w], (new_w, new_h))
                    sx, sy = w / new_w, h / new_h
                else:
                    roi = frame[y:y+h, x:x+w]
                    sx = sy = 1.0
                
                if roi.size > 0:
                    try:
                        results = self.model(roi, verbose=False, imgsz=640)

                        for r in results:
                            if r.boxes:
                                for box in r.boxes:
                                    cls = int(box.cls[0])
                                    conf = float(box.conf[0])

                                    if cls == 0 and conf > self.conf_threshold:
                                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                                        if scale != 1.0:
                                            x1, y1 = int(x1 * sx), int(y1 * sy)
                                            x2, y2 = int(x2 * sx), int(y2 * sy)

                                        cx, cy = x + (x1 + x2)//2, y + (y1 + y2)//2

                                        if self.point_in_mask(cx, cy, table.mask) or \
                                            self.point_in_mask(x + x1, y + y1, table.mask):
                                            yolo_people += 1
                    except Exception:
                        pass

        # итоговая логика
        has_activity = has_motion_activity or (yolo_people > 0)

        return has_activity, active_cells, total_motion, yolo_people

    def update_table_state(self, table, has_activity, active_cells, total_motion, yolo_people, timestamp, frame_count):
        """
        Обновляет состояние стола и логирует все промежуточные шаги.
        
        Args:
            table: Объект столика
            has_activity: Наличие активности
            active_cells: Количество активных ячеек
            total_motion: Общее количество движения
            yolo_people: Количество обнаруженных людей YOLO
            timestamp: Текущее время видео
            frame_count: Номер кадра
        """
        # Обновляем историю движения
        table.motion_history.append(total_motion)

        video_time = format_timestamp(timestamp)
        status_msg = f"[{video_time}] Стол {table.id}: "
        status_msg += f"Активность={'Да' if has_activity else 'Нет'}, "
        status_msg += f"Ячеек активных={active_cells}/9, "
        status_msg += f"Motion={total_motion}, YOLO={yolo_people}, "
        status_msg += f"Состояние={'ЗАНЯТ' if table.is_occupied else 'СВОБОДЕН'}"
        self.logger.info(status_msg)

        # Логика посадки
        if has_activity and not table.is_occupied:
            if table.occupation_start_time is None:
                table.occupation_start_time = timestamp
                self.logger.info(f"[{timestamp:.2f}s] 🟡 Стол #{table.id}: старт стабилизации посадки")
            
            duration = timestamp - table.occupation_start_time
            if duration >= self.stabilization_time:
                table.is_occupied = True
                real_timestamp_sec = int(timestamp - self.no_motion_timeout)
                table.events.append({
                    "table_id": table.id,
                    "frame": frame_count,
                    "timestamp": format_timestamp(max(0, real_timestamp_sec)),
                    "timestamp_sec": max(0, real_timestamp_sec),
                    "event_type": "table_occupied",
                    "description": f"Клиент сел за стол #{table.id}"
                })
                self.logger.info(f"[{timestamp:.2f}s] ✅ Стол #{table.id}: КЛИЕНТ СЕЛ")
                table.occupation_start_time = None
                table.last_motion_time = timestamp
                table.no_motion_start = None

        # Логика ухода
        elif table.is_occupied:
            if has_activity:
                table.last_motion_time = timestamp
                table.no_motion_start = None
                self.logger.debug(f"[{timestamp:.2f}s] Стол #{table.id}: движение продолжается, стол занят")
            else:
                if table.no_motion_start is None:
                    table.no_motion_start = timestamp
                    self.logger.info(f"[{timestamp:.2f}s] ⚠️ Стол #{table.id}: нет активности, старт таймера ухода ({self.no_motion_timeout}s)")

                idle_duration = timestamp - table.no_motion_start
                self.logger.info(f"[{timestamp:.2f}s] ⏱ Стол #{table.id}: без движения {idle_duration:.1f}/{self.no_motion_timeout}s")
                
                if idle_duration >= self.no_motion_timeout:
                    table.is_occupied = False
                    real_timestamp_sec = int(timestamp - self.no_motion_timeout)
                    table.events.append({
                        "table_id": table.id,
                        "frame": frame_count,
                        "timestamp": format_timestamp(max(0, real_timestamp_sec)),
                        "timestamp_sec": max(0, real_timestamp_sec),
                        "event_type": "table_empty",
                        "description": f"Стол #{table.id} освободился"
                    })
                    self.logger.info(f"[{timestamp:.2f}s] 🚪 Стол #{table.id}: КЛИЕНТ УШЕЛ (нет активности {idle_duration:.1f}s)")
                    table.occupation_start_time = None
                    table.no_motion_start = None
                    table.last_motion_time = None

        # Сброс ложной стабилизации
        elif not has_activity and not table.is_occupied and table.occupation_start_time is not None:
            if timestamp - table.occupation_start_time < self.stabilization_time:
                self.logger.info(f"[{timestamp:.2f}s] 🟡 Стол #{table.id}: ложное срабатывание, сброс таймера посадки")
                table.occupation_start_time = None

    def process_video(self):
        """Основной метод обработки видео."""
        self.init_model()
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {self.video_path}")
        
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("Не удалось прочитать первый кадр")
        
        tables_data = self.select_single_table(first_frame)
        for data in tables_data:
            self.tables.append(Table(
                data['id'], data['contour'], data['mask'], 
                data['roi'], data['cells']
            ))
        
        self.logger.info(f"Выбрано столиков: {len(self.tables)}")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_count = 0
        start = datetime.now()
        
        print(f"\n=== НАЧАЛО ОБРАБОТКИ ===")
        print(f"Столиков: {len(self.tables)}")
        print(f"Сетка: 3x3 для каждого столика")
        print(f"Стабилизация посадки: {self.stabilization_time}с")
        print(f"Таймаут отсутствия активности: {self.no_motion_timeout}с")
        print(f"Порог движения на ячейку: {self.motion_threshold}")
        print(f"FPS видео: {self.fps:.1f}")
        print(f"Выходное видео: {self.output_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_count / self.fps
            
            if frame_count % self.skip_frames == 0:
                for table in self.tables:
                    has_activity, active_cells, total_motion, yolo_people = self.detect_activity(frame, table, frame_count)
                    self.update_table_state(table, has_activity, active_cells, total_motion, yolo_people, timestamp, frame_count)
            
            # Визуализация
            for table in self.tables:
                color = table.get_color()
                pts = table.contour.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, color, 3)
                
                # Рисуем сетку 3x3
                for cell in table.cells:
                    self.draw_mask_outline(frame, cell, color)
                
                M = cv2.moments(table.contour)
                if M["m00"] != 0:
                    cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                    cv2.putText(frame, f"#{table.id}", (cx-15, cy-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(frame, table.get_status_text(), (cx-80, cy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            
            total_events = sum(len(t.events) for t in self.tables)
            cv2.putText(frame, f"Tables: {len(self.tables)} | Events: {total_events}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % 300 == 0:
                progress = 100 * frame_count / total_frames
                print(f"Прогресс: {progress:.1f}% | Кадров: {frame_count}")
        
        cap.release()
        out.release()
        
        total_time = (datetime.now() - start).total_seconds()
        print(f"\n=== ОБРАБОТКА ЗАВЕРШЕНА ===")
        print(f"Время: {total_time:.1f}с | Сохранено: {self.output_path}")
    
    def calculate_statistics(self):
        """Рассчитывает и выводит статистику по событиям."""
        if not self.tables:
            print("\nНет столиков для анализа")
            return
        
        all_events = []
        for table in self.tables:
            for event in table.events:
                event_copy = event.copy()
                event_copy['table_id'] = table.id
                all_events.append(event_copy)
        
        if not all_events:
            print("\nНет событий для анализа")
            return
        
        df = pd.DataFrame(all_events)
        base_name = get_base_name(self.video_path)
        events_filename = os.path.join("reports", f"events_log_{base_name}.csv")
        self.events_filename = events_filename

        df.to_csv(events_filename, index=False, encoding='utf-8')
        print(f"\n📁 Лог событий: {events_filename}")
        
        print("\n" + "="*60)
        print("📊 СТАТИСТИКА")
        print("="*60)

        idle_times = []

        for table in self.tables:
            table_events = sorted(
                [e for e in all_events if e['table_id'] == table.id],
                key=lambda x: x['timestamp_sec']
            )

            for i in range(len(table_events) - 1):
                curr = table_events[i]
                next_e = table_events[i + 1]

                if curr['event_type'] == 'table_empty' and next_e['event_type'] == 'table_occupied':
                    idle_time = next_e['timestamp_sec'] - curr['timestamp_sec']
                    idle_times.append(idle_time)

        if idle_times:
            avg_idle = sum(idle_times) / len(idle_times)
            print(f"\n⏳ Среднее время между клиентами: {avg_idle:.1f} сек ({avg_idle/60:.1f} мин)")
            self.logger.info(f"Среднее время между клиентами: {avg_idle:.1f} сек")
        else:
            print("\n⏳ Недостаточно данных для расчета среднего времени между клиентами")
        
        for table in self.tables:
            table_events = [e for e in all_events if e['table_id'] == table.id]
            occupied = [e for e in table_events if e['event_type'] == 'table_occupied']
            empty = [e for e in table_events if e['event_type'] == 'table_empty']
            
            print(f"\n🪑 Стол #{table.id}: посадок {len(occupied)}, освобождений {len(empty)}")
            for occ, emp in zip(occupied, empty):
                duration = emp['timestamp_sec'] - occ['timestamp_sec']
                print(f"   Посадка: {occ['timestamp']}с → Уход: {emp['timestamp']}с (длительность: {duration:.0f}с)")
        
        print("\n📁 Сохраненные файлы:")
        print(f"   🎥 Видео: {self.output_path}")
        print(f"   📊 CSV:   {self.events_filename}")
        print(f"   🧾 Лог:   {self.log_filename}")


def main():
    """Главная функция запуска системы детекции."""
    ensure_dirs()
    parser = argparse.ArgumentParser(description='Детекция уборки столиков (сетка 3x3)')
    parser.add_argument('--video', type=str, required=True, help='Путь к видео')
    parser.add_argument('--output', type=str, default=None, help='Выходное видео')
    parser.add_argument('--skip-frames', type=int, default=30, help='Пропуск кадров')
    parser.add_argument('--conf-threshold', type=float, default=0.1, help='Порог YOLO (0.15-0.25)')
    parser.add_argument('--stabilization-time', type=float, default=10.0, help='Время для фиксации посадки (сек)')
    parser.add_argument('--no-motion-timeout', type=float, default=20.0, help='Время без движения для фиксации ухода (сек)')
    parser.add_argument('--motion-threshold', type=int, default=5000, help='Порог движения на ячейку (пиксели)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"❌ Ошибка: {args.video} не найден")
        return
    
    print(f"\n🎥 ЗАПУСК")
    print(f"Видео: {args.video}")
    print(f"Результат: {args.output if args.output else get_default_output_path(args.video)}")
    
    system = MultiTableDetectionSystem(
        video_path=args.video,
        output_path=args.output,
        skip_frames=args.skip_frames,
        conf_threshold=args.conf_threshold,
        stabilization_time=args.stabilization_time,
        motion_threshold=args.motion_threshold,
        no_motion_timeout=args.no_motion_timeout
    )
    
    try:
        system.process_video()
        system.calculate_statistics()
        print("\n✅ Готово!")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
