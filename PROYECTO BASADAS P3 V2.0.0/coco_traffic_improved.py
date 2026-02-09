#!/usr/bin/env python3

import cv2
import numpy as np
from ultralytics import YOLO
import requests
from PIL import Image
import io
import os
from collections import defaultdict
import math
import numpy as np
from scipy.spatial import distance as dist


class MetricsCalculator:
    """Calculadora de métricas de evaluación: Precision, Recall, IoU"""
    
    def __init__(self):
        self.total_tp = 0  # True Positives
        self.total_fp = 0  # False Positives  
        self.total_fn = 0  # False Negatives
        self.total_iou = []  # Lista de IoU values
        
    def calculate_iou(self, box1, box2):
        """Calcula IoU entre dos bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Área de intersección
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0  # No hay intersección
            
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Área de unión
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate_detections(self, predictions, ground_truth, iou_threshold=0.5):
        """Evalúa detecciones comparando con ground truth"""
        if len(ground_truth) == 0:
            self.total_fp += len(predictions)
            return
            
        if len(predictions) == 0:
            self.total_fn += len(ground_truth)
            return
            
        # Matriz de IoU entre predicciones y ground truth
        iou_matrix = np.zeros((len(predictions), len(ground_truth)))
        
        for i, pred in enumerate(predictions):
            for j, gt in enumerate(ground_truth):
                if pred.get('class_id') == gt.get('class_id'):  # Misma clase
                    iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                    iou_matrix[i][j] = iou
                    self.total_iou.append(iou)
        
        # Asignación usando umbral de IoU
        used_gt = set()
        used_pred = set()
        
        # Ordenar por IoU descendente
        matches = []
        for i in range(len(predictions)):
            for j in range(len(ground_truth)):
                if iou_matrix[i][j] >= iou_threshold:
                    matches.append((i, j, iou_matrix[i][j]))
        
        matches.sort(key=lambda x: x[2], reverse=True)
        
        # Asignar matches
        for pred_idx, gt_idx, iou_val in matches:
            if pred_idx not in used_pred and gt_idx not in used_gt:
                self.total_tp += 1
                used_pred.add(pred_idx)
                used_gt.add(gt_idx)
        
        # False positives: predicciones no matched
        self.total_fp += len(predictions) - len(used_pred)
        
        # False negatives: ground truth no matched
        self.total_fn += len(ground_truth) - len(used_gt)
    
    def get_metrics(self):
        """Calcula y retorna las métricas finales"""# Calcula precision, recall, F1-score e IoU promedio
        precision = self.total_tp / (self.total_tp + self.total_fp) if (self.total_tp + self.total_fp) > 0 else 0.0 
        recall = self.total_tp / (self.total_tp + self.total_fn) if (self.total_tp + self.total_fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        avg_iou = np.mean(self.total_iou) if self.total_iou else 0.0
        
        return {
            'precision': precision,
            'recall': recall, 
            'f1_score': f1_score,
            'avg_iou': avg_iou,
            'total_tp': self.total_tp,
            'total_fp': self.total_fp,
            'total_fn': self.total_fn
        }


class ObjectTracker:
    """Clase para rastrear objetos usando centroide y distancia euclidiana"""
    
    def __init__(self, max_distance=50, max_disappeared=10):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
    
    def register(self, centroid, class_id, confidence):
        """Registra un nuevo objeto"""
        self.objects[self.next_id] = {
            'centroid': centroid,
            'class_id': class_id,
            'confidence': confidence,
            'trail': [centroid]
        }
        self.disappeared[self.next_id] = 0
        self.next_id += 1
    
    def deregister(self, object_id):
        """Elimina un objeto del tracker"""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, detections):
        """Actualiza el tracker con nuevas detecciones"""
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return {}
        
        if len(self.objects) == 0:
            for detection in detections:
                centroid, class_id, confidence = detection
                self.register(centroid, class_id, confidence)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[id]['centroid'] for id in object_ids]
            
            distances = np.linalg.norm(
                np.array(object_centroids)[:, np.newaxis] - 
                np.array([det[0] for det in detections]), axis=2
            )
            
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]
            
            used_row_indices = set()
            used_col_indices = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue
                
                if distances[row, col] > self.max_distance:
                    continue
                
                object_id = object_ids[row]
                centroid, class_id, confidence = detections[col]
                
                self.objects[object_id]['centroid'] = centroid
                self.objects[object_id]['class_id'] = class_id
                self.objects[object_id]['confidence'] = confidence
                self.objects[object_id]['trail'].append(centroid)
                
                if len(self.objects[object_id]['trail']) > 15:
                    self.objects[object_id]['trail'].pop(0)
                
                self.disappeared[object_id] = 0
                
                used_row_indices.add(row)
                used_col_indices.add(col)
            
            unused_row_indices = set(range(0, distances.shape[0])).difference(used_row_indices)
            for row in unused_row_indices:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            unused_col_indices = set(range(0, distances.shape[1])).difference(used_col_indices)
            for col in unused_col_indices:
                centroid, class_id, confidence = detections[col]
                self.register(centroid, class_id, confidence)
        
        return self.objects


class COCOTrafficAnalyzer:
    """Analizador mejorado para imágenes de COCO con filtrado de tráfico"""
    
    def __init__(self, enable_tracking=True):
        # Cargar YOLOv8 pre-entrenado en COCO
        print("Cargando YOLOv8 (pre-entrenado en COCO)...")
        self.model = YOLO("yolov8n.pt")
        print("Modelo YOLOv8 cargado")
        
        # Configuración balanceada: buena precisión + buena detección
        self.confidence_threshold = 0.65  # Balance entre precisión y recall
        self.iou_threshold = 0.40  # Threshold para Non-Maximum Suppression
        self.min_detection_area = 300  # Área mínima en píxeles (reducida para capturar objetos pequeños)
        
        # Umbrales específicos por clase para mejor balance
        self.class_confidence = {
            0: 0.60,  # persona - umbral más bajo (más importante detectarlas)
            1: 0.65,  # bicicleta
            2: 0.65,  # auto
            3: 0.65,  # motocicleta
            5: 0.70,  # autobus - umbral más alto (más fáciles de ver)
            7: 0.70   # camion - umbral más alto (más fáciles de ver)
        }
        
        # SOLO clases relevantes para análisis de tráfico  
        self.traffic_classes = {
            0: 'persona',      # person
            1: 'bicicleta',    # bicycle  
            2: 'auto',         # car
            3: 'motocicleta',  # motorcycle
            5: 'autobus',      # bus
            7: 'camion'        # truck
        }
        
        # Colores para cada clase
        self.colors = {
            0: (0, 255, 0),    # persona - verde
            1: (255, 255, 0),  # bicicleta - cyan
            2: (0, 0, 255),    # auto - azul
            3: (255, 0, 255),  # motocicleta - magenta  
            5: (0, 255, 255),  # autobus - amarillo
            7: (255, 0, 0)     # camion - rojo
        }
        
        # Sistema de tracking con parámetros balanceados
        self.enable_tracking = enable_tracking
        if self.enable_tracking:
            self.trackers = {class_id: ObjectTracker(max_distance=80, max_disappeared=4) 
                           for class_id in self.traffic_classes.keys()}
            self.unique_object_counts = defaultdict(set)  # Contar objetos únicos
            self.total_tracked = defaultdict(int)  # Total de objetos rastreados
            
        # NUEVO: Inicializar calculadora de métricas
        self.metrics_calc = MetricsCalculator()#
    
    def reset_trackers(self):
        """Reinicia los trackers para comenzar con IDs nuevos"""
        if self.enable_tracking:
            self.trackers = {class_id: ObjectTracker(max_distance=80, max_disappeared=4) 
                           for class_id in self.traffic_classes.keys()}
    
    def process_image(self, image, frame_number=0):
        """Procesa imagen y filtra SOLO detecciones de tráfico con tracking"""
        
        # Detección con YOLOv8 con parámetros optimizados
        results = self.model(
            image,
            conf=self.confidence_threshold,  # Mayor confianza requerida
            iou=self.iou_threshold,  # Mejor supresión de detecciones duplicadas
            verbose=False  # Silenciar output detallado
        )
        
        # NUEVO: Detección de alta confianza para simular ground truth
        #Ejecuta YOLO con un umbral de confianza muy alto para obtener detecciones 
        # casi seguras, que se usan como ground truth simulado.
        ground_truth_results = self.model(
            image,
            conf=0.85,  # Confianza muy alta para ground truth
            iou=self.iou_threshold,
            verbose=False
        )
        
        # Filtrar solo clases de tráfico con alta confianza
        filtered_detections = []
        detections_by_class = defaultdict(list)
        ground_truth = []  # NUEVO: Para almacenar ground truth
        
        # Procesar detecciones normales
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls.cpu().numpy()[0])
                    confidence = float(box.conf.cpu().numpy()[0])
                    
                    # FILTRO: Solo clases de tráfico con confianza específica por clase
                    if class_id in self.traffic_classes:
                        # Aplicar umbral específico para cada clase
                        min_confidence = self.class_confidence.get(class_id, self.confidence_threshold)
                        if confidence < min_confidence:
                            continue
                        
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()# Coordenadas de bounding box
                        
                        # Calcular área de detección para filtrar objetos muy pequeños
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        
                        # Filtrar detecciones muy pequeñas (probable ruido)
                        if area < self.min_detection_area:
                            continue
                        
                        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        
                        detection = {
                            'class_id': class_id,
                            'class_name': self.traffic_classes[class_id],
                            'confidence': confidence,
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'center': center
                        }
                        filtered_detections.append(detection)
                        
                        # Para tracking
                        if self.enable_tracking:
                            detections_by_class[class_id].append((center, class_id, confidence))
        
        # NUEVO: Procesar ground truth (alta confianza)
        #Extrae las detecciones válidas de YOLO, se queda solo con objetos de tráfico
        #filtra los pequeños (ruido) y los guarda como ground truth para análisis y métricas.
        for result in ground_truth_results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls.cpu().numpy()[0])
                    confidence = float(box.conf.cpu().numpy()[0])
                    
                    if class_id in self.traffic_classes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        area = (x2 - x1) * (y2 - y1)
                        
                        if area >= self.min_detection_area:
                            ground_truth.append({
                                'class_id': class_id,
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'confidence': confidence
                            })
        
        # NUEVO: Evaluar métricas comparando detecciones vs ground truth
        self.metrics_calc.evaluate_detections(filtered_detections, ground_truth)
        
        # Actualizar trackers si está habilitado (funcionalidad original intacta)
        tracked_objects = {}
        if self.enable_tracking:
            for class_id, detections in detections_by_class.items():
                tracked_objects[class_id] = self.trackers[class_id].update(detections)
                
                # Contar objetos únicos
                for obj_id in tracked_objects[class_id].keys():
                    unique_id = f"{class_id}_{obj_id}"
                    if unique_id not in self.unique_object_counts[class_id]:
                        self.unique_object_counts[class_id].add(unique_id)
                        self.total_tracked[self.traffic_classes[class_id]] += 1
        
        return filtered_detections, tracked_objects if self.enable_tracking else {}
    
    def draw_results(self, image, detections, tracked_objects=None):
        """Dibuja resultados con tracking y trayectorias"""
        result = image.copy()
        
        # Si hay tracking, dibujar objetos rastreados con trayectorias
        if self.enable_tracking and tracked_objects:
            for class_id, objects in tracked_objects.items():
                color = self.colors.get(class_id, (255, 255, 255))
                class_name = self.traffic_classes[class_id]
                
                for obj_id, obj_data in objects.items():
                    centroid = obj_data['centroid']
                    confidence = obj_data['confidence']
                    trail = obj_data['trail']
                    
                    # Dibujar trayectoria
                    if len(trail) > 1:
                        for i in range(1, len(trail)):
                            thickness = int(np.sqrt(32 / float(i + 1)) * 1.5)
                            cv2.line(result, trail[i-1], trail[i], color, thickness)
                    
                    # Dibujar centroide con ID único
                    cv2.circle(result, centroid, 7, color, -1)
                    cv2.circle(result, centroid, 9, (255, 255, 255), 2)
                    
                    # Etiqueta con ID único de tracking
                    label = f"{class_name} ID:{obj_id} ({confidence:.2f})"
                    
                    # Fondo para texto
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(result, (centroid[0]-5, centroid[1]-30), 
                                (centroid[0]+text_w+5, centroid[1]-10), (0, 0, 0), -1)
                    
                    # Texto
                    cv2.putText(result, label, (centroid[0], centroid[1]-15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            # Sin tracking, dibujar detecciones normales
            for i, det in enumerate(detections):
                class_id = det['class_id']
                class_name = det['class_name'] 
                confidence = det['confidence']
                bbox = det['bbox']
                center = det['center']
                color = self.colors[class_id]
                
                # Dibujar bounding box
                cv2.rectangle(result, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                
                # Dibujar punto central
                cv2.circle(result, center, 6, color, -1)
                
                # Etiqueta
                label = f"{class_name}_{i}: {confidence:.2f}"
                
                # Fondo para texto
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(result, (bbox[0], bbox[1]-25), (bbox[0]+text_w+10, bbox[1]), (0, 0, 0), -1)
                
                # Texto
                cv2.putText(result, label, (bbox[0]+5, bbox[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return result

def get_coco_traffic_images():
    """URLs de imágenes REALES de COCO con tráfico"""# Fuente: Flickr (COCO Train2017)
    return [
        "http://farm7.staticflickr.com/6035/6292445906_dcb4133c67_z.jpg",
        "http://farm6.staticflickr.com/5022/5679421199_fea112b087_z.jpg", 
        "http://farm9.staticflickr.com/8263/8703641816_80c3673de3_z.jpg",
        "https://farm4.staticflickr.com/3357/3180229799_249761e7af_z.jpg",
        "http://farm3.staticflickr.com/2586/3885470623_bc84631c22_z.jpg",
        "http://farm4.staticflickr.com/3366/3327801742_f69499ec72_z.jpg",
        "http://farm9.staticflickr.com/8048/8089005305_a6b2feda80_z.jpg",
        "http://farm9.staticflickr.com/8108/8453221995_d27f280075_z.jpg",
        "http://farm8.staticflickr.com/7143/6779976763_b45b68d0af_z.jpg",
        "http://farm1.staticflickr.com/115/278279849_8b3f8d076c_z.jpg",
        "http://farm3.staticflickr.com/2565/3848599066_abcf0243d5_z.jpg",
        "http://farm5.staticflickr.com/4081/4793601580_37e417b655_z.jpg",
        "http://farm8.staticflickr.com/7265/8151250528_e43a66ca50_z.jpg",
        "http://farm4.staticflickr.com/3141/2736665098_0b0870f51f_z.jpg",
        "http://farm1.staticflickr.com/79/238205035_ef5c64ced1_z.jpg"
    ]

def download_image(url):
    """Descarga imagen desde URL"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            image = Image.open(io.BytesIO(response.content))
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def analyze_coco_traffic():
    """Análisis principal de tráfico en dataset COCO"""
    
    print("ANALIZADOR DE TRAFICO - DATASET COCO")
    print("=" * 60)
    print("Descargando imágenes REALES del dataset COCO")
    print("Analizando: personas, autos, buses, motocicletas, camiones, bicicletas")  
    print("Filtrando: semáforos, paraguas y objetos irrelevantes")
    print("BALANCE OPTIMIZADO: Umbrales por clase, área mínima 300px")
    print("=" * 60)
    
    # Crear analizador
    analyzer = COCOTrafficAnalyzer()
    
    # URLs de COCO
    image_urls = get_coco_traffic_images()
    
    # Contadores
    processed_images = 0
    total_detections = 0
    class_counts = {name: 0 for name in analyzer.traffic_classes.values()}
    
    print(f"\nProcesando {len(image_urls)} imágenes de COCO...")
    print("Presiona cualquier tecla para siguiente imagen")
    print("Presiona 'q' para salir\n")
    
    # Configurar ventana
    width, height = 900, 600
    
    for i, url in enumerate(image_urls):
        print(f"Descargando imagen {i+1}/{len(image_urls)}...")
        print(f"URL: {url}")
        
        # Reiniciar trackers para cada imagen nueva (IDs frescos)
        analyzer.reset_trackers()
        
        # Descargar imagen
        image = download_image(url)
        if image is None:
            print("Error descargando imagen")
            continue
        
        # Redimensionar
        image = cv2.resize(image, (width, height))
        processed_images += 1
        
        # Analizar tráfico (FILTRADO) con tracking
        detections, tracked_objects = analyzer.process_image(image, i)
        result_frame = analyzer.draw_results(image, detections, tracked_objects)
        
        # Líneas de referencia para análisis visual
        cv2.line(result_frame, (0, height//2), (width, height//2), (255, 255, 0), 2)  # Horizontal
        cv2.line(result_frame, (width//2, 0), (width//2, height), (255, 255, 0), 2)   # Vertical
        
        # Contar por clase
        current_count = len(detections)
        total_detections += current_count
        
        for det in detections:# Contar detecciones por clase para estadísticas finales
            class_counts[det['class_name']] += 1
        
        # Información en pantalla 
        cv2.rectangle(result_frame, (10, 10), (450, 120), (0, 0, 0), -1)# Fondo para texto
        cv2.putText(result_frame, f"COCO Imagen {i+1}/{len(image_urls)}", 
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)# Título con número de imagen
        cv2.putText(result_frame, f"Detecciones TRAFICO: {current_count}", 
                   (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)# Conteo de detecciones en esta imagen
        cv2.putText(result_frame, f"Dataset: COCO (Solo Trafico)", 
                   (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)# Fuente de datos
        
        # Contadores por clase
        y_offset = 130
        class_summary = {}
        for det in detections:# Resumir conteo por clase para mostrar en pantalla
            class_name = det['class_name']
            if class_name not in class_summary:
                class_summary[class_name] = 0
            class_summary[class_name] += 1
        
        for class_name, count in class_summary.items():# Mostrar conteo por clase en pantalla
            cv2.putText(result_frame, f"{class_name}: {count}", 
                       (width-200, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25
        
        # Mostrar imagen
        cv2.namedWindow('COCO Traffic Analyzer - Solo Trafico', cv2.WINDOW_NORMAL)# Configurar ventana para mostrar resultados
        cv2.resizeWindow('COCO Traffic Analyzer - Solo Trafico', width, height)# Mostrar resultados con detecciones y tracking
        cv2.imshow('COCO Traffic Analyzer - Solo Trafico', result_frame)# Esperar interacción del usuario para avanzar a la siguiente imagen
        
        # Esperar entrada
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
    
    # Resultados finales
    print("\n" + "="*70)
    print("RESULTADOS FINALES - TRAFICO EN DATASET COCO")
    print("="*70)
    print(f"Imágenes procesadas: {processed_images}")
    print(f"Total detecciones TRAFICO: {total_detections}")# Estadísticas finales con enfoque en tráfico
    print(f"Fuente: COCO Train2017 (Filtrado para tráfico)")# Balance optimizado para detectar objetos de tráfico relevantes con umbrales específicos por clase
    print(f"Modelo: YOLOv8 pre-entrenado en COCO")# Clases de tráfico analizadas
    print(f"Clases analizadas: {list(analyzer.traffic_classes.values())}")
    
    if analyzer.enable_tracking:# Mostrar conteo de objetos únicos rastreados por clase
        print("\n--- SEGUIMIENTO DE OBJETOS UNICOS ---")
        total_unique = sum(analyzer.total_tracked.values())
        print(f"Total objetos UNICOS rastreados: {total_unique}")
        for class_name, count in analyzer.total_tracked.items():# Mostrar conteo de objetos únicos por clase
            if count > 0:
                print(f"   {class_name}: {count} objetos únicos")# Estadísticas por clase con porcentaje del total de detecciones
    
    if total_detections > 0:
        print("\nESTADISTICAS POR CLASE (Detecciones totales):")
        for class_name, count in class_counts.items():# Mostrar conteo y porcentaje por clase
            if count > 0:
                percentage = (count / total_detections) * 100
                print(f"   {class_name}: {count} detecciones ({percentage:.1f}%)")# Resumen final con enfoque en tráfico, mostrando conteo total, objetos únicos rastreados y estadísticas por clase para las imágenes de COCO analizadas
    else:
        print("\nNo se detectaron objetos de tráfico")
    
    # NUEVO: Mostrar métricas de evaluación
    metrics = analyzer.metrics_calc.get_metrics()
    print("\n MÉTRICAS DE EVALUACIÓN:")
    print(f"   Precision: {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)")
    print(f"   Recall: {metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)")
    print(f"   F1-Score: {metrics['f1_score']:.3f}")
    print(f"   IoU Promedio: {metrics['avg_iou']:.3f}")
    print(f"   True Positives: {metrics['total_tp']}")
    print(f"   False Positives: {metrics['total_fp']}")
    print(f"   False Negatives: {metrics['total_fn']}")
    
    cv2.destroyAllWindows()
    print(f"\nAnálisis completado - {total_detections} objetos de tráfico en COCO")# Mensaje final con resumen de resultados

def analyze_local_video():
    """Analiza un video local del usuario"""
    
    print("\nANALIZADOR DE VIDEO LOCAL")
    print("=" * 50)
    print("Analiza tus propios videos usando YOLOv8 + COCO")
    print("Filtrado para: personas, autos, buses, motocicletas, camiones, bicicletas")
    print("BALANCE OPTIMIZADO: Umbrales por clase, detección mejorada")
    print("=" * 50)
    
    # Ejemplos de rutas
    print("\nEjemplos de rutas:")
    print("  C:\\Users\\User\\Downloads\\mi_video.mp4")
    print("  D:\\Videos\\trafico_calle.avi") 
    print("  C:\\Users\\User\\Desktop\\video_trafico.mov")
    
    video_path = input("\nIngresa la ruta completa de tu video: ").strip().replace('"', '')
    
    # Verificar archivo
    if not os.path.exists(video_path):
        print("\nERROR: Archivo no encontrado!")
        print("Verifica que la ruta sea correcta y el archivo exista")
        return
    
    # Información del video
    video_name = os.path.basename(video_path)
    file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
    
    print(f"\nVideo encontrado:")
    print(f"  Nombre: {video_name}")
    print(f"  Tamaño: {file_size:.1f} MB")
    
    # Cargar video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("\nERROR: No se puede abrir el video")
        return
    
    # Propiedades del video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"\nPropiedades:")
    print(f"  Resolución: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Duración: {duration:.1f} segundos")
    
    # Crear analizador
    analyzer = COCOTrafficAnalyzer()
    
    print(f"\nIniciando análisis... Presiona 'q' para salir")
    
    frame_count = 0
    total_detections = 0
    class_counts = {name: 0 for name in analyzer.traffic_classes.values()}
    display_width, display_height = 900, 600
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Redimensionar para display
            display_frame = cv2.resize(frame, (display_width, display_height))
            
            # Analizar con tracking
            detections, tracked_objects = analyzer.process_image(display_frame, frame_count)
            result_frame = analyzer.draw_results(display_frame, detections, tracked_objects)
            
            # Actualizar contadores
            current_count = len(detections)
            total_detections += current_count
            
            for det in detections:
                class_counts[det['class_name']] += 1
            
            # Información en pantalla
            cv2.rectangle(result_frame, (10, 10), (350, 120), (0, 0, 0), -1)
            cv2.putText(result_frame, f"Frame: {frame_count}", 
                       (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(result_frame, f"Detecciones: {current_count}", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(result_frame, f"Total: {total_detections}", 
                       (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(result_frame, "Video Local", 
                       (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Líneas de referencia
            cv2.line(result_frame, (0, display_height//2), (display_width, display_height//2), (255, 255, 0), 1)
            cv2.line(result_frame, (display_width//2, 0), (display_width//2, display_height), (255, 255, 0), 1)
            
            # Mostrar
            cv2.namedWindow('Analizador Video Local', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Analizador Video Local', display_width, display_height)
            cv2.imshow('Analizador Video Local', result_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nInterrumpido")
    
    finally:
        # Resultados finales
        print("\n" + "="*60)
        print("RESULTADOS FINALES - VIDEO LOCAL")
        print("="*60)
        print(f"Video: {video_name}")
        print(f"Frames procesados: {frame_count}")
        print(f"Total detecciones: {total_detections}")
        
        if analyzer.enable_tracking:
            print("\n--- OBJETOS UNICOS RASTREADOS ---")
            total_unique = sum(analyzer.total_tracked.values())
            print(f"Total objetos UNICOS: {total_unique}")
            for class_name, count in analyzer.total_tracked.items():
                if count > 0:
                    print(f"  {class_name}: {count} objetos únicos")
        
        if total_detections > 0:
            print("\nESTADISTICAS (Detecciones totales):")
            for class_name, count in class_counts.items():
                if count > 0:
                    percentage = (count / total_detections) * 100
                    print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # NUEVO: Mostrar métricas de evaluación
        metrics = analyzer.metrics_calc.get_metrics()
        print("\n MÉTRICAS DE EVALUACIÓN:")
        print(f"   Precision: {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)")
        print(f"   Recall: {metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)")
        print(f"   F1-Score: {metrics['f1_score']:.3f}")
        print(f"   IoU Promedio: {metrics['avg_iou']:.3f}")
        print(f"   True Positives: {metrics['total_tp']}")
        print(f"   False Positives: {metrics['total_fp']}")
        print(f"   False Negatives: {metrics['total_fn']}")
        
        cap.release()
        cv2.destroyAllWindows()

def main_menu():
    """Menú principal de opciones"""
    
    print("ANALIZADOR DE TRAFICO CON YOLOV8 + COCO")
    print("=" * 50)
    print("Elige una opción:")
    print()
    print("1. Analizar IMAGENES del dataset COCO")
    print("   - Imágenes reales de COCO con tráfico")
    print("   - 15 imágenes predefinidas")
    print()
    print("2. Analizar VIDEO LOCAL")
    print("   - Tu propio video (MP4, AVI, MOV)")
    print("   - Desde cualquier carpeta")
    print()
    print("3. Salir")
    print("=" * 50)
    
    while True:
        try:
            choice = input("\nSelecciona opción (1, 2 o 3): ").strip()
            
            if choice == "1":
                print("\nOpción seleccionada: Imágenes COCO")
                analyze_coco_traffic()
                break
            elif choice == "2":
                print("\nOpción seleccionada: Video Local")
                analyze_local_video()
                break
            elif choice == "3":
                print("\nSaliendo...")
                break
            else:
                print("Opción inválida. Usa 1, 2 o 3.")
        
        except KeyboardInterrupt:
            print("\n\nSaliendo...")
            break

if __name__ == "__main__":
    main_menu()