#!/usr/bin/env python3
"""
EJEMPLOS DE USO - Control de Aforo y Flujo Vehicular con Imágenes COCO
Análisis visual con indicadores de estado (verde/rojo)
"""

import cv2
import numpy as np
from ultralytics import YOLO
import requests
from PIL import Image
import io
from collections import defaultdict


class AforoAnalyzer:
    """Analizador de control de aforo con indicador visual"""
    
    def __init__(self, capacidad_maxima):
        print("Cargando modelo YOLOv8...")
        self.model = YOLO("yolov8n.pt")
        self.capacidad_maxima = capacidad_maxima
        self.confidence_threshold = 0.60  # Umbral para personas
        
        print(f" Sistema listo - Capacidad máxima: {capacidad_maxima} personas")
    
    def analizar_imagen(self, image):
        """Analiza cantidad de personas en la imagen"""
        # Detección
        results = self.model(image, conf=self.confidence_threshold, verbose=False)
        
        personas_count = 0
        detecciones = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls.cpu().numpy()[0])
                    
                    # Solo personas (class_id = 0)
                    if class_id == 0:
                        confidence = float(box.conf.cpu().numpy()[0])
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Filtrar área mínima
                        area = (x2 - x1) * (y2 - y1)
                        if area >= 300:
                            personas_count += 1
                            detecciones.append({
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'confidence': confidence
                            })
        
        # Determinar estado
        porcentaje = (personas_count / self.capacidad_maxima) * 100
        estado = "VERDE" if personas_count < self.capacidad_maxima else "ROJO"
        
        return personas_count, estado, porcentaje, detecciones
    
    def dibujar_resultado(self, image, personas_count, estado, porcentaje, detecciones):
        """Dibuja resultado con indicador de estado"""
        result = image.copy()
        h, w = result.shape[:2]
        
        # Dibujar personas detectadas
        for det in detecciones:
            bbox = det['bbox']
            confidence = det['confidence']
            
            # Color verde para las personas
            cv2.rectangle(result, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # ID y confianza
            label = f"Persona {confidence:.2f}"
            cv2.putText(result, label, (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Panel de aforo (grande y visible)
        panel_height = 180
        cv2.rectangle(result, (10, 10), (450, panel_height), (0, 0, 0), -1)
        cv2.rectangle(result, (10, 10), (450, panel_height), (255, 255, 255), 2)
        
        # Título
        cv2.putText(result, "CONTROL DE AFORO", (25, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Información
        cv2.putText(result, f"Personas detectadas: {personas_count}", (25, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result, f"Capacidad maxima: {self.capacidad_maxima}", (25, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result, f"Ocupacion: {porcentaje:.1f}%", (25, 135),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # SEMÁFORO GRANDE (Estado visual)
        semaforo_x = w - 150
        semaforo_y = 30
        semaforo_size = 100
        
        # Fondo del semáforo
        cv2.rectangle(result, (semaforo_x - 10, semaforo_y - 10), 
                     (semaforo_x + semaforo_size + 10, semaforo_y + semaforo_size + 50),
                     (0, 0, 0), -1)
        
        if estado == "VERDE":
            # Círculo verde
            cv2.circle(result, (semaforo_x + semaforo_size // 2, semaforo_y + semaforo_size // 2), 
                      40, (0, 255, 0), -1)
            cv2.putText(result, "APTO", (semaforo_x + 20, semaforo_y + semaforo_size + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # Círculo rojo
            cv2.circle(result, (semaforo_x + semaforo_size // 2, semaforo_y + semaforo_size // 2), 
                      40, (0, 0, 255), -1)
            cv2.putText(result, "LLENO", (semaforo_x + 15, semaforo_y + semaforo_size + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return result


class FlujoVehicularAnalyzer:
    """Analizador de flujo vehicular con indicador de congestión"""
    
    def __init__(self, umbral_congestion):
        print("Cargando modelo YOLOv8...")
        self.model = YOLO("yolov8n.pt")
        self.umbral_congestion = umbral_congestion
        self.confidence_threshold = 0.65
        
        # Clases vehiculares
        self.vehicle_classes = {
            2: 'auto',
            3: 'motocicleta',
            5: 'autobus',
            7: 'camion'
        }
        
        print(f"Sistema listo - Umbral de congestión: {umbral_congestion} vehículos")
    
    def analizar_imagen(self, image):
        """Analiza cantidad de vehículos en la imagen"""
        results = self.model(image, conf=self.confidence_threshold, verbose=False)
        
        vehiculos_count = 0
        vehiculos_por_tipo = defaultdict(int)
        detecciones = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls.cpu().numpy()[0])
                    
                    # Solo vehículos
                    if class_id in self.vehicle_classes:
                        confidence = float(box.conf.cpu().numpy()[0])
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Filtrar área mínima
                        area = (x2 - x1) * (y2 - y1)
                        if area >= 300:
                            vehiculos_count += 1
                            vehiculos_por_tipo[self.vehicle_classes[class_id]] += 1
                            detecciones.append({
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'tipo': self.vehicle_classes[class_id],
                                'confidence': confidence,
                                'class_id': class_id
                            })
        
        # Determinar estado
        if vehiculos_count < self.umbral_congestion * 0.5:
            estado = "VERDE"
            nivel = "FLUIDO"
        elif vehiculos_count < self.umbral_congestion:
            estado = "AMARILLO"
            nivel = "MODERADO"
        else:
            estado = "ROJO"
            nivel = "CONGESTION"
        
        return vehiculos_count, estado, nivel, vehiculos_por_tipo, detecciones
    
    def dibujar_resultado(self, image, vehiculos_count, estado, nivel, vehiculos_por_tipo, detecciones):
        """Dibuja resultado con indicador de tráfico"""
        result = image.copy()
        h, w = result.shape[:2]
        
        # Colores por tipo de vehículo
        colors = {
            'auto': (0, 0, 255),
            'motocicleta': (255, 0, 255),
            'autobus': (0, 255, 255),
            'camion': (255, 0, 0)
        }
        
        # Dibujar vehículos detectados
        for det in detecciones:
            bbox = det['bbox']
            tipo = det['tipo']
            confidence = det['confidence']
            color = colors.get(tipo, (255, 255, 255))
            
            cv2.rectangle(result, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            label = f"{tipo} {confidence:.2f}"
            cv2.putText(result, label, (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Panel de flujo vehicular
        panel_height = 220
        cv2.rectangle(result, (10, 10), (450, panel_height), (0, 0, 0), -1)
        cv2.rectangle(result, (10, 10), (450, panel_height), (255, 255, 255), 2)
        
        # Título
        cv2.putText(result, "FLUJO VEHICULAR", (25, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Información
        y_offset = 75
        cv2.putText(result, f"Total vehiculos: {vehiculos_count}", (25, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30
        
        # Desglose por tipo
        for tipo, count in vehiculos_por_tipo.items():
            if count > 0:
                cv2.putText(result, f"  {tipo}: {count}", (25, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors.get(tipo, (255, 255, 255)), 2)
                y_offset += 25
        
        cv2.putText(result, f"Umbral congestion: {self.umbral_congestion}", (25, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # SEMÁFORO DE TRÁFICO
        semaforo_x = w - 150
        semaforo_y = 30
        semaforo_size = 100
        
        # Fondo
        cv2.rectangle(result, (semaforo_x - 10, semaforo_y - 10), 
                     (semaforo_x + semaforo_size + 10, semaforo_y + semaforo_size + 50),
                     (0, 0, 0), -1)
        
        # Color según estado
        if estado == "VERDE":
            color_semaforo = (0, 255, 0)
        elif estado == "AMARILLO":
            color_semaforo = (0, 255, 255)
        else:
            color_semaforo = (0, 0, 255)
        
        cv2.circle(result, (semaforo_x + semaforo_size // 2, semaforo_y + semaforo_size // 2), 
                  40, color_semaforo, -1)
        
        # Texto del nivel
        text_x = semaforo_x + 10 if len(nivel) <= 7 else semaforo_x
        cv2.putText(result, nivel, (text_x, semaforo_y + semaforo_size + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_semaforo, 2)
        
        return result


def get_coco_traffic_images():
    """URLs de imágenes de COCO con tráfico"""
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
        print(f"Error descargando: {e}")
        return None


def ejemplo_control_aforo():
    """Control de aforo en espacios públicos con imágenes COCO"""
    print("\n" + "=" * 70)
    print(" CONTROL DE AFORO")
    print("=" * 70)
    print("\nEste sistema analiza la cantidad de personas en imágenes")
    print("y determina si se excede la capacidad máxima establecida.\n")
    
    # Solicitar capacidad máxima
    while True:
        try:
            capacidad = int(input(" Ingresa la capacidad máxima de personas: "))
            if capacidad > 0:
                break
            print("Debe ser un número positivo")
        except ValueError:
            print("Ingresa un número válido")
    
    # Solicitar cantidad de imágenes
    while True:
        try:
            num_imagenes = int(input(" ¿Cuántas imágenes quieres analizar? (1-15): "))
            if 1 <= num_imagenes <= 15:
                break
            print(" Debe ser un número entre 1 y 15")
        except ValueError:
            print(" Ingresa un número válido")
    
    print(f"\n Capacidad configurada: {capacidad} personas")
    print(f" Se analizarán {num_imagenes} imágenes")
    print("\nAnalizando imágenes...\n")
    
    analyzer = AforoAnalyzer(capacidad)
    urls = get_coco_traffic_images()[:num_imagenes]
    
    resultados = {'verde': 0, 'rojo': 0, 'total_personas': 0}
    
    for i, url in enumerate(urls, 1):
        print(f" Imagen {i}/{num_imagenes}: ", end="", flush=True)
        
        image = download_image(url)
        if image is None:
            print("Error al descargar")
            continue
        
        # Analizar
        personas, estado, porcentaje, detecciones = analyzer.analizar_imagen(image)
        result_img = analyzer.dibujar_resultado(image, personas, estado, porcentaje, detecciones)
        
        # Emoji según estado
        emoji = "🟢" if estado == "VERDE" else "🔴"
        print(f"{emoji} {personas} personas - {estado} ({porcentaje:.1f}%)")
        
        # Guardar estadísticas
        resultados['total_personas'] += personas
        if estado == "VERDE":
            resultados['verde'] += 1
        else:
            resultados['rojo'] += 1
        
        # Mostrar imagen
        cv2.imshow(f"Control de Aforo - Imagen {i}/{num_imagenes}", result_img)
        key = cv2.waitKey(0)
        
        if key == ord('q'):
            print("\n Análisis interrumpido por el usuario")
            break
    
    cv2.destroyAllWindows()
    
    # Resumen final
    print("\n" + "=" * 70)
    print(" RESUMEN FINAL")
    print("=" * 70)
    print(f"Imágenes analizadas: {resultados['verde'] + resultados['rojo']} / {num_imagenes}")
    print(f"Total personas detectadas: {resultados['total_personas']}")
    print(f"Promedio por imagen: {resultados['total_personas'] / max(1, resultados['verde'] + resultados['rojo']):.1f}")
    print(f"\nAforo APTO: {resultados['verde']} imágenes")
    print(f"Aforo EXCEDIDO: {resultados['rojo']} imágenes")
    print(f"Tasa de cumplimiento: {(resultados['verde'] / max(1, resultados['verde'] + resultados['rojo'])) * 100:.1f}%")
    print("=" * 70)


def ejemplo_flujo_vehicular():
    """Análisis de flujo vehicular con imágenes COCO"""
    print("\n" + "=" * 70)
    print("FLUJO VEHICULAR")
    print("=" * 70)
    print("\nEste sistema analiza la cantidad de vehículos en imágenes")
    print("y determina el nivel de tráfico (fluido, moderado, congestión).\n")
    
    # Solicitar umbral de congestión
    while True:
        try:
            umbral = int(input(" Ingresa el umbral de congestión (# vehículos): "))
            if umbral > 0:
                break
            print("Debe ser un número positivo")
        except ValueError:
            print("Ingresa un número válido")
    
    # Solicitar cantidad de imágenes
    while True:
        try:
            num_imagenes = int(input(" ¿Cuántas imágenes quieres analizar? (1-15): "))
            if 1 <= num_imagenes <= 15:
                break
            print(" Debe ser un número entre 1 y 15")
        except ValueError:
            print(" Ingresa un número válido")
    
    print(f"\n Umbral configurado: {umbral} vehículos")
    print(f"  FLUIDO: < {int(umbral * 0.5)} vehículos")
    print(f"  MODERADO: {int(umbral * 0.5)}-{umbral} vehículos")
    print(f"  CONGESTIÓN: > {umbral} vehículos")
    print(f" Se analizarán {num_imagenes} imágenes")
    print("\nAnalizando imágenes...\n")
    
    analyzer = FlujoVehicularAnalyzer(umbral)
    urls = get_coco_traffic_images()[:num_imagenes]
    
    resultados = {
        'verde': 0, 'amarillo': 0, 'rojo': 0,
        'total_vehiculos': 0,
        'auto': 0, 'motocicleta': 0, 'autobus': 0, 'camion': 0
    }
    
    for i, url in enumerate(urls, 1):
        print(f"Imagen {i}/{num_imagenes}: ", end="", flush=True)
        
        image = download_image(url)
        if image is None:
            print("Error al descargar")
            continue
        
        # Analizar
        vehiculos, estado, nivel, vehiculos_por_tipo, detecciones = analyzer.analizar_imagen(image)
        result_img = analyzer.dibujar_resultado(image, vehiculos, estado, nivel, vehiculos_por_tipo, detecciones)
        
        # Emoji según estado
        emoji_map = {"VERDE": "🟢", "AMARILLO": "🟡", "ROJO": "🔴"}
        emoji = emoji_map[estado]
        
        desglose = " | ".join([f"{tipo}: {count}" for tipo, count in vehiculos_por_tipo.items()])
        print(f"{emoji} {vehiculos} vehículos - {nivel} ({desglose})")
        
        # Guardar estadísticas
        resultados['total_vehiculos'] += vehiculos
        resultados[estado.lower()] += 1
        for tipo, count in vehiculos_por_tipo.items():
            resultados[tipo] += count
        
        # Mostrar imagen
        cv2.imshow(f"Flujo Vehicular - Imagen {i}/{num_imagenes}", result_img)
        key = cv2.waitKey(0)
        
        if key == ord('q'):
            print("\n Análisis interrumpido por el usuario")
            break
    
    cv2.destroyAllWindows()
    
    # Resumen final
    total_imgs = resultados['verde'] + resultados['amarillo'] + resultados['rojo']
    print("\n" + "=" * 70)
    print(" RESUMEN FINAL")
    print("=" * 70)
    print(f"Imágenes analizadas: {total_imgs} / {num_imagenes}")
    print(f"Total vehículos detectados: {resultados['total_vehiculos']}")
    print(f"Promedio por imagen: {resultados['total_vehiculos'] / max(1, total_imgs):.1f}")
    print(f"\nDesglose por tipo:")
    print(f"   Autos: {resultados['auto']}")
    print(f"    Motocicletas: {resultados['motocicleta']}")
    print(f"   Autobuses: {resultados['autobus']}")
    print(f"   Camiones: {resultados['camion']}")
    print(f"\nEstados de tráfico:")
    print(f"   FLUIDO: {resultados['verde']} imágenes")
    print(f"   MODERADO: {resultados['amarillo']} imágenes")
    print(f"   CONGESTIÓN: {resultados['rojo']} imágenes")
    print("=" * 70)


def menu_principal():
    """Menú de selección de ejemplos"""
    print("\n" + "=" * 70)
    print(" SISTEMA DE ANÁLISIS DE TRÁFICO")
    print("=" * 70)
    print("\nSelecciona el modo de análisis:\n")
    print("1.  Control de Aforo (detección de personas)")
    print("2.  Flujo Vehicular (detección de vehículos)")
    print("0.  Salir")
    print("=" * 70)
    
    ejemplos = {
        '1': ejemplo_control_aforo,
        '2': ejemplo_flujo_vehicular
    }
    
    while True:
        opcion = input("\n Selecciona una opción (0-2): ").strip()
        
        if opcion == '0':
            print("\n ¡Hasta luego!")
            break
        elif opcion in ejemplos:
            print()
            ejemplos[opcion]()
            print("\n Análisis completado!")
            
            continuar = input("\n¿Probar otro modo? (s/n): ").strip().lower()
            if continuar != 's':
                break
        else:
            print(" Opción inválida. Intenta de nuevo.")


if __name__ == "__main__":
    print("\n SISTEMA DE ANÁLISIS DE TRÁFICO - IMÁGENES COCO")
    print("Utiliza YOLOv8 para detectar personas y vehículos en imágenes")
    print("=" * 70)
    
    try:
        menu_principal()
    except KeyboardInterrupt:
        print("\n\n Programa interrumpido. ¡Hasta luego!")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()

