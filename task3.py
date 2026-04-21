# pokedex_squirtle_mvp_working.py
# ------------------------------------------------------------
# MVP tipo "Pokedex" en tiempo real
# - Reusa la lógica de cámara/render del MVP viejo que sí funcionaba
# - Usa tu modelo entrenado de Squirtle por defecto
# - Webcam o video .mp4
# - Ultralytics YOLO + OpenCV
# - Dibujo manual de cajas (sin .plot())
# - FPS en tiempo real
# - conf e iou ajustables desde la GUI
# - Grabación opcional del video anotado
# ------------------------------------------------------------

import os
import time
from collections import Counter
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO


class PokedexSquirtleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pokedex Scanner MVP")
        self.root.geometry("1450x860")
        self.root.minsize(1200, 720)

        # Carpeta base del script
        if "__file__" in globals():
            self.base_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self.base_dir = os.getcwd()

        # ---------- Estado ----------
        self.cap = None
        self.model = None
        self.model_path_loaded = None
        self.running = False
        self.after_id = None
        self.video_writer = None
        self.record_output_path = None
        self.last_annotated_frame = None

        self.frame_counter = 0
        self.prev_time = None
        self.fps = 0.0
        self.session_counts = Counter()
        self.last_detected = "Ninguno"

        # ---------- Variables GUI ----------
        self.source_var = tk.StringVar(value="Webcam")
        self.webcam_index_var = tk.StringVar(value="0")
        self.video_path_var = tk.StringVar(value="")
        self.model_path_var = tk.StringVar(
            value=r"runs\detect\pokemon_detector\squirtle_yolov8n-2\weights\best.pt"
        )

        self.conf_var = tk.DoubleVar(value=0.25)
        self.iou_var = tk.DoubleVar(value=0.70)
        self.imgsz_var = tk.IntVar(value=640)

        self.record_var = tk.BooleanVar(value=True)
        self.mirror_var = tk.BooleanVar(value=True)

        self.status_var = tk.StringVar(value="Listo para iniciar.")
        self.current_source_label_var = tk.StringVar(value="Fuente actual: Ninguna")

        self._build_ui()
        self._apply_style()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------
    def _apply_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("TFrame", background="#7f1d1d")
        style.configure("TLabelframe", background="#7f1d1d", foreground="white")
        style.configure("TLabelframe.Label", background="#7f1d1d", foreground="white")
        style.configure("TLabel", background="#7f1d1d", foreground="white")
        style.configure("Header.TLabel", font=("Segoe UI", 20, "bold"))
        style.configure("Small.TLabel", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10, "bold"))
        style.configure("TCheckbutton", background="#7f1d1d", foreground="white")
        style.configure("TCombobox", fieldbackground="#f5f5f5", background="#f5f5f5")
        style.configure("Horizontal.TScale", background="#7f1d1d")

        self.root.configure(bg="#7f1d1d")

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        header = ttk.Label(main, text="Pokedex Scanner MVP", style="Header.TLabel")
        header.pack(anchor="w", pady=(0, 10))

        content = ttk.Frame(main)
        content.pack(fill="both", expand=True)

        self.left_panel = ttk.Frame(content)
        self.left_panel.pack(side="left", fill="y", padx=(0, 12))

        self.center_panel = ttk.Frame(content)
        self.center_panel.pack(side="left", fill="both", expand=True, padx=(0, 12))

        self.right_panel = ttk.Frame(content, width=340)
        self.right_panel.pack(side="right", fill="y")

        self._build_left_panel()
        self._build_center_panel()
        self._build_right_panel()

    def _build_left_panel(self):
        # Configuración
        config_box = ttk.LabelFrame(self.left_panel, text="Configuración", padding=12)
        config_box.pack(fill="x", pady=(0, 12))

        ttk.Label(config_box, text="Fuente").pack(anchor="w")
        source_combo = ttk.Combobox(
            config_box,
            textvariable=self.source_var,
            values=["Webcam", "Video"],
            state="readonly",
            width=20
        )
        source_combo.pack(fill="x", pady=(4, 10))
        source_combo.bind("<<ComboboxSelected>>", lambda e: self._toggle_source_fields())

        ttk.Label(config_box, text="Índice de webcam").pack(anchor="w")
        self.webcam_entry = ttk.Entry(config_box, textvariable=self.webcam_index_var)
        self.webcam_entry.pack(fill="x", pady=(4, 10))

        ttk.Checkbutton(
            config_box,
            text="Espejar webcam",
            variable=self.mirror_var
        ).pack(anchor="w", pady=(0, 10))

        ttk.Label(config_box, text="Ruta de video .mp4").pack(anchor="w")
        video_row = ttk.Frame(config_box)
        video_row.pack(fill="x", pady=(4, 10))
        self.video_entry = ttk.Entry(video_row, textvariable=self.video_path_var)
        self.video_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(video_row, text="Buscar", command=self.browse_video).pack(side="left", padx=(6, 0))

        ttk.Label(config_box, text="Modelo YOLO .pt").pack(anchor="w")
        model_row = ttk.Frame(config_box)
        model_row.pack(fill="x", pady=(4, 0))
        self.model_entry = ttk.Entry(model_row, textvariable=self.model_path_var)
        self.model_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(model_row, text="Buscar", command=self.browse_model).pack(side="left", padx=(6, 0))

        # Hiperparámetros
        hyper_box = ttk.LabelFrame(self.left_panel, text="Hiperparámetros", padding=12)
        hyper_box.pack(fill="x", pady=(0, 12))

        ttk.Label(hyper_box, text="Confidence threshold (conf)").pack(anchor="w")
        conf_scale = ttk.Scale(hyper_box, from_=0.05, to=0.95, variable=self.conf_var, orient="horizontal")
        conf_scale.pack(fill="x", pady=(6, 0))
        self.conf_value_label = ttk.Label(hyper_box, text=f"{self.conf_var.get():.2f}", style="Small.TLabel")
        self.conf_value_label.pack(anchor="e", pady=(0, 10))
        conf_scale.bind("<Motion>", lambda e: self._refresh_slider_labels())
        conf_scale.bind("<ButtonRelease-1>", lambda e: self._refresh_slider_labels())

        ttk.Label(hyper_box, text="NMS IoU threshold (iou)").pack(anchor="w")
        iou_scale = ttk.Scale(hyper_box, from_=0.05, to=0.95, variable=self.iou_var, orient="horizontal")
        iou_scale.pack(fill="x", pady=(6, 0))
        self.iou_value_label = ttk.Label(hyper_box, text=f"{self.iou_var.get():.2f}", style="Small.TLabel")
        self.iou_value_label.pack(anchor="e", pady=(0, 10))
        iou_scale.bind("<Motion>", lambda e: self._refresh_slider_labels())
        iou_scale.bind("<ButtonRelease-1>", lambda e: self._refresh_slider_labels())

        ttk.Label(hyper_box, text="Tamaño de inferencia (imgsz)").pack(anchor="w")
        imgsz_combo = ttk.Combobox(
            hyper_box,
            textvariable=self.imgsz_var,
            values=[320, 416, 512, 640, 768],
            state="readonly"
        )
        imgsz_combo.pack(fill="x", pady=(4, 10))

        ttk.Checkbutton(
            hyper_box,
            text="Grabar video anotado",
            variable=self.record_var
        ).pack(anchor="w")

        # Controles
        actions_box = ttk.LabelFrame(self.left_panel, text="Controles", padding=12)
        actions_box.pack(fill="x", pady=(0, 12))

        ttk.Button(actions_box, text="Iniciar", command=self.start_capture).pack(fill="x", pady=(0, 8))
        ttk.Button(actions_box, text="Detener", command=self.stop_capture).pack(fill="x", pady=(0, 8))
        ttk.Button(actions_box, text="Guardar snapshot", command=self.save_snapshot).pack(fill="x")

        # Estado
        status_box = ttk.LabelFrame(self.left_panel, text="Estado", padding=12)
        status_box.pack(fill="x")

        ttk.Label(status_box, textvariable=self.current_source_label_var, wraplength=280).pack(anchor="w", pady=(0, 8))
        ttk.Label(status_box, textvariable=self.status_var, wraplength=280).pack(anchor="w")

        self._toggle_source_fields()

    def _build_center_panel(self):
        video_box = ttk.LabelFrame(self.center_panel, text="Escáner", padding=10)
        video_box.pack(fill="both", expand=True)

        self.video_label = tk.Label(
            video_box,
            bg="#000000",
            fg="white",
            text="Aquí aparecerá la cámara o video",
            font=("Segoe UI", 16),
            width=100,
            height=35
        )
        self.video_label.pack(fill="both", expand=True)

    def _build_right_panel(self):
        pokedex_box = ttk.LabelFrame(self.right_panel, text="Panel Pokedex", padding=12)
        pokedex_box.pack(fill="both", expand=True)

        self.summary_label = ttk.Label(
            pokedex_box,
            text="FPS: 0.00\nDetecciones actuales: 0\nÚltimo detectado: Ninguno",
            justify="left",
            font=("Consolas", 11)
        )
        self.summary_label.pack(anchor="w", fill="x", pady=(0, 10))

        ttk.Label(pokedex_box, text="Detecciones del frame actual").pack(anchor="w")
        self.current_text = tk.Text(
            pokedex_box,
            height=12,
            bg="#08152d",
            fg="#f8fafc",
            insertbackground="white",
            relief="flat",
            font=("Consolas", 10)
        )
        self.current_text.pack(fill="x", pady=(4, 10))
        self.current_text.config(state="disabled")

        ttk.Label(pokedex_box, text="Resumen de la sesión").pack(anchor="w")
        self.session_text = tk.Text(
            pokedex_box,
            height=18,
            bg="#08152d",
            fg="#f8fafc",
            insertbackground="white",
            relief="flat",
            font=("Consolas", 10)
        )
        self.session_text.pack(fill="both", expand=True, pady=(4, 0))
        self.session_text.config(state="disabled")

    # --------------------------------------------------------
    # Helpers UI
    # --------------------------------------------------------
    def _toggle_source_fields(self):
        source = self.source_var.get()
        if source == "Webcam":
            self.webcam_entry.config(state="normal")
            self.video_entry.config(state="disabled")
        else:
            self.webcam_entry.config(state="disabled")
            self.video_entry.config(state="normal")

    def _refresh_slider_labels(self):
        self.conf_value_label.config(text=f"{self.conf_var.get():.2f}")
        self.iou_value_label.config(text=f"{self.iou_var.get():.2f}")

    def browse_video(self):
        path = filedialog.askopenfilename(
            title="Seleccionar video",
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv"), ("Todos los archivos", "*.*")]
        )
        if path:
            self.video_path_var.set(path)

    def browse_model(self):
        path = filedialog.askopenfilename(
            title="Seleccionar modelo",
            filetypes=[("Pesos YOLO", "*.pt"), ("Todos los archivos", "*.*")]
        )
        if path:
            self.model_path_var.set(path)

    def _set_text(self, widget, content):
        widget.config(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, content)
        widget.config(state="disabled")

    def _resolve_path(self, raw_path):
        raw_path = raw_path.strip()
        if not raw_path:
            return raw_path
        if os.path.isabs(raw_path):
            return os.path.normpath(raw_path)
        return os.path.normpath(os.path.join(self.base_dir, raw_path))

    def _get_class_name(self, cls_id):
        names = self.model.names
        if isinstance(names, dict):
            return str(names.get(int(cls_id), cls_id))
        if isinstance(names, (list, tuple)) and 0 <= int(cls_id) < len(names):
            return str(names[int(cls_id)])
        return str(cls_id)

    # --------------------------------------------------------
    # Modelo y captura
    # --------------------------------------------------------
    def load_model_if_needed(self):
        model_path = self._resolve_path(self.model_path_var.get())

        if not model_path:
            raise ValueError("Debe indicar una ruta de modelo .pt")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el modelo en:\n{model_path}")

        if self.model is None or self.model_path_loaded != model_path:
            self.status_var.set(f"Cargando modelo: {model_path}")
            self.root.update_idletasks()
            self.model = YOLO(model_path)
            self.model_path_loaded = model_path
            self.status_var.set(f"Modelo cargado correctamente: {os.path.basename(model_path)}")

    def start_capture(self):
        if self.running:
            self.status_var.set("La captura ya está en ejecución.")
            return

        try:
            self.load_model_if_needed()
        except Exception as e:
            messagebox.showerror("Error al cargar el modelo", str(e))
            return

        source_type = self.source_var.get()

        if source_type == "Webcam":
            try:
                cam_index = int(self.webcam_index_var.get().strip())
            except ValueError:
                messagebox.showerror("Error", "El índice de webcam debe ser un número entero.")
                return

            # MISMA LÓGICA DEL CÓDIGO VIEJO QUE SÍ FUNCIONABA
            self.cap = cv2.VideoCapture(cam_index)
            self.current_source_label_var.set(f"Fuente actual: Webcam {cam_index}")
        else:
            video_path = self._resolve_path(self.video_path_var.get())
            if not video_path:
                messagebox.showerror("Error", "Debe seleccionar un video.")
                return
            if not os.path.exists(video_path):
                messagebox.showerror("Error", f"La ruta del video no existe:\n{video_path}")
                return

            self.cap = cv2.VideoCapture(video_path)
            self.current_source_label_var.set(f"Fuente actual: {os.path.basename(video_path)}")

        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Error", "No se pudo abrir la fuente de video.")
            self.cap = None
            return

        self.running = True
        self.prev_time = None
        self.fps = 0.0
        self.frame_counter = 0
        self.session_counts = Counter()
        self.last_detected = "Ninguno"
        self.record_output_path = None
        self.last_annotated_frame = None

        self._release_writer_if_needed()

        self.status_var.set("Captura iniciada.")
        self.update_frame()

    def stop_capture(self):
        self.running = False

        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self._release_writer_if_needed()

        if self.record_output_path:
            self.status_var.set(f"Captura detenida. Video guardado en: {self.record_output_path}")
        else:
            self.status_var.set("Captura detenida.")

    def _release_writer_if_needed(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    # --------------------------------------------------------
    # Procesamiento principal
    # --------------------------------------------------------
    def update_frame(self):
        if not self.running or self.cap is None:
            return

        loop_start = time.time()
        ret, frame = self.cap.read()

        if not ret:
            if self.source_var.get() == "Video":
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()

            if not ret:
                self.status_var.set("No se pudo leer el frame.")
                self.stop_capture()
                return

        # Opcional: espejar webcam
        if self.source_var.get() == "Webcam" and self.mirror_var.get():
            frame = cv2.flip(frame, 1)

        conf = float(self.conf_var.get())
        iou = float(self.iou_var.get())
        imgsz = int(self.imgsz_var.get())

        try:
            results = self.model(
                frame,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                verbose=False
            )
        except Exception as e:
            self.status_var.set(f"Error en inferencia: {e}")
            self.stop_capture()
            return

        annotated = frame.copy()
        current_counts = Counter()

        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()

            for box, cls_id, score in zip(xyxy, cls_ids, confs):
                x1, y1, x2, y2 = map(int, box)

                class_name = self._get_class_name(cls_id)
                current_counts[class_name] += 1
                self.session_counts[class_name] += 1
                self.last_detected = class_name

                color = self._color_from_class(cls_id)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                label = f"{class_name} {score:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)

                text_y1 = max(0, y1 - th - 10)
                text_y2 = y1
                text_x2 = x1 + tw + 10

                cv2.rectangle(annotated, (x1, text_y1), (text_x2, text_y2), color, -1)
                cv2.putText(
                    annotated,
                    label,
                    (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

        elapsed = time.time() - loop_start
        current_fps = 1.0 / elapsed if elapsed > 0 else 0.0

        if self.fps == 0.0:
            self.fps = current_fps
        else:
            self.fps = (0.85 * self.fps) + (0.15 * current_fps)

        self.frame_counter += 1

        total_current = sum(current_counts.values())
        overlay_lines = [
            "POKEDEX SCANNER",
            f"FPS: {self.fps:.2f}",
            f"conf: {conf:.2f}",
            f"iou: {iou:.2f}",
            f"detecciones: {total_current}"
        ]

        overlay_y = 34
        for i, line in enumerate(overlay_lines):
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            scale = 0.95 if i == 0 else 0.75
            thickness = 3 if i == 0 else 2

            cv2.putText(
                annotated,
                line,
                (12, overlay_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thickness,
                cv2.LINE_AA
            )
            overlay_y += 32

        self.last_annotated_frame = annotated.copy()

        self.summary_label.config(
            text=(
                f"FPS: {self.fps:.2f}\n"
                f"Detecciones actuales: {total_current}\n"
                f"Último detectado: {self.last_detected}"
            )
        )

        if current_counts:
            current_text = "\n".join(
                f"- {name}: {count}"
                for name, count in sorted(current_counts.items(), key=lambda x: (-x[1], x[0]))
            )
        else:
            current_text = "Sin detecciones en este frame."

        if self.session_counts:
            session_text = "\n".join(
                f"- {name}: {count}"
                for name, count in sorted(self.session_counts.items(), key=lambda x: (-x[1], x[0]))
            )
        else:
            session_text = "Aún no hay detecciones."

        self._set_text(self.current_text, current_text)
        self._set_text(self.session_text, session_text)

        if self.record_var.get():
            self._write_output_video(annotated)

        self._show_frame_in_gui(annotated)

        self.after_id = self.root.after(10, self.update_frame)

    # --------------------------------------------------------
    # Utilidades visuales
    # --------------------------------------------------------
    def _show_frame_in_gui(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        label_w = self.video_label.winfo_width()
        label_h = self.video_label.winfo_height()

        if label_w < 50 or label_h < 50:
            label_w, label_h = 960, 640

        resized = self._resize_keep_aspect(frame_rgb, label_w, label_h)

        image = Image.fromarray(resized)
        tk_image = ImageTk.PhotoImage(image=image)

        self.video_label.configure(image=tk_image, text="")
        self.video_label.image = tk_image

    def _resize_keep_aspect(self, image_rgb, max_w, max_h):
        h, w = image_rgb.shape[:2]
        if w == 0 or h == 0:
            return image_rgb

        scale = min(max_w / w, max_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        return cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _color_from_class(self, cls_id):
        palette = [
            (255, 99, 71),
            (60, 179, 113),
            (65, 105, 225),
            (255, 215, 0),
            (186, 85, 211),
            (255, 140, 0),
            (0, 206, 209),
            (220, 20, 60),
        ]
        return palette[int(cls_id) % len(palette)]

    # --------------------------------------------------------
    # Salidas
    # --------------------------------------------------------
    def _write_output_video(self, frame_bgr):
        if self.video_writer is None:
            os.makedirs("runs_mvp", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.record_output_path = os.path.join("runs_mvp", f"pokedex_run_{timestamp}.mp4")

            height, width = frame_bgr.shape[:2]

            source_fps = 20.0
            if self.cap is not None:
                detected_fps = self.cap.get(cv2.CAP_PROP_FPS)
                if detected_fps and detected_fps > 1:
                    source_fps = detected_fps

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                self.record_output_path,
                fourcc,
                source_fps,
                (width, height)
            )

        self.video_writer.write(frame_bgr)

    def save_snapshot(self):
        if self.last_annotated_frame is None:
            messagebox.showinfo("Aviso", "No hay frame visible para guardar.")
            return

        os.makedirs("snapshots_mvp", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join("snapshots_mvp", f"snapshot_{timestamp}.jpg")
        cv2.imwrite(out_path, self.last_annotated_frame)
        self.status_var.set(f"Captura guardada en: {out_path}")

    # --------------------------------------------------------
    # Cierre
    # --------------------------------------------------------
    def on_close(self):
        self.stop_capture()
        self.root.destroy()


def main():
        root = tk.Tk()
        app = PokedexSquirtleApp(root)
        root.mainloop()


if __name__ == "__main__":
    main()