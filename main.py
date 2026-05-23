"""
Attendance System GUI
Real-time face recognition with liveness detection and emotion display.
Innovative feature: attendance logging with timestamps, exportable to CSV.
"""

import os
import json
import csv
import time
import threading
import datetime
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from PIL import Image, ImageTk

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms

from models import FaceEmbeddingNet, EmotionNet, LivenessNet
import config

# emotion display info
EMOTION_LABELS = config.EMOTION_CLASSES
EMOTION_ICONS = ['😠', '🤢', '😨', '😊', '😐', '😢', '😲']

# inference-time transforms (no augmentation)
face_infer_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

emotion_infer_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


class FaceSystem:
    """Handles all ML model inference and the face database."""

    def __init__(self):
        self.device = config.DEVICE
        print(f"Loading models on {self.device}...")

        self.face_model = self._load_face_model()
        self.emotion_model = self._load_emotion_model()
        self.liveness_model = self._load_liveness_model()

        # opencv haar cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)

        # face database: name -> embedding tensor
        self.db_path = 'face_database.json'
        self.face_db = {}
        self._load_db()

        self.threshold = config.VERIFICATION_THRESHOLD

    def _load_face_model(self):
        model = FaceEmbeddingNet(embedding_dim=config.EMBEDDING_DIM, num_classes=None).to(self.device)
        path = os.path.join(config.MODEL_SAVE_DIR, 'face_classification.pth')
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=self.device), strict=False)
            print("Loaded face model.")
        else:
            print("WARNING: face model not found, using random weights (run training first).")
        model.eval()
        return model

    def _load_emotion_model(self):
        model = EmotionNet(num_classes=config.NUM_EMOTIONS).to(self.device)
        path = os.path.join(config.MODEL_SAVE_DIR, 'emotion.pth')
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=self.device))
            print("Loaded emotion model.")
        else:
            print("WARNING: emotion model not found.")
        model.eval()
        return model

    def _load_liveness_model(self):
        model = LivenessNet(pretrained=False).to(self.device)
        path = os.path.join(config.MODEL_SAVE_DIR, 'liveness.pth')
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=self.device))
            print("Loaded liveness model.")
        else:
            print("WARNING: liveness model not found.")
        model.eval()
        return model

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        return faces if len(faces) > 0 else []

    def get_embedding(self, face_rgb):
        """Get l2-normalised face embedding from an RGB numpy array."""
        t = face_infer_transform(face_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.face_model(t)
        return emb.cpu().squeeze(0)

    def predict_liveness(self, face_rgb):
        """Returns (is_real: bool, confidence: float)."""
        t = face_infer_transform(face_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.liveness_model(t)
            probs = F.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).item()
        conf = probs[0, pred].item()
        return pred == 1, conf  # 1 = real

    def predict_emotion(self, face_rgb):
        """Returns (label: str, icon: str, confidence: float)."""
        t = emotion_infer_transform(face_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.emotion_model(t)
            probs = F.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).item()
        conf = probs[0, pred].item()
        return EMOTION_LABELS[pred], EMOTION_ICONS[pred], conf

    def identify(self, embedding):
        """Find best match in database. Returns (name or None, score)."""
        if not self.face_db:
            return None, 0.0
        best_name = None
        best_score = -1.0
        for name, stored in self.face_db.items():
            stored_t = torch.tensor(stored)
            score = F.cosine_similarity(embedding.unsqueeze(0), stored_t.unsqueeze(0)).item()
            if score > best_score:
                best_score = score
                best_name = name
        if best_score >= self.threshold:
            return best_name, best_score
        return None, best_score

    def register(self, name, embedding):
        self.face_db[name] = embedding.tolist()
        self._save_db()

    def remove(self, name):
        if name in self.face_db:
            del self.face_db[name]
            self._save_db()

    def _load_db(self):
        if os.path.exists(self.db_path):
            with open(self.db_path) as f:
                self.face_db = json.load(f)
            print(f"Loaded {len(self.face_db)} registered faces.")

    def _save_db(self):
        with open(self.db_path, 'w') as f:
            json.dump(self.face_db, f)


class AttendanceLogger:
    """
    Innovative feature: logs attendance with timestamps to a CSV file.
    Prevents duplicate entries within a short cooldown window.
    """

    LOG_FILE = 'attendance_log.csv'
    COOLDOWN_SEC = 60  # min seconds between logging same person

    def __init__(self):
        self.recent = {}  # name -> last logged timestamp
        if not os.path.exists(self.LOG_FILE):
            with open(self.LOG_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Name', 'Date', 'Time', 'Emotion'])

    def log(self, name, emotion):
        now = time.time()
        last = self.recent.get(name, 0)
        if now - last < self.COOLDOWN_SEC:
            return False  # too soon

        self.recent[name] = now
        dt = datetime.datetime.now()
        with open(self.LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, dt.strftime('%Y-%m-%d'), dt.strftime('%H:%M:%S'), emotion])
        return True

    def get_today_log(self):
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        rows = []
        if os.path.exists(self.LOG_FILE):
            with open(self.LOG_FILE) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['Date'] == today:
                        rows.append(row)
        return rows


class AttendanceApp:
    """Main GUI application."""

    BG = '#1e1e1e'
    PANEL_BG = '#2b2b2b'
    GREEN = '#00e676'
    RED = '#ff1744'
    ORANGE = '#ff9100'
    WHITE = '#ffffff'
    GRAY = '#9e9e9e'

    def __init__(self, root):
        self.root = root
        self.root.title("Attendance System - Face Recognition")
        self.root.configure(bg=self.BG)
        self.root.geometry("1100x750")
        self.root.resizable(False, False)

        self.system = FaceSystem()
        self.logger = AttendanceLogger()

        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.pending_emb = None  # most recent face embedding (for registration)

        self._build_ui()
        self._update_loop()

    def _label(self, parent, text, fg=None, font=None, **kw):
        return tk.Label(parent, text=text, fg=fg or self.WHITE,
                        bg=self.PANEL_BG, font=font or ('Arial', 11), **kw)

    def _build_ui(self):
        # left: camera
        cam_frame = tk.Frame(self.root, bg=self.BG)
        cam_frame.place(x=10, y=10, width=660, height=510)
        self.video_lbl = tk.Label(cam_frame, bg='black')
        self.video_lbl.pack(fill='both', expand=True)

        # right: info panel
        panel = tk.Frame(self.root, bg=self.PANEL_BG, bd=0)
        panel.place(x=680, y=10, width=410, height=510)

        tk.Label(panel, text="ATTENDANCE SYSTEM", fg=self.GREEN, bg=self.PANEL_BG,
                 font=('Arial', 14, 'bold')).pack(pady=(15, 10))

        def section(title):
            tk.Label(panel, text=title, fg=self.GRAY, bg=self.PANEL_BG,
                     font=('Arial', 9)).pack(anchor='w', padx=15, pady=(10, 0))

        section("IDENTITY")
        self.name_lbl = tk.Label(panel, text="No face detected", fg=self.GRAY,
                                 bg=self.PANEL_BG, font=('Arial', 16, 'bold'))
        self.name_lbl.pack(anchor='w', padx=15)

        section("MATCH SCORE")
        self.score_lbl = tk.Label(panel, text="---", fg=self.WHITE,
                                  bg=self.PANEL_BG, font=('Arial', 12))
        self.score_lbl.pack(anchor='w', padx=15)

        section("LIVENESS")
        self.liveness_lbl = tk.Label(panel, text="---", fg=self.WHITE,
                                     bg=self.PANEL_BG, font=('Arial', 12))
        self.liveness_lbl.pack(anchor='w', padx=15)

        section("EMOTION")
        self.emotion_lbl = tk.Label(panel, text="---", fg=self.WHITE,
                                    bg=self.PANEL_BG, font=('Arial', 12))
        self.emotion_lbl.pack(anchor='w', padx=15)

        section("ATTENDANCE STATUS")
        self.attend_lbl = tk.Label(panel, text="---", fg=self.WHITE,
                                   bg=self.PANEL_BG, font=('Arial', 11))
        self.attend_lbl.pack(anchor='w', padx=15)

        # buttons
        btn_frame = tk.Frame(panel, bg=self.PANEL_BG)
        btn_frame.pack(pady=20, fill='x', padx=15)

        tk.Button(btn_frame, text="Register Face", command=self._register,
                  bg='#0066cc', fg='white', font=('Arial', 11, 'bold'),
                  relief='flat', padx=10, pady=6).pack(side='left', padx=(0, 8))

        tk.Button(btn_frame, text="View Log", command=self._show_log,
                  bg='#444', fg='white', font=('Arial', 11),
                  relief='flat', padx=10, pady=6).pack(side='left')

        # registered count
        section("REGISTERED FACES")
        self.reg_count_lbl = tk.Label(panel, text="0", fg=self.WHITE,
                                      bg=self.PANEL_BG, font=('Arial', 12))
        self.reg_count_lbl.pack(anchor='w', padx=15)

        # bottom: today's attendance table
        table_frame = tk.Frame(self.root, bg=self.BG)
        table_frame.place(x=10, y=530, width=1080, height=200)

        tk.Label(table_frame, text="Today's Attendance", fg=self.GRAY, bg=self.BG,
                 font=('Arial', 10)).pack(anchor='w')

        cols = ('Name', 'Date', 'Time', 'Emotion')
        self.tree = ttk.Treeview(table_frame, columns=cols, show='headings', height=6)
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=260)

        style = ttk.Style()
        style.configure('Treeview', background='#2b2b2b', foreground='white',
                        fieldbackground='#2b2b2b', rowheight=25)
        style.configure('Treeview.Heading', background='#333', foreground='white')

        self.tree.pack(fill='both', expand=True)

        # status bar
        self.status_lbl = tk.Label(self.root, text="Ready", bg='#333', fg='white',
                                   font=('Arial', 9), anchor='w')
        self.status_lbl.place(x=0, y=720, width=1100, height=25)

    def _update_loop(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)  # mirror
            display = frame.copy()

            faces = self.system.detect_faces(frame)

            if len(faces) > 0:
                # pick the largest detected face
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

                pad = 15
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(frame.shape[1], x + w + pad)
                y2 = min(frame.shape[0], y + h + pad)
                face_crop = frame[y1:y2, x1:x2]
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

                # liveness check first
                is_real, live_conf = self.system.predict_liveness(face_rgb)

                if is_real:
                    emb = self.system.get_embedding(face_rgb)
                    self.pending_emb = emb

                    name, score = self.system.identify(emb)
                    emotion, icon, emo_conf = self.system.predict_emotion(face_rgb)

                    # update info panel
                    if name:
                        self.name_lbl.config(text=name, fg=self.GREEN)
                        self.score_lbl.config(text=f"{score:.3f}", fg=self.GREEN)
                        display_name = name

                        # log attendance
                        logged = self.logger.log(name, emotion)
                        if logged:
                            self.attend_lbl.config(text=f"Logged at {datetime.datetime.now().strftime('%H:%M:%S')}", fg=self.GREEN)
                            self._refresh_table()
                        else:
                            self.attend_lbl.config(text="Already logged today", fg=self.GRAY)
                    else:
                        self.name_lbl.config(text="Unknown (press Register)", fg=self.ORANGE)
                        self.score_lbl.config(text=f"{score:.3f}", fg=self.ORANGE)
                        display_name = "Unknown"
                        self.attend_lbl.config(text="Not registered", fg=self.GRAY)

                    self.liveness_lbl.config(text=f"REAL  ({live_conf:.2f})", fg=self.GREEN)
                    self.emotion_lbl.config(text=f"{icon}  {emotion}  ({emo_conf:.2f})")

                    # green box on frame
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 230, 118), 2)
                    cv2.putText(display, display_name, (x, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 118), 2)

                    self.status_lbl.config(text=f"Face detected | liveness: REAL | emotion: {emotion}")

                else:
                    # spoofed face
                    self.name_lbl.config(text="Spoof Detected!", fg=self.RED)
                    self.liveness_lbl.config(text=f"FAKE  ({live_conf:.2f})", fg=self.RED)
                    self.emotion_lbl.config(text="---")
                    self.attend_lbl.config(text="Access denied", fg=self.RED)

                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 220), 2)
                    cv2.putText(display, "FAKE", (x, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 220), 2)

                    self.status_lbl.config(text="Anti-spoofing triggered: fake face detected")
            else:
                self.name_lbl.config(text="No face detected", fg=self.GRAY)
                self.liveness_lbl.config(text="---")
                self.emotion_lbl.config(text="---")
                self.status_lbl.config(text="No face in frame")

            self.reg_count_lbl.config(text=str(len(self.system.face_db)))

            # draw to tkinter
            img = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (660, 495))
            img_pil = Image.fromarray(img)
            img_tk = ImageTk.PhotoImage(img_pil)
            self.video_lbl.config(image=img_tk)
            self.video_lbl.image = img_tk

        self.root.after(30, self._update_loop)

    def _register(self):
        if self.pending_emb is None:
            messagebox.showwarning("Register", "No face detected. Please stand in front of the camera.")
            return
        name = simpledialog.askstring("Register New Face", "Enter name:")
        if name and name.strip():
            name = name.strip()
            self.system.register(name, self.pending_emb)
            messagebox.showinfo("Registered", f"'{name}' has been registered successfully.")
            self.status_lbl.config(text=f"Registered new face: {name}")

    def _show_log(self):
        log_win = tk.Toplevel(self.root)
        log_win.title("Attendance Log")
        log_win.geometry("600x400")
        log_win.configure(bg=self.BG)

        tk.Label(log_win, text="Full Attendance Log", fg=self.WHITE, bg=self.BG,
                 font=('Arial', 13, 'bold')).pack(pady=10)

        cols = ('Name', 'Date', 'Time', 'Emotion')
        tree = ttk.Treeview(log_win, columns=cols, show='headings', height=15)
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=140)
        tree.pack(fill='both', expand=True, padx=10)

        if os.path.exists(AttendanceLogger.LOG_FILE):
            with open(AttendanceLogger.LOG_FILE) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    tree.insert('', 'end', values=(row['Name'], row['Date'], row['Time'], row['Emotion']))

        tk.Button(log_win, text="Export CSV", command=lambda: messagebox.showinfo("Export", f"Log saved to {AttendanceLogger.LOG_FILE}"),
                  bg='#0066cc', fg='white', font=('Arial', 10)).pack(pady=8)

    def _refresh_table(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        for row in self.logger.get_today_log():
            self.tree.insert('', 'end', values=(row['Name'], row['Date'], row['Time'], row['Emotion']))

    def on_close(self):
        self.running = False
        self.cap.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = AttendanceApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == '__main__':
    main()
