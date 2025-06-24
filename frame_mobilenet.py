import tkinter as tk
from tkinter import filedialog, messagebox
from mobilenet import ejecutar_mobilenet
import os 

class MobilenetFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(bg='#f0f0f0')

        # Título
        title = tk.Label(self, text="Transfer Learning MobileNetV2", font=("Arial", 16, "bold"), bg='#f0f0f0')
        title.pack(pady=10)

        # Selección de ZIP
        btn_zip = tk.Button(self, text="Seleccionar ZIP de Imágenes", command=self.seleccionar_zip)
        btn_zip.pack(pady=5)
        self.lbl_zip = tk.Label(self, text="Ningún archivo seleccionado", bg='#f0f0f0')
        self.lbl_zip.pack(pady=2)

        # Parámetros
        params_frame = tk.Frame(self, bg='#f0f0f0')
        params_frame.pack(pady=10)
        tk.Label(params_frame, text="Épocas:", bg='#f0f0f0').grid(row=0, column=0, sticky='e')
        self.entry_epochs = tk.Entry(params_frame, width=5)
        self.entry_epochs.insert(0, "10")
        self.entry_epochs.grid(row=0, column=1, padx=5)

        tk.Label(params_frame, text="Batch Size:", bg='#f0f0f0').grid(row=0, column=2, sticky='e')
        self.entry_batch = tk.Entry(params_frame, width=5)
        self.entry_batch.insert(0, "32")
        self.entry_batch.grid(row=0, column=3, padx=5)

        tk.Label(params_frame, text="Learning Rate:", bg='#f0f0f0').grid(row=0, column=4, sticky='e')
        self.entry_lr = tk.Entry(params_frame, width=8)
        self.entry_lr.insert(0, "0.001")
        self.entry_lr.grid(row=0, column=5, padx=5)

        # Botón entrenar
        btn_train = tk.Button(self, text="Entrenar", command=self.entrenar)
        btn_train.pack(pady=5)

        # Área de log/resultados
        self.text_log = tk.Text(self, height=15, width=80)
        self.text_log.pack(pady=10)

        # Ruta del ZIP
        self.path_zip = None

    def seleccionar_zip(self):
        path = filedialog.askopenfilename(
            title="Seleccionar archivo ZIP",
            filetypes=[("ZIP files", "*.zip")]
        )
        if path:
            self.path_zip = path
            self.lbl_zip.config(text=os.path.basename(path))

    def entrenar(self):
        if not self.path_zip:
            messagebox.showwarning("Aviso", "Debe seleccionar un archivo ZIP primero.")
            return
        try:
            epochs = int(self.entry_epochs.get())
            batch = int(self.entry_batch.get())
            lr = float(self.entry_lr.get())
        except ValueError:
            messagebox.showerror("Error", "Parámetros inválidos.")
            return

        self.text_log.delete("1.0", tk.END)
        self.text_log.insert(tk.END, "Iniciando entrenamiento...\n")
        history, metrics = ejecutar_mobilenet(self.path_zip, epochs, batch, lr)

        self.text_log.insert(tk.END, f"\nEvaluación final: Pérdida={metrics['loss']:.4f}, Accuracy={metrics['accuracy']:.4f}\n")
        self.text_log.insert(tk.END, "Entrenamiento completado.\n")
        # Opcional: mostrar curvas desde history.history
        try:
            acc = history.history['accuracy']
            loss = history.history['loss']
            val_acc = history.history['val_accuracy']
            val_loss = history.history['val_loss']
            self.text_log.insert(tk.END, f"Epochs: {list(range(1, len(acc)+1))}\n")
        except Exception:
            pass
