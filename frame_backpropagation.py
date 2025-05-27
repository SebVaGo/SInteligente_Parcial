import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
import numpy as np
import backprop_module as backprop


PRIMARY_BG   = "#2E3440"
HOVER_BG     = "#434C5E"
BUTTON_ACTIVE = "#81A1C1"
TEXT_COLOR   = "#ECEFF4"
CONTENT_BG   = "#ECEFF4"
CARD_BORDER  = "#D8DEE9"

class BackpropFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, style="Content.TFrame")

        ttk.Label(self, text="Backpropagation (CSV o manual)", 
                  font=("Segoe UI",16,"bold"), background=CONTENT_BG).pack(pady=10)

        params = ttk.Labelframe(self, text="Parámetros", style="Card.TLabelframe", padding=8)
        params.pack(fill="x", padx=20, pady=(0,10))
        ttk.Label(params, text="Épocas:", background=CONTENT_BG).grid(row=0, column=0, sticky="w")
        self.epochs_entry = ttk.Entry(params, width=6); self.epochs_entry.insert(0, "10000")
        self.epochs_entry.grid(row=0, column=1, padx=5)
        ttk.Label(params, text="LR:", background=CONTENT_BG).grid(row=0, column=2, sticky="w")
        self.lr_entry = ttk.Entry(params, width=6); self.lr_entry.insert(0, "0.1")
        self.lr_entry.grid(row=0, column=3, padx=5)
        ttk.Label(params, text="Neur. ocultas:", background=CONTENT_BG).grid(row=0, column=4, sticky="w")
        self.hidden_entry = ttk.Entry(params, width=6); self.hidden_entry.insert(0, "2")
        self.hidden_entry.grid(row=0, column=5, padx=5)

        btns = ttk.Frame(self, style="Content.TFrame")
        btns.pack(fill="x", padx=20)
        ttk.Button(btns, text="Cargar CSV", command=self.load_csv).pack(side="left")
        ttk.Button(btns, text="Usar manual", command=self.use_manual).pack(side="left", padx=10)

        self.data_card = ttk.Labelframe(self, text="Datos de entrenamiento", 
                                        style="Card.TLabelframe", padding=8)
        self.data_card.pack(fill="both", expand=False, padx=20, pady=(10,5))
        self._build_manual_widget()

        ttk.Button(self, text="Entrenar y Ejecutar", command=self.run).pack(pady=10)

        card = ttk.Labelframe(self, text="Salida de la red", style="Card.TLabelframe", padding=8)
        card.pack(fill="both", expand=True, padx=20, pady=(0,20))
        vsb = ttk.Scrollbar(card, orient="vertical")
        vsb.pack(side="right", fill="y")
        self.out_txt = tk.Text(card, height=8, wrap="none", yscrollcommand=vsb.set,
                               bg="#FFFFFF", fg="#2E3440", bd=0, highlightthickness=0,
                               font=("Consolas",11), padx=8, pady=8)
        self.out_txt.pack(fill="both", expand=True)
        vsb.config(command=self.out_txt.yview)

        self._csv_data = None   

    def _build_manual_widget(self):
        for w in self.data_card.winfo_children(): w.destroy()
        instr = ("Formato por línea: x1,x2,...,xn->y1,y2,...,ym\n"
                 "Ejemplo XOR: 0,1->1")
        ttk.Label(self.data_card, text=instr, background=CONTENT_BG, font=("Segoe UI",9)).pack(anchor="w", pady=(0,5))
        self.data_txt = tk.Text(self.data_card, height=5, font=("Consolas",10), padx=6, pady=6)
        self.data_txt.pack(fill="both", expand=True)
        # precarga
        self.data_txt.insert("1.0", "0,0->0\n0,1->1\n1,0->1\n1,1->0")
        self._csv_data = None

    def _build_csv_widget(self, path):
        for w in self.data_card.winfo_children(): w.destroy()
        ttk.Label(self.data_card, text=f"CSV cargado:\n{path}", 
                  background=CONTENT_BG, font=("Segoe UI",10)).pack(anchor="w", pady=5)

    def load_csv(self):
        path = filedialog.askopenfilename(
            filetypes=[("CSV files","*.csv"),("All","*.*")])
        if not path:
            return
        try:
            df = pd.read_csv(path, header=None)
        except Exception as e:
            messagebox.showerror("Error al leer CSV", str(e))
            return
        if df.shape[1] < 2:
            messagebox.showerror("Formato inválido", "El CSV debe tener al menos 2 columnas")
            return
        self._csv_data = df
        self._build_csv_widget(path)

    def use_manual(self):
        self._build_manual_widget()

    def run(self):
        epochs = int(self.epochs_entry.get())
        lr     = float(self.lr_entry.get())
        hidden = int(self.hidden_entry.get())

        # decidir fuente de datos
        if self._csv_data is not None:
            df = self._csv_data.values
            X_arr = df[:, :-1].astype(float)
            Y_arr = df[:, -1].reshape(-1, 1).astype(float)
        else:
            lines = self.data_txt.get("1.0", "end").strip().splitlines()
            X_list, Y_list = [], []
            for line in lines:
                if "->" not in line: continue
                left, right = line.split("->")
                xi = [float(x) for x in left.split(",") if x!=""]
                yi = [float(y) for y in right.split(",") if y!=""]
                X_list.append(xi)
                Y_list.append(yi)
            X_arr = np.array(X_list)
            Y_arr = np.array(Y_list)

        if X_arr.size == 0:
            messagebox.showerror("Error","No hay datos válidos para entrenar")
            return

        salida = backprop.ejecutar_backprop(
            inputs=X_arr,
            expected_output=Y_arr,
            epochs=epochs,
            lr=lr,
            hidden_neurons=hidden
        )

        self.out_txt.config(state="normal")
        self.out_txt.delete("1.0","end")
        self.out_txt.insert("end", "Salida final:\n")
        self.out_txt.insert("end", np.round(salida, 4).tolist())
        self.out_txt.config(state="disabled")