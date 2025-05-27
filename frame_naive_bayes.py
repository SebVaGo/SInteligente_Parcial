import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import naivebayes


PRIMARY_BG   = "#2E3440"
HOVER_BG     = "#434C5E"
BUTTON_ACTIVE = "#81A1C1"
TEXT_COLOR   = "#ECEFF4"
CONTENT_BG   = "#ECEFF4"
CARD_BORDER  = "#D8DEE9"


class NaiveBayesFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, style="Content.TFrame")

        params_frame = ttk.Labelframe(self, text="Parámetros", style="Card.TLabelframe", padding=8)
        params_frame.pack(fill="x", padx=20, pady=10)

        ttk.Label(params_frame, text="Test size:", background=CONTENT_BG).grid(row=0, column=0, sticky="w")
        self.test_size_entry = ttk.Entry(params_frame, width=6); self.test_size_entry.insert(0, "0.2")
        self.test_size_entry.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(params_frame, text="Random state:", background=CONTENT_BG).grid(row=0, column=2, sticky="w")
        self.random_state_entry = ttk.Entry(params_frame, width=6); self.random_state_entry.insert(0, "22")
        self.random_state_entry.grid(row=0, column=3, padx=5, pady=2)

        ttk.Label(params_frame, text="Imputer:", background=CONTENT_BG).grid(row=0, column=4, sticky="w")
        self.imputer_combo = ttk.Combobox(params_frame, values=["mean","median"], width=8, state="readonly")
        self.imputer_combo.set("mean")
        self.imputer_combo.grid(row=0, column=5, padx=5, pady=2)

        ttk.Button(self, text="Entrenar / Actualizar", command=self.train).pack(pady=(0,10))

        # Logs
        logs_card = ttk.Labelframe(self, text="Logs de entrenamiento", style="Card.TLabelframe", padding=8)
        logs_card.pack(fill="both", expand=True, padx=20, pady=5)
        self.log_txt = tk.Text(logs_card, height=8, wrap="word",
                               bg="#FFFFFF", fg="#2E3440", bd=0, highlightthickness=0,
                               font=("Consolas",10), padx=6, pady=6)
        vsb = ttk.Scrollbar(logs_card, orient="vertical", command=self.log_txt.yview)
        self.log_txt.configure(yscrollcommand=vsb.set)
        self.log_txt.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # Área de sliders y resultado (puedes reusar lo que tenías)
        self.sliders_card = ttk.Labelframe(self, text="Atributos", style="Card.TLabelframe", padding=8)
        self.sliders_card.pack(fill="x", padx=20, pady=(10,5))
        self.sliders = {}

        self.result_card = ttk.Labelframe(self, text="Resultado", style="Card.TLabelframe", padding=8)
        self.result_card.pack(fill="x", padx=20, pady=(5,20))
        self.lbl_res = ttk.Label(self.result_card, text="", font=("Segoe UI",12), background=CONTENT_BG)
        self.lbl_res.pack()

        # Entrenamos por primera vez
        self.train()

    def train(self):
        # recoger parámetros
        ts = float(self.test_size_entry.get())
        rs = int(self.random_state_entry.get())
        imp = self.imputer_combo.get()

        model, le, features, acc, logs = naivebayes.entrenar_modelo(
            test_size=ts, random_state=rs, imputer_strategy=imp
        )
        # actualizamos precisión
        if hasattr(self, "acc_lbl"):
            self.acc_lbl.config(text=f"Precisión: {acc:.2f}")
        else:
            self.acc_lbl = ttk.Label(self, text=f"Precisión: {acc:.2f}", font=("Segoe UI",14), background=CONTENT_BG)
            self.acc_lbl.pack(pady=5)

        # volcamos logs
        self.log_txt.config(state="normal")
        self.log_txt.delete("1.0", tk.END)
        for line in logs:
            self.log_txt.insert(tk.END, line + "\n")
        self.log_txt.config(state="disabled")

        # reconstruimos sliders
        for w in self.sliders_card.winfo_children():
            w.destroy()
        self.sliders.clear()
        for feat in features[:10]:
            ttk.Label(self.sliders_card, text=feat, background=CONTENT_BG).pack(anchor="w")
            s = ttk.Scale(self.sliders_card, from_=0, to=100, orient="horizontal")
            s.pack(fill="x", pady=2)
            self.sliders[feat] = s

        # guardamos para predecir
        self._model, self._le, self._features = model, le, features

        # enlazamos botón predecir
        if not hasattr(self, "pred_btn"):
            ttk.Button(self, text="Predecir", command=self.predict).pack(pady=5)

    def predict(self):
        data = [self.sliders[f].get() if f in self.sliders else 50 for f in self._features]
        arr = np.array(data).reshape(1,-1)
        p = self._model.predict(arr)[0]
        diag = self._le.inverse_transform([p])[0]
        self.lbl_res.config(text=diag)