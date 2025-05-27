import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import mochila



PRIMARY_BG   = "#2E3440"
HOVER_BG     = "#434C5E"
BUTTON_ACTIVE = "#81A1C1"
TEXT_COLOR   = "#ECEFF4"
CONTENT_BG   = "#ECEFF4"
CARD_BORDER  = "#D8DEE9"

class MochilaFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, style="Content.TFrame")

        frm = ttk.Frame(self, style="Content.TFrame")
        frm.pack(fill="x", pady=(0,10))

        ttk.Label(frm, text="Capacidad máxima:", background=CONTENT_BG).grid(row=0, column=0, sticky="w")
        self.cap_entry = ttk.Entry(frm, width=6)
        self.cap_entry.grid(row=0, column=1, padx=(5,20))
        self.cap_entry.insert(0, "6")  

        ttk.Label(frm, text="Peso:", background=CONTENT_BG).grid(row=0, column=2, sticky="w")
        self.peso_entry = ttk.Entry(frm, width=6)
        self.peso_entry.grid(row=0, column=3, padx=5)
        ttk.Label(frm, text="Valor:", background=CONTENT_BG).grid(row=0, column=4, sticky="w")
        self.valor_entry = ttk.Entry(frm, width=6)
        self.valor_entry.grid(row=0, column=5, padx=5)

        btn_add = ttk.Button(frm, text="Agregar objeto", command=self.agregar_objeto)
        btn_add.grid(row=0, column=6, padx=10)

        self.tree = ttk.Treeview(self, columns=("peso","valor"), show="headings", height=6)
        self.tree.heading("peso", text="Peso")
        self.tree.heading("valor", text="Valor")
        self.tree.pack(fill="x", padx=20)

        ttk.Button(self, text="Ejecutar Genético", command=self.run).pack(pady=10)

        card = ttk.Labelframe(self, text="Resultado Mochila", style="Card.TLabelframe", padding=10)
        card.pack(fill="both", expand=True, padx=20, pady=5)
        self.txt = tk.Text(card, bg="#FFFFFF", fg="#2E3440", bd=0, highlightthickness=0,
                           font=("Consolas",11), padx=8, pady=8)
        self.txt.pack(fill="both", expand=True)

    def agregar_objeto(self):
        try:
            w = float(self.peso_entry.get())
            v = float(self.valor_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Peso y valor deben ser numéricos")
            return
        self.tree.insert("", "end", values=(w,v))
        self.peso_entry.delete(0, tk.END)
        self.valor_entry.delete(0, tk.END)

    def run(self):
        # 1) Leer capacidad
        try:
            cap = float(self.cap_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Capacidad debe ser numérica")
            return

        # 2) Leer lista de objetos de la tabla
        objs = []
        for iid in self.tree.get_children():
            w,v = self.tree.item(iid, "values")
            objs.append({"peso": float(w), "valor": float(v)})

        if not objs:
            messagebox.showwarning("Atención", "Agrega al menos un objeto")
            return

        # 3) Volcar al módulo
        mochila.objetos = objs
        mochila.CAPACIDAD_MAXIMA = cap

        # 4) Ejecutar el genético
        sel, peso, valor, historia = mochila.ejecutar_genetico()

        # 5) Mostrar resultados
        self.txt.config(state="normal")
        self.txt.delete("1.0", tk.END)
        self.txt.insert(tk.END, "Selección:\n")
        for o in sel:
            self.txt.insert(tk.END, f"  • peso={o['peso']}, valor={o['valor']}\n")
        self.txt.insert(tk.END, f"\nPeso total: {peso}\nValor total: {valor}\n")
        self.txt.insert(tk.END, "Historia:\n")
        self.txt.insert(tk.END, ", ".join(str(x) for x in historia))
        self.txt.config(state="disabled")