import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

import naivebayes
import mochila
import backprop_module as backprop  

mochila.objetos = [
    {"peso": 3, "valor": 25},
    {"peso": 2, "valor": 20},
    {"peso": 1, "valor": 15},
    {"peso": 4, "valor": 40},
    {"peso": 5, "valor": 50},
]
mochila.CAPACIDAD_MAXIMA = 6

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Proyecto Software Inteligente")
        self.geometry("900x700")

        self.tab_control = ttk.Notebook(self)
        
        # 1) Diagnóstico Cáncer
        self.tab_nb = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_nb, text='Diagnóstico Cáncer')
        self.crear_interfaz_naivebayes()

        # 2) Problema Mochila
        self.tab_mochila = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_mochila, text='Problema Mochila')
        self.crear_interfaz_mochila()

        # 3) Backpropagation
        self.tab_bp = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_bp, text='Backpropagation')
        self.crear_interfaz_backprop()

        self.tab_control.pack(expand=1, fill='both')

    def crear_interfaz_naivebayes(self):
        self.model, self.le, self.features, self.accuracy = naivebayes.entrenar_modelo()
        ttk.Label(self.tab_nb, text=f"Precisión del modelo: {self.accuracy:.2f}", font=("Arial", 14)).pack(pady=10)

        frame_sliders = ttk.Frame(self.tab_nb)
        frame_sliders.pack(padx=10, pady=10, fill='x')

        self.sliders = {}
        features_to_use = self.features[:10]
        for feat in features_to_use:
            ttk.Label(frame_sliders, text=feat).pack(anchor='w')
            s = ttk.Scale(frame_sliders, from_=0, to=100, orient='horizontal')
            s.pack(fill='x', pady=2)
            self.sliders[feat] = s

        ttk.Button(self.tab_nb, text="Predecir Diagnóstico", command=self.predecir_naivebayes).pack(pady=20)
        self.result_nb = ttk.Label(self.tab_nb, text="", font=("Arial", 12))
        self.result_nb.pack()

    def predecir_naivebayes(self):
        input_data = [
            self.sliders[f].get() if f in self.sliders else 50
            for f in self.features
        ]
        arr = np.array(input_data).reshape(1, -1)
        pred = self.model.predict(arr)[0]
        diag = self.le.inverse_transform([pred])[0]
        self.result_nb.config(text=f"Diagnóstico predicho: {diag}")

    def crear_interfaz_mochila(self):
        ttk.Button(self.tab_mochila, text="Ejecutar Algoritmo Genético", command=self.ejecutar_mochila).pack(pady=10)
        self.resultado_mochila = tk.Text(self.tab_mochila, height=20, width=80)
        self.resultado_mochila.pack(pady=10)

    def ejecutar_mochila(self):
        seleccion, peso, valor, historia = mochila.ejecutar_genetico()
        self.resultado_mochila.delete(1.0, tk.END)
        self.resultado_mochila.insert(tk.END, "Objetos seleccionados:\n")
        for obj in seleccion:
            self.resultado_mochila.insert(tk.END, f"  • peso={obj['peso']}, valor={obj['valor']}\n")
        self.resultado_mochila.insert(tk.END, f"\nPeso total: {peso}\n")
        self.resultado_mochila.insert(tk.END, f"Valor total: {valor}\n")
        self.resultado_mochila.insert(tk.END, "\nMejor valor por generación:\n")
        self.resultado_mochila.insert(tk.END, ", ".join(str(v) for v in historia))

    def crear_interfaz_backprop(self):
        ttk.Label(self.tab_bp, text="Backpropagation para XOR", font=("Arial", 14)).pack(pady=10)
        ttk.Button(self.tab_bp, text="Ejecutar Backpropagation", command=self.ejecutar_backprop).pack(pady=20)
        self.result_bp = tk.Text(self.tab_bp, height=10, width=60)
        self.result_bp.pack(pady=10)

    def ejecutar_backprop(self):
        output = backprop.ejecutar_backprop()
        self.result_bp.delete(1.0, tk.END)
        rounded = np.round(output, 3)
        self.result_bp.insert(tk.END, "Salida final de la red (XOR):\n")
        self.result_bp.insert(tk.END, f"{rounded}")

if __name__ == "__main__":
    app = App()
    app.mainloop()
