import tkinter as tk
from tkinter import ttk, messagebox
import naivebayes
import mochila

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

        # Pestaña Naive Bayes
        self.tab_nb = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_nb, text='Diagnóstico Cáncer')
        self.crear_interfaz_naivebayes()

        # Pestaña Mochila
        self.tab_mochila = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_mochila, text='Problema Mochila')
        self.crear_interfaz_mochila()

        self.tab_control.pack(expand=1, fill='both')

    # --- Interfaz Naive Bayes ---
    def crear_interfaz_naivebayes(self):
        # Entrenar el modelo y obtener columnas
        self.model, self.le, self.features, self.accuracy = naivebayes.entrenar_modelo()
        tk.Label(self.tab_nb, text=f"Precisión del modelo: {self.accuracy:.2f}", font=("Arial", 14)).pack(pady=10)

        self.sliders = {}
        frame_sliders = ttk.Frame(self.tab_nb)
        frame_sliders.pack(padx=10, pady=10, fill='x')

        # Para no mostrar las 30 características, tomamos las 10 primeras para interfaz (puedes cambiar)
        features_to_use = self.features[:10]

        for feat in features_to_use:
            label = ttk.Label(frame_sliders, text=feat)
            label.pack(anchor='w')
            slider = ttk.Scale(frame_sliders, from_=0, to=100, orient='horizontal')
            slider.pack(fill='x', pady=2)
            self.sliders[feat] = slider

        btn = ttk.Button(self.tab_nb, text="Predecir Diagnóstico", command=self.predecir_naivebayes)
        btn.pack(pady=20)

        self.result_nb = ttk.Label(self.tab_nb, text="", font=("Arial", 12))
        self.result_nb.pack()

    def predecir_naivebayes(self):
        valores = []
        for feat in self.sliders:
            val = self.sliders[feat].get()
            valores.append(val)
        # Crear dataframe con las columnas que usa el modelo (completo)
        # Para las columnas que no están en sliders, asignamos promedio (podrías ajustar)
        input_data = []
        for feat in self.features:
            if feat in self.sliders:
                input_data.append(self.sliders[feat].get())
            else:
                input_data.append(50)  # valor promedio estándar

        import numpy as np
        arr = np.array(input_data).reshape(1, -1)
        pred = self.model.predict(arr)[0]
        diag = self.le.inverse_transform([pred])[0]
        self.result_nb.config(text=f"Diagnóstico predicho: {diag}")

    # --- Interfaz Mochila ---
    def crear_interfaz_mochila(self):
        # Lista predefinida de objetos: [(peso, valor), ...]
        self.objetos = [
            (10, 60),
            (20, 100),
            (30, 120),
            (25, 75),
            (15, 50)
        ]

        self.mochila_frame = ttk.Frame(self.tab_mochila)
        self.mochila_frame.pack(pady=10)

        tk.Label(self.mochila_frame, text="Capacidad máxima de la mochila:").grid(row=0, column=0, sticky='w')
        self.capacidad_entry = ttk.Entry(self.mochila_frame)
        self.capacidad_entry.grid(row=0, column=1)

        # Mostrar los objetos predefinidos en el Treeview
        self.tree = ttk.Treeview(self.mochila_frame, columns=("peso", "valor"), show='headings', height=8)
        self.tree.heading('peso', text='Peso')
        self.tree.heading('valor', text='Valor')
        self.tree.grid(row=1, column=0, columnspan=2, pady=10)

        for obj in self.objetos:
            self.tree.insert('', 'end', values=obj)

        tk.Label(self.mochila_frame, text="Peso:").grid(row=2, column=0)
        self.peso_entry = ttk.Entry(self.mochila_frame)
        self.peso_entry.grid(row=2, column=1)

        tk.Label(self.mochila_frame, text="Valor:").grid(row=3, column=0)
        self.valor_entry = ttk.Entry(self.mochila_frame)
        self.valor_entry.grid(row=3, column=1)

        btn_agregar = ttk.Button(self.mochila_frame, text="Agregar objeto", command=self.agregar_objeto)
        btn_agregar.grid(row=4, column=0, columnspan=2, pady=5)

        btn_ejecutar = ttk.Button(self.mochila_frame, text="Ejecutar Algoritmo Genético", command=self.ejecutar_mochila)
        btn_ejecutar.grid(row=5, column=0, columnspan=2, pady=10)

        self.result_mochila = ttk.Label(self.mochila_frame, text="", font=("Arial", 12))
        self.result_mochila.grid(row=6, column=0, columnspan=2, pady=10)

    def crear_interfaz_mochila(self):
        # Botón para ejecutar algoritmo genético
        self.btn_ejecutar = ttk.Button(self.tab_mochila, text="Ejecutar Algoritmo Genético", command=self.ejecutar_mochila)
        self.btn_ejecutar.pack(pady=10)

        # Text widget para mostrar resultados
        self.resultado_mochila = tk.Text(self.tab_mochila, height=20, width=80)
        self.resultado_mochila.pack(pady=10)

    def ejecutar_mochila(self):
        seleccion, peso, valor, historia = mochila.ejecutar_genetico()

        self.resultado_mochila.delete(1.0, tk.END)
        self.resultado_mochila.insert(tk.END, "Objetos seleccionados:\n")
        for obj in seleccion:
            self.resultado_mochila.insert(tk.END, f"Peso: {obj['peso']}, Valor: {obj['valor']}\n")

        self.resultado_mochila.insert(tk.END, f"\nPeso total: {peso}\n")
        self.resultado_mochila.insert(tk.END, f"Valor total: {valor}\n")

        self.resultado_mochila.insert(tk.END, "\nMejor valor por generación:\n")
        self.resultado_mochila.insert(tk.END, ", ".join(str(v) for v in historia))


# Ejecutar la app
if __name__ == "__main__":
    app = App()
    app.mainloop()