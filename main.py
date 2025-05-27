import tkinter as tk
from tkinter import ttk
import numpy as np
import frame_naive_bayes as nb_frame
import mochila
import backprop_module as backprop


PRIMARY_BG   = "#2E3440"
HOVER_BG     = "#434C5E"
BUTTON_ACTIVE = "#81A1C1"
TEXT_COLOR   = "#ECEFF4"
CONTENT_BG   = "#ECEFF4"
CARD_BORDER  = "#D8DEE9"

class SidebarButton(ttk.Frame):
    def __init__(self, parent, text, command):
        super().__init__(parent, style="Sidebar.TFrame")
        self.command = command
        lbl = ttk.Label(self, text=text, style="Sidebar.TLabel", anchor="w")
        lbl.pack(fill="x", expand=True, padx=10, pady=12)
        lbl.bind("<Button-1>", lambda e: self.on_click())
        lbl.bind("<Enter>", lambda e: self.configure(style="SidebarHover.TFrame"))
        lbl.bind("<Leave>", lambda e: self.configure(style="Sidebar.TFrame"))
    def on_click(self):
        for sib in self.master.winfo_children():
            sib.configure(style="Sidebar.TFrame")
        self.configure(style="SidebarActive.TFrame")
        self.command()

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Proyecto Software Inteligente")
        self.geometry("900x600")
        self.configure(bg=PRIMARY_BG)
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Sidebar.TFrame", background=PRIMARY_BG)
        style.configure("SidebarHover.TFrame", background=HOVER_BG)
        style.configure("SidebarActive.TFrame", background=BUTTON_ACTIVE)
        style.configure("Sidebar.TLabel", background=PRIMARY_BG, foreground=TEXT_COLOR, font=("Segoe UI",11))
        style.configure("Content.TFrame", background=CONTENT_BG)
        style.configure("Card.TLabelframe", background=CONTENT_BG, bordercolor=CARD_BORDER, borderwidth=1, relief="solid")
        style.configure("Card.TLabelframe.Label", background=CONTENT_BG, foreground="#2E3440", font=("Segoe UI",12,"bold"))
        style.configure("TButton", font=("Segoe UI",11), padding=(8,4))
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        sidebar = ttk.Frame(self, style="Sidebar.TFrame", width=200)
        sidebar.grid(row=0, column=0, sticky="ns")
        SidebarButton(sidebar, "Diagnóstico Cáncer", self.show_nb).pack(fill="x")
        SidebarButton(sidebar, "Problema Mochila", self.show_mochila).pack(fill="x")
        SidebarButton(sidebar, "Backpropagation", self.show_backprop).pack(fill="x")
        container = ttk.Frame(self, style="Content.TFrame")
        container.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.frames = {}
        for F in (nb_frame.NaiveBayesFrame, MochilaFrame, BackpropFrame):
            f = F(container)
            self.frames[F] = f
            f.grid(row=0, column=0, sticky="nsew")
        sidebar.winfo_children()[0].on_click()
    def show_nb(self):
        self.frames[nb_frame.NaiveBayesFrame].tkraise()
    def show_mochila(self):
        self.frames[MochilaFrame].tkraise()
    def show_backprop(self):
        self.frames[BackpropFrame].tkraise()



class MochilaFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, style="Content.TFrame")
        btn = ttk.Button(self, text="Ejecutar Genético", command=self.run)
        btn.pack(pady=20)
        card = ttk.Labelframe(self, text="Resultado Mochila", style="Card.TLabelframe", padding=10)
        card.pack(fill="both", expand=True, padx=20, pady=5)
        self.txt = tk.Text(card, bg="#FFFFFF", fg="#2E3440", bd=0, highlightthickness=0, font=("Consolas",11), padx=8, pady=8)
        self.txt.pack(fill="both", expand=True)
    def run(self):
        sel,peso,valor,hist = mochila.ejecutar_genetico()
        self.txt.config(state="normal"); self.txt.delete("1.0",tk.END)
        self.txt.insert(tk.END, "Selección:\n")
        for o in sel: self.txt.insert(tk.END, f"peso={o['peso']}, valor={o['valor']}\n")
        self.txt.insert(tk.END, f"\nPeso total: {peso}\nValor total: {valor}\nHistorial: {hist}")
        self.txt.config(state="disabled")

class BackpropFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, style="Content.TFrame")
        ttk.Label(self, text="Backpropagation XOR", font=("Segoe UI",16,"bold"), background=CONTENT_BG).pack(pady=10)
        ttk.Button(self, text="Ejecutar", command=self.run).pack(pady=10)
        card = ttk.Labelframe(self, text="Salida de la red", style="Card.TLabelframe", padding=10)
        card.pack(fill="both", expand=True, padx=20, pady=5)
        vsb = ttk.Scrollbar(card, orient="vertical")
        vsb.pack(side="right", fill="y")
        self.txt = tk.Text(card, height=8, wrap="none", yscrollcommand=vsb.set,
                           bg="#FFFFFF", fg="#2E3440", bd=0, highlightthickness=0,
                           font=("Consolas",11), padx=8, pady=8)
        self.txt.pack(fill="both", expand=True)
        vsb.config(command=self.txt.yview)
    def run(self):
        out = backprop.ejecutar_backprop()
        rnd = np.round(out,3)
        self.txt.config(state="normal"); self.txt.delete("1.0",tk.END)
        self.txt.insert(tk.END, f"{rnd}")
        self.txt.config(state="disabled")

if __name__ == "__main__":
    App().mainloop()
