import tkinter as tk
from tkinter import ttk
import mochila_module as mm
import naive_bayes_module as nbm

class ProjectFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="white")
        self.controller = controller
    def show(self):
        self.lift()

class HomeFrame(ProjectFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        label = tk.Label(self, text="Parcial Software Inteligente",
                         font=("Helvetica", 24), bg="white")
        label.place(relx=0.5, rely=0.5, anchor="center")

class MochilaFrame(ProjectFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        params = tk.Frame(self, bg="white")
        tk.Label(params, text="Capacidad máxima:", bg="white").grid(row=0, column=0, sticky="e")
        self.cap_entry = tk.Entry(params, width=10)
        self.cap_entry.insert(0, "6")
        self.cap_entry.grid(row=0, column=1, padx=5, pady=5)
        params.pack(pady=5)

        list_frame = tk.Frame(self, bg="white")
        canvas = tk.Canvas(list_frame, bg="white", height=200)
        scroll = tk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        self.obj_frame = tk.Frame(canvas, bg="white")
        self.obj_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0,0), window=self.obj_frame, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        list_frame.pack(fill="x", padx=10)

        self.entries = []
        default_objs = [
            ("Cuaderno",1.2,15),("Lápiz",0.1,5),("Plumones",2.0,25),
            ("Regla",0.5,8),("Escuadra",1.8,20),("Calculadora",0.7,35),
            ("Compás",0.6,12),("Libro de texto",3.5,50),("Tablet",1.3,55),
            ("USB",0.2,10),("Agenda",0.8,18),("Colores",1.1,22)
        ]
        for i,(name,peso,valor) in enumerate(default_objs):
            tk.Label(self.obj_frame, text=name, bg="white").grid(row=i, column=0, sticky="w", padx=5, pady=2)
            w = tk.Entry(self.obj_frame, width=8);
            w.insert(0,str(peso)); w.grid(row=i, column=1, padx=5)
            v = tk.Entry(self.obj_frame, width=8);
            v.insert(0,str(valor)); v.grid(row=i, column=2, padx=5)
            self.entries.append((name,w,v))

        self.btn = tk.Button(self, text="Ejecutar Mochila", command=self.run,
                             width=25, height=2, font=("Helvetica",12))
        self.btn.pack(pady=10)
        self.text = tk.Text(self, bg="white", state="disabled", width=80, height=15)
        self.text.pack(pady=5)

    def run(self):
        try:
            mm.CAPACIDAD_MAXIMA = float(self.cap_entry.get())
        except:
            mm.CAPACIDAD_MAXIMA = 0
        mm.objetos = []
        for name,w_e,v_e in self.entries:
            try:
                w=float(w_e.get()); v=float(v_e.get())
                mm.objetos.append({"nombre":name, "peso":w, "valor":v})
            except:
                continue
        selec,peso,val,his = mm.ejecutar_genetico()
        self.text.config(state="normal"); self.text.delete("1.0",tk.END)
        self.text.insert(tk.END,"Mochila óptima contiene:\n")
        for o in selec:
            self.text.insert(tk.END,f"- {o['nombre']} (Peso: {o['peso']}kg, Valor: {o['valor']})\n")
        self.text.insert(tk.END,f"\nPeso total: {peso:.2f}kg / {mm.CAPACIDAD_MAXIMA}kg\n")
        self.text.insert(tk.END,f"Valor total: {val}\nGeneraciones: {len(his)}\n\n")
        self.text.insert(tk.END,"Evolución por generación:\n")
        for i,fit in enumerate(his,1): self.text.insert(tk.END,f"Gen {i}: {fit}\n")
        self.text.config(state="disabled")

class NaiveBayesFrame(ProjectFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.btn = tk.Button(self, text="Ejecutar Naive Bayes", command=self.run,
                             width=25, height=2, font=("Helvetica",12))
        self.btn.pack(pady=20)
        self.label = tk.Label(self, text="", font=("Helvetica",14), bg="white")
        self.label.pack(pady=10)

    def run(self):
        accuracy = nbm.ejecutar_naive_bayes()
        self.label.config(text=f"Accuracy: {accuracy*100:.2f}%")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Launcher de Proyectos")
        self.geometry("950x800")
        container = tk.Frame(self); container.pack(fill="both",expand=True)
        menu = tk.Frame(self,width=200,bg="#ddd"); menu.pack(side="left",fill="y")
        buttons = [("Inicio",HomeFrame),("Mochila",MochilaFrame),("Naive Bayes",NaiveBayesFrame)]
        for name,cls in buttons:
            btn=tk.Button(menu,text=name,command=lambda c=cls:self.show_frame(c),
                          width=20,height=2,font=("Helvetica",12))
            btn.pack(pady=10)
        self.frames={}
        for _,cls in buttons:
            f=cls(container,self); f.place(relx=0,rely=0,relwidth=1,relheight=1)
            self.frames[cls]=f
        self.show_frame(HomeFrame)
    def show_frame(self,cls): self.frames[cls].show()

if __name__=="__main__": app=App(); app.mainloop()