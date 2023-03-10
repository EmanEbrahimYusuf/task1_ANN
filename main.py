from msilib.schema import TextStyle
import tkinter as tk
from tkinter import *
#from tkinter.tix import Select
from turtle import position, width
from classtask1 import TaskOne
from classtask1 import confusion_matrix
from task1 import task
# setting root window:
cols=[]
specise=[]
lr=0
epo=0
root = tk.Tk()
root.title("Tkinter Checkbox")
root.config(bg="black")
root.geometry("850x340")

# label text1:
tk.Label(root, text="Featurs", font="Bahnschrift 20", bg="#100E17", fg="green").place(x=130, y=35)
########################################################################features################################################
#############################################

def limit_feature():
    i=0
    cols.clear()
    if(CheckVar1.get()==1):
        i+=1
        cols.append('bill_length_mm')
    if(CheckVar2.get()==1):
        i+=1
        cols.append('bill_depth_mm')
    if(CheckVar3.get()==1):
        i+=1 
        cols.append('flipper_length_mm') 
    if(CheckVar4.get()==1):
        i+=1
        cols.append('gender')
    if(CheckVar5.get()==1):
        i+=1
        cols.append('body_mass_g') 
    if(i>=2):
        if(CheckVar1.get()!=1):
            F1.config(state='disabled')
        if(CheckVar2.get()!=1):
            F2.config(state='disabled')
        if(CheckVar3.get()!=1):
            F3.config(state='disabled')
        if(CheckVar4.get()!=1):
            F4.config(state='disabled')
        if(CheckVar5.get()!=1):
            F5.config(state='disabled')
    else:
        F1.config(state='normal')
        F2.config(state='normal')
        F3.config(state='normal')
        F4.config(state='normal')
        F5.config(state='normal')

CheckVar1 = IntVar()
CheckVar2 = IntVar()
CheckVar3 = IntVar()
CheckVar4 = IntVar()
CheckVar5 = IntVar()
# select features:
F1=Checkbutton(root,variable=CheckVar1, text="bill_length ",font="BahnschriftLight 13", takefocus=0, bg="#137B13", fg="black", activebackground="#137B13", activeforeground="darkgreen", anchor = 'w',bd=0, highlightthickness=0, width=17, selectcolor="#137B13", height=2, command=limit_feature)
F1.place(x=90, y=83)
F2=Checkbutton(root, text="bill_depth",variable=CheckVar2,font="BahnschriftLight 13", takefocus=0, bg="#31A231", fg="black", activebackground="#31A231", activeforeground="green", bd=0,anchor = 'w', highlightthickness=0, width=17, selectcolor="#31A231", height=2, command=limit_feature)
F2.place(x=90, y=120)
F3=Checkbutton(root, text="flipper_length", variable=CheckVar3,font="BahnschriftLight 13", takefocus=0, bg="#5BC85B", fg="black", activebackground="#5BC85B", activeforeground="green", bd=0,anchor = 'w', highlightthickness=0, width=17, selectcolor="#5BC85B", height=2, command=limit_feature)
F3.place(x=90, y=163)
F4=Checkbutton(root, text="gender", font="BahnschriftLight 13",variable=CheckVar4, takefocus=0, bg="#90EE90", fg="black", activebackground="#90EE90", activeforeground="green", bd=0,anchor = 'w', highlightthickness=0, width=17, selectcolor="#90EE90", height=2, command=limit_feature)
F4.place(x=90, y=203)
F5=Checkbutton(root, text="body_mass", font="BahnschriftLight 13",takefocus=0, variable=CheckVar5,bg="#C1FFC1", fg="black", activebackground="#C1FFC1", activeforeground="green", anchor = 'w',bd=0, highlightthickness=0, width=17, selectcolor="#C1FFC1", height=2, command=limit_feature)
F5.place(x=90, y=243)
#############################################################clasess#############################################################################################
def limit_classes():
    i=0
    specise.clear()
    if(CheckVar6.get()==1):
        i+=1
        specise.extend(['Adelie','Gentoo'])
    if(CheckVar7.get()==1):
        i+=1
        specise.extend(['Chinstrap','Gentoo'])
    if(CheckVar8.get()==1):
        i+=1 
        specise.extend(['Adelie','Chinstrap'])
    if(i>=1):
        if(CheckVar6.get()!=1):
            option1.config(state='disabled')
        if(CheckVar7.get()!=1):
            option2.config(state='disabled')
        if(CheckVar8.get()!=1):
            option3.config(state='disabled')

    else:
        option1.config(state='normal')
        option2.config(state='normal')
        option3.config(state='normal')
        

CheckVar6 = IntVar()
CheckVar7 = IntVar()
CheckVar8 = IntVar()

# label text2:
tk.Label(root, text="Classes", font="Bahnschrift 20", bg="#100E17", fg="#f44336").place(x=420, y=35)

# select classes:
option1=tk.Checkbutton(root,variable=CheckVar6, text="Adelie & Gentoo",anchor = 'w', font="BahnschriftLight 13", takefocus=0, bg="#100E30", fg="#F44336", activebackground="#100E38", activeforeground="#F44336", bd=0, highlightthickness=0, width=20, selectcolor="black", height=2,command=limit_classes)
option1.place(x=360, y=83)
option2=tk.Checkbutton(root,variable=CheckVar7 ,text="Gentoo & Chinstrap", anchor = 'w',font="BahnschriftLight 13", takefocus=0, bg="#100E30", fg="#E91E63", activebackground="#100E38", activeforeground="#E91E63", bd=0, highlightthickness=0, width=20, selectcolor="black", height=2,command=limit_classes)
option2.place(x=360, y=120)
option3=tk.Checkbutton(root,variable=CheckVar8 ,text="Adelie & Chinstrap",anchor = 'w', font="BahnschriftLight 13", takefocus=0, bg="#100E30", fg="#9C27B0", activebackground="#100E38", activeforeground="#9C27B0", bd=0, highlightthickness=0, width=20, selectcolor="black", height=2,command=limit_classes)
option3.place(x=360, y=163)
##############################################################################################################################################################################3
tk.Label(root, text="Learning Rate", font="Bahnschrift 20", bg="#100E17", fg="green").place(x=600, y=35)
e=Entry(root)
lr = float('0'+e.get())
e.place(x=605,y=90)
###################################
tk.Label(root, text="Epochs Number", font="Bahnschrift 20", bg="#100E17", fg="green").place(x=600, y=120)
e2=Entry(root)
epo = int('0'+e2.get())
e2.place(x=605,y=180)
#####################################
tk.Label(root, text="Add Bais", font="Bahnschrift 20", bg="#100E17", fg="green").place(x=600, y=220)
check_1 = IntVar()

addbais = Checkbutton(root, \
                 onvalue = 1, offvalue = 0 \
                    ,variable= check_1
                 ,bg="#100E30").place(x=600,y=270)

##########################################
b=Button(root,text="Apply",width=15,bg="#100E17",fg="#f44336",command=lambda :task(cols,specise[0],specise[1],float('0'+e.get()),int('0'+e2.get()),check_1.get()))
b.pack()
b.place(x=700,y=300)
# window in mainloop:
print(check_1.get())
root.mainloop()





