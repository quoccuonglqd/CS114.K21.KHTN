r"""Date: 12/3/2020
    Author: Nguyen Quoc Cuong
"""
from tkinter import *
import os
from PIL import ImageTk,Image
from Component import Component

class Titlebar(Component):
	r"""A Titlebar class for displaying titlebar

    Args:
        master(tkinter.Tk): reference to the window
    """
	ROW = 0
	FONT_PATH = 'Font/arialbd.ttf'
	TEXT_SIZE = 14
	LABEL_FG = "#380937"
	ICON_PATH = 'image/81CP2QSYmdL._SY355_.jpg'
	ICON_HEIGHT = 32
	ICON_WIDTH = 30
	ICON_COLUMN = 0
	LABEL_HEIGHT = 1
	LABEL_WIDTH = 63
	LABEL_BG = '#66ffff'
	LABEL_COLUMN = 1
	QUITBUTTON_HEIGHT = 30
	QUITBUTTON_WIDTH = 30
	QUITBUTTON_PATH = 'image/quitbutton.png'
	QUITBUTTON_COLUMN = 3
	MINIMIZEBUTTON_HEIGHT = 30
	MINIMIZEBUTTON_WIDTH = 30
	MINIMIZEBUTTON_PATH = 'image/minimizebutton.jpg'
	MINIMIZEBUTTON_COLUMN = 2
	TITLEBAR_RELIEF = RAISED
	TITLEBAR_BG = "#66ffff"

	def __init__(self,app,master):
		super(Titlebar, self).__init__(master,Titlebar.TITLEBAR_RELIEF,Titlebar.TITLEBAR_BG)
		
		self.icon_img = ImageTk.PhotoImage( Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),Titlebar.ICON_PATH)))
		self.icon = Label(self,image=self.icon_img,height=Titlebar.ICON_HEIGHT,width=Titlebar.ICON_WIDTH)
		self.icon.grid(row=Titlebar.ROW,column=Titlebar.ICON_COLUMN)

		self.label1 = Label(self,text='    DEMO   ',font=(Titlebar.FONT_PATH,Titlebar.TEXT_SIZE),foreground=Titlebar.LABEL_FG,
			anchor=W,height=Titlebar.LABEL_HEIGHT,width=Titlebar.LABEL_WIDTH,bg=Titlebar.LABEL_BG)
		self.label1.grid(row=Titlebar.ROW,column=Titlebar.LABEL_COLUMN)

		self.quitbutton_img = ImageTk.PhotoImage(Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),Titlebar.QUITBUTTON_PATH)))
		self.quitbutton = Button(self,image=self.quitbutton_img,command = self.master.quit,height=Titlebar.QUITBUTTON_HEIGHT,width=Titlebar.QUITBUTTON_WIDTH)
		self.quitbutton.grid(row=Titlebar.ROW,column=Titlebar.QUITBUTTON_COLUMN)

		self.minimizebutton_img = ImageTk.PhotoImage(Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),Titlebar.MINIMIZEBUTTON_PATH)))
		self.minimizebutton = Button(self,image=self.minimizebutton_img,command = self.Minimize,height=Titlebar.MINIMIZEBUTTON_HEIGHT,width=Titlebar.MINIMIZEBUTTON_WIDTH)
		self.minimizebutton.grid(row=Titlebar.ROW,column=Titlebar.MINIMIZEBUTTON_COLUMN)
		
		self.grid(row=0,column=0,columnspan=2)


		self.grid(row=0,column=0)
		self.bind("<Map>",self.Frame_mapped)

	def Frame_mapped(self,e):
		r"""
            Reconstruct the screen
        """
		self.master.update_idletasks()
		self.master.overrideredirect(True)
		self.master.state('normal')
		self.master.master.overrideredirect(False)

	def Minimize(self):
		r"""
            Minimize the window
        """
		self.master.update_idletasks()
		self.master.overrideredirect(False)
		self.master.state('iconic')
		self.master.master.overrideredirect(True)

		
