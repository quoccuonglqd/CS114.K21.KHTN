r"""Date: 12/3/2020
    Author: Nguyen Quoc Cuong
"""
from tkinter import *

class Component(Frame):
	r"""A generic Commponent class derived from tkinter.Frame to contain widget

	    Args:
	        the_window(tkinter.Tk): reference to the window
	        relief('flat','sunken','raised','groove','ridge'): border decoration
	        bg(str): background color
	        width(int): width
	        height(int): height
	""" 
	def __init__(self,the_window,relief='raised',bg='SystemButtonFace',width=0,height=0):
		super().__init__(the_window)
		self["relief"] = relief
		self["bg"] = bg
		self["width"] = width
		self["height"] = height
		self["bg"] = bg

	def Disable(self):
		for child in self.winfo_children():
			child.configure(state='disable')
		self.lower()

	def Enable(self):
		for child in self.winfo_children():
			child.configure(state='normal')
		self.lift()
		self.master.focus_set()
		self.focus_set()
