from __future__ import absolute_import
from tkinter import *
import os
import os.path as osp
from PIL import ImageTk,Image
from Titlebar import Titlebar
from Character_detect import Character_detect
from Character_recognize import *
from tkinter import filedialog
import cv2 as cv

class NewRoot(Tk):    
    def __init__(self):
        Tk.__init__(self)
        self.attributes('-alpha', 0.0)

class MyMain(Toplevel):
    def __init__(self, master):
        Toplevel.__init__(self, master)
        self.overrideredirect(1)

    def on_close(self):
        self.master.destroy()
        self.destroy()
        
    
class Application(object):
	HEIGHT = 397
	WIDTH = 805

	def __init__(self,master):
		self.master = master
		self.titlebar = Titlebar(self,self.master)
		self.img = cv.cvtColor(cv.imread('image/20140603_0003_BCCTC_0.png'),cv.COLOR_BGR2RGB)
		self.img = ImageTk.PhotoImage(Image.fromarray(self.img).resize((400,100),Image.ANTIALIAS))
		self.image = Label(master,padx = 600,pady=100,image=self.img)
		self.image.grid(row=1,column=0,sticky=N)
		self.firstbutton = Button(master,text='Load line',command=self.Load_line)
		self.firstbutton.grid(row=2,column=0,padx = 300,sticky=N)
		self.secondbutton = Button(master,text='Load character',command=self.Load_character)
		self.secondbutton.grid(row=3,column=0,padx = 300,sticky=N)
		self.label = Label(master,text='Bản chất của thành công',font=("Arial", 34))
		self.label.grid(row=4,column=0,pady=(50,50))

		self.craft = Character_detect()

	def Load_line(self):
		r"""
            Load an image from local directory 
        """
		self.path = filedialog.askopenfilename(initialdir = os.path.dirname(os.path.abspath(__file__)),title = "Select file",
			                                    filetypes = (("all files","*.*"),("png files","*.png"),("jpeg files","*.jpg")))
		if len(self.path) == 0:
			return
		self.bbox = self.craft.detect(self.path)
		head,tail = osp.split(self.path)
		self.img = cv.cvtColor(cv.imread("Detect_result/res_" + tail[:-4] + '.jpg'),cv.COLOR_BGR2RGB)
		self.img = ImageTk.PhotoImage(Image.fromarray(self.img).resize((500,125),Image.ANTIALIAS))
		self.image.configure(image=self.img)
		self.result_str = ''
		self.img_ = cv.cvtColor(cv.imread(self.path),cv.COLOR_BGR2RGB)
		for i in range(len(self.bbox)):
			if i and self.bbox[i][0] > self.bbox[i-1][2] + 5:
				self.result_str = self.result_str + ' '
			char = Predict(self.img_[self.bbox[i][1]:self.bbox[i][3]+1,self.bbox[i][0]:self.bbox[i][2]+1])
			#print(char)
			self.result_str = self.result_str + char
		
		self.label.configure(text=self.result_str)

	def Load_character(self):
		r"""
            Load an image from local directory 
        """
		self.path = filedialog.askopenfilename(initialdir = os.path.dirname(os.path.abspath(__file__)),title = "Select file",
			                                    filetypes = (("all files","*.*"),("png files","*.png"),("jpeg files","*.jpg")))
		if len(self.path) == 0:
			return
		self.img = cv.cvtColor(cv.imread(self.path),cv.COLOR_BGR2RGB)
		self.result_str = Predict(self.img)
		self.img = ImageTk.PhotoImage(Image.fromarray(self.img).resize((500,125),Image.ANTIALIAS))
		self.image.configure(image=self.img)
		
		
		self.label.configure(text=self.result_str)

	def Run(self):
		window_height = Application.HEIGHT
		window_width = Application.WIDTH
		screen_width = self.master.winfo_screenwidth()
		screen_height = self.master.winfo_screenheight()
		x_cordinate = int((screen_width/2) - (window_width/2))
		y_cordinate = int((screen_height/2) - (window_height/2))
		self.master.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))
		self.master.mainloop()

	def Close(self):
		self.master.on_close()

	def Transform(self):
		t = Algorithm()
		img = t.Color_space_convert(img=self.image_executer.pq[self.image_executer.it][0], src_cs = 'RGB', dst_cs = 'GRAY')
		self.image_executer.Transform(self.image_executer.pq[self.image_executer.it][0],img,'Gray scale')

rootofroot = NewRoot()
root = MyMain(rootofroot)
a = Application(root)
a.Run()