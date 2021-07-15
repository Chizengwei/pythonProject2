import os
import shutil
import tkinter as tk
import tkinter.messagebox
from tkinter import ttk


def get_target_path(name):
    path = os.path.join(os.getcwd(), name + '.zip')
    return path
    pass

def luncher():

    pass

def packer(path, out, name):
    outpath = os.path.join(out, name + '.zip')
    window = tk.Tk()
    s1 = ttk.Style()
    s1.theme_use('winnative')
    window.title('主窗口')
    # window.resizable(False,False)
    # window.geometry = ('300*200-5+40')
    # window.minsize(False,False)
    # window.attributes('-alpha',0)

    # thestate=window.state()
    # print(thestate)
    window.state('iconic')

    if os.path.exists(get_target_path(name)):
        yes = tk.messagebox.askokcancel('对话框', message='工作目录存在此文件！\n要删除此文件吗？？')
        if yes:
            os.remove(get_target_path(name))
            print('文件已删除！')
    elif os.path.exists(outpath):
        yes = tk.messagebox.askokcancel('对话框', message='输出目录存在此文件！\n要重新覆盖吗？？')
        if yes:
            getpath = shutil.make_archive(name, format='zip', root_dir=path)
            shutil.copy(getpath, outpath)
            os.remove(getpath)
            print('文件已覆盖')
        else:
            no = tk.messagebox.askokcancel('对话框', message='删除？？')
            if no:
                os.remove(outpath)
        pass

    else:
        todo=tk.messagebox.askokcancel('对话框', message='要打包zip文件吗？')
        if todo:
            getpath = shutil.make_archive(name, format='zip', root_dir=path)
            print('文件打包完成')
            tk.messagebox.askquestion('对话框', message='文件打包完成！')
            shutil.copy(getpath, outpath)
            os.remove(getpath)


source = 'C:\\Users\\Administrator\\PycharmProjects\\pythonProject2\\addons'
output_dir = 'C:\\Users\\Administrator\\Desktop'

if __name__ == '__main__':
    packer(source, output_dir,'MXXw')
# if not os.path.exi