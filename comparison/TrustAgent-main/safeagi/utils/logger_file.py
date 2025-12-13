import os
from datetime import datetime, date
import sys

class Logger(object):
    def __init__(self, log_path, on=True, enable_file_output=False):  # 添加enable_file_output参数
        self.log_path = log_path
        self.on = on
        self.enable_file_output = enable_file_output  # 保存文件输出标志
        
        if self.on and self.enable_file_output:  # 只有开启文件输出时才检查文件
            while os.path.isfile(self.log_path):
                self.log_path += "+"

    def log(self, string, newline=True):
        if self.on:
            # 只有开启文件输出时才写入文件
            if self.enable_file_output:
                with open(self.log_path, "a") as logf:
                    today = date.today()
                    today_date = today.strftime("%m/%d/%Y")
                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    string_with_time = today_date + ", " + current_time + ": " + string
                    logf.write(string_with_time)
                    if newline:
                        logf.write("\n")
            
            # 总是打印到控制台
            sys.stdout.write(string)
            if newline:
                sys.stdout.write("\n")
            sys.stdout.flush()