import sys
import logging

def error_message_detail(error: Exception, error_detai: sys) -> str:
    """
    Extracts error info inluding file name, line number & error message.

    error: occured exception
    error_detail: sys module to get traceback details
    
    return -> a formatted error message string
    """

    _, _, exc_tb = error_detai.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename

    line_num = exc_tb.tb_lineno
    error_msg = f'Error occured in python script: [{file_name}] at line number [{line_num}]: {str(error)}'

    logging.error(error_msg)

    return error_msg

class MyException(Exception):
    """
    handles errors in US visa application
    """

    def __init__(self, error_msg: str, error_detail: sys):
        super().__init__(error_msg)

        self.erro_msg = error_message_detail(error_msg, error_detail)

    def __str__(self) -> str:
        return self.erro_msg